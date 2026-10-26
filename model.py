import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bernstein_coefficients(order: int) -> torch.Tensor:
    """Polynomial coefficients for B_{p,K}(lambda) on lambda in [0, 2]."""
    if order < 0:
        raise ValueError("order must be non-negative")

    # theta[p, r] 表示第 p 个 Bernstein 滤波器中 L^r H 的系数。
    theta = torch.zeros(order + 1, order + 1, dtype=torch.float32)
    for p in range(order + 1):
        base = math.comb(order, p)
        for r in range(p, order + 1):
            m = r - p
            theta[p, r] = base * math.comb(order - p, m) * ((-1.0) ** m) / (2.0**r)
    return theta


def normalized_adjacency(graph, device: torch.device) -> torch.Tensor:
    """Build cached S = D^{-1/2} A D^{-1/2} as a torch sparse tensor."""
    # 不再用 DGL update_all 做传播，避免某些 DGL/PyTorch/CUDA 组合在 gspmm backward 中崩溃。
    cache_key = (str(device), graph.num_nodes(), graph.num_edges())
    cache = getattr(graph, "_bernstein_norm_adj_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    src, dst = graph.edges()
    src = src.to(device)
    dst = dst.to(device)
    deg_inv_sqrt = graph.in_degrees().float().clamp(min=1).pow(-0.5).to(device)
    values = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([dst, src], dim=0)
    adj = torch.sparse_coo_tensor(
        indices,
        values,
        size=(graph.num_nodes(), graph.num_nodes()),
        device=device,
    ).coalesce()

    cache[cache_key] = adj
    setattr(graph, "_bernstein_norm_adj_cache", cache)
    return adj


def normalized_laplacian(graph, feat: torch.Tensor) -> torch.Tensor:
    """Apply L = I - D^{-1/2} A D^{-1/2} to node features."""
    # smoothed = D^{-1/2} A D^{-1/2} feat；A 已在 data.py 中无向化、去重并加自环。
    smoothed = torch.sparse.mm(normalized_adjacency(graph, feat.device), feat)
    # 归一化拉普拉斯传播：L feat = feat - smoothed。
    return feat - smoothed


def default_frequency_indices(order: int) -> Tuple[List[int], List[int]]:
    # 滤波器数量是 order + 1。数量为奇数时，中间滤波器同时进入低频和高频。
    num_filters = order + 1
    mid = num_filters // 2
    if num_filters % 2 == 0:
        low = list(range(0, mid))
        high = list(range(mid, num_filters))
    else:
        low = list(range(0, mid + 1))
        high = list(range(mid, num_filters))
    return low, high


def _validate_indices(indices: Iterable[int], order: int, name: str) -> List[int]:
    # 检查用户手动传入的低/高频滤波器索引是否合法。
    values = list(indices)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for idx in values:
        if idx < 0 or idx > order:
            raise ValueError(f"{name} contains {idx}, but valid range is [0, {order}]")
    return values


def _projection(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    # 用于把拼接后的频谱表示降回 hidden_dim。
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class BernsteinEncoder(nn.Module):
    """Bernstein multi-spectrum encoder."""

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        order: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.order = order
        # 原始节点特征 X 先经过 MLP 得到隐藏表示 H。
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 全部 Bernstein 滤波器输出拼接后，经过独立 MLP 得到主表示 z。
        self.proj = _projection((order + 1) * hidden_dim, hidden_dim, dropout)
        # Bernstein 系数不是可训练参数，但需要跟随模型移动到 CPU/GPU。
        self.register_buffer("theta", bernstein_coefficients(order))

    def forward(self, graph, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_encoder(feat)

        # 先共享计算 H, LH, L^2H, ..., L^K H。
        powers = [h]
        current = h
        for _ in range(self.order):
            current = normalized_laplacian(graph, current)
            powers.append(current)

        # powers: [K+1, N, d]；z_all: [N, K+1, d]，包含每个节点的所有滤波器响应。
        powers = torch.stack(powers, dim=0)
        z_all = torch.einsum("pr,rnd->npd", self.theta.to(feat.device), powers)
        # 全频分支：把 Z_0...Z_K 拼接，再通过 MLP 得到最终主表示 z。
        z = self.proj(z_all.reshape(z_all.shape[0], -1))
        return z_all, z


class BernsteinBinaryClassifier(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        order: int = 2,
        dropout: float = 0.3,
        low_indices: Optional[Iterable[int]] = None,
        high_indices: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        if low_indices is None or high_indices is None:
            default_low, default_high = default_frequency_indices(order)
            low_indices = default_low if low_indices is None else low_indices
            high_indices = default_high if high_indices is None else high_indices

        self.low_indices = _validate_indices(low_indices, order, "low_indices")
        self.high_indices = _validate_indices(high_indices, order, "high_indices")

        self.encoder = BernsteinEncoder(in_feats, hidden_dim, order, dropout)
        # 低频和高频分支分别使用自己的 MLP，不和全频主分支共用投影层。
        self.low_proj = _projection(len(self.low_indices) * hidden_dim, hidden_dim, dropout)
        self.high_proj = _projection(len(self.high_indices) * hidden_dim, hidden_dim, dropout)
        self.low_head = nn.Linear(hidden_dim, 2)
        self.high_head = nn.Linear(hidden_dim, 2)
        self.main_head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _project_branch(z_all: torch.Tensor, indices: List[int], projection: nn.Module) -> torch.Tensor:
        return projection(z_all[:, indices, :].reshape(z_all.shape[0], -1))

    def forward(self, graph, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_all, z = self.encoder(graph, feat)
        # 低频/高频分支先取对应滤波器，再拼接成 [N, len(indices)*d]。
        z_low = self._project_branch(z_all, self.low_indices, self.low_proj)
        z_high = self._project_branch(z_all, self.high_indices, self.high_proj)

        # 三路预测：低频 p_L，高频 p_H，全频主预测 p_A。
        logits_low = self.low_head(z_low)
        logits_high = self.high_head(z_high)
        logits = self.main_head(z)

        return {
            "Z_all": z_all,
            "z": z,
            "z_L": z_low,
            "z_H": z_high,
            "logits": logits,
            "logits_L": logits_low,
            "logits_H": logits_high,
            "p_A": F.softmax(logits, dim=1)[:, 1],
            "p_L": F.softmax(logits_low, dim=1)[:, 1],
            "p_H": F.softmax(logits_high, dim=1)[:, 1],
        }

    def branch_logits_from_filters(self, z_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 单独训练低/高频证据分支时使用 detach 后的 z_all，
        # 只更新 low/high 的 projection 和 head，不反向影响共享 encoder。
        z_low = self._project_branch(z_all, self.low_indices, self.low_proj)
        z_high = self._project_branch(z_all, self.high_indices, self.high_proj)
        return self.low_head(z_low), self.high_head(z_high)
