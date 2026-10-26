import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    # 固定随机种子，保证划分、噪声注入和模型初始化可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_index_list(value: Optional[str]) -> Optional[List[int]]:
    # 把命令行输入的 "0,1" 转成 [0, 1]。
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def label_to_index(labels: torch.Tensor) -> torch.Tensor:
    # one-hot/soft-label 在统计和评估时需要转回类别编号。
    if labels.dim() == 1:
        return labels.long()
    return labels.argmax(dim=1).long()


def split_masks(
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    dataset_name: str,
    amazon_start: int = 3305,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 按 train_ratio / val_ratio 分层划分训练、验证和测试节点。
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    label_idx = label_to_index(labels)
    labels_np = label_idx.detach().cpu().numpy()
    num_nodes = labels_np.shape[0]
    # 保留 BWGNN 在 Amazon 数据集上跳过前 3305 个节点的划分习惯。
    start = amazon_start if dataset_name.lower() == "amazon" and num_nodes > amazon_start else 0
    indices = np.arange(start, num_nodes)

    idx_train, idx_rest = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels_np[indices],
    )
    relative_val = val_ratio / (1.0 - train_ratio)
    idx_val, idx_test = train_test_split(
        idx_rest,
        train_size=relative_val,
        random_state=seed,
        shuffle=True,
        stratify=labels_np[idx_rest],
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    return train_mask, val_mask, test_mask


def inject_balanced_label_noise(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    noise_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 只对训练集注入标签噪声；少数类翻转 m 个，多数类也翻转 m 个，保持类别比例不变。
    if noise_ratio <= 0:
        return labels.clone(), torch.zeros(labels.shape[0], dtype=torch.bool)
    if noise_ratio > 1:
        raise ValueError("noise_ratio must be in [0, 1]")

    label_idx = label_to_index(labels).detach().cpu()
    train_idx = train_mask.detach().cpu().nonzero(as_tuple=False).squeeze(1)
    train_labels = label_idx[train_idx]
    counts = torch.bincount(train_labels, minlength=2)
    minority = int(torch.argmin(counts).item())
    majority = 1 - minority
    # m 以少数类训练节点数量为基准。
    flip_count = int(torch.floor(counts[minority].float() * noise_ratio).item())

    noisy_idx = label_idx.clone()
    noise_mask = torch.zeros(label_idx.shape[0], dtype=torch.bool)
    if flip_count > 0:
        generator = torch.Generator()
        generator.manual_seed(seed)
        minority_idx = train_idx[train_labels == minority]
        majority_idx = train_idx[train_labels == majority]
        chosen_min = minority_idx[torch.randperm(minority_idx.numel(), generator=generator)[:flip_count]]
        chosen_maj = majority_idx[torch.randperm(majority_idx.numel(), generator=generator)[:flip_count]]
        noisy_idx[chosen_min] = majority
        noisy_idx[chosen_maj] = minority
        noise_mask[chosen_min] = True
        noise_mask[chosen_maj] = True

    noisy_labels = F.one_hot(noisy_idx.to(labels.device), num_classes=2).to(labels.dtype)
    return noisy_labels, noise_mask


def weighted_soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 兼容 soft-label 的加权交叉熵：
    # class_weight 控制类别权重，用于类别不平衡实验。
    if targets.dim() == 1:
        targets = F.one_hot(targets.long(), num_classes=logits.shape[1]).float()
    else:
        targets = targets.float()
    log_probs = F.log_softmax(logits, dim=1)
    loss_per_class = -targets * log_probs
    if class_weight is not None:
        weight_cls = class_weight.detach().to(logits.device).float().clamp_min(0.0)
        loss_per_class = loss_per_class * weight_cls.view(1, -1)
    loss = loss_per_class.sum(dim=1)
    if sample_weight is not None:
        weight = sample_weight.detach().to(logits.device).float().clamp_min(0.0)
        return (loss * weight).sum() / weight.sum().clamp_min(1e-12)
    return loss.mean()


def compute_class_weights(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    mode: str,
    beta: float = 0.999,
    power: float = 1.0,
) -> Optional[torch.Tensor]:
    # 只根据训练集的观测标签计算类别权重，避免验证/测试标签信息泄漏。
    if mode == "none":
        return None
    if mode not in {"inverse", "effective"}:
        raise ValueError("class_weight_mode must be one of: none, inverse, effective")
    if power < 0:
        raise ValueError("class_weight_power must be non-negative")

    label_idx = label_to_index(labels)
    train_labels = label_idx[train_mask]
    counts = torch.bincount(train_labels, minlength=2).float().clamp_min(1.0)

    if mode == "inverse":
        # 反频率权重：少数类权重大，多数类权重小。
        weights = counts.sum() / (counts.numel() * counts)
    else:
        # Effective number 权重，比纯反频率更温和，适合先做保守实验。
        beta_tensor = torch.tensor(beta, dtype=counts.dtype, device=counts.device).clamp(0.0, 0.999999)
        effective_num = 1.0 - torch.pow(beta_tensor, counts)
        weights = (1.0 - beta_tensor) / effective_num.clamp_min(1e-12)

    if power != 1.0:
        weights = weights.pow(power)
    # 按训练集类别分布归一化，使平均训练 loss 尺度接近未加权版本。
    weights = weights * counts.sum() / (weights * counts).sum().clamp_min(1e-12)
    return weights.detach()


def binary_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 二分类熵，先 clamp 防止出现 0 * log(0) 的 NaN。
    prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
    prob = prob.clamp(eps, 1.0 - eps)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


def compute_cleanliness(
    p_l: torch.Tensor,
    p_h: torch.Tensor,
    p_a: torch.Tensor,
    gamma: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 根据低频、高频、全频三路预测计算频率一致清洁度 c_i。
    p_bar = (p_l + p_h + p_a) / 3.0
    disagreement = ((p_l - p_h).abs() + (p_l - p_a).abs() + (p_h - p_a).abs()) / 3.0
    entropy = (binary_entropy(p_l) + binary_entropy(p_h) + binary_entropy(p_a)) / (3.0 * np.log(2.0))
    cleanliness = torch.exp(-gamma * disagreement) * (1.0 - entropy)
    cleanliness = torch.nan_to_num(cleanliness, nan=0.0, posinf=1.0, neginf=0.0)
    cleanliness = cleanliness.clamp(0.0, 1.0)
    margin = (2.0 * p_bar - 1.0).abs()
    return cleanliness.detach(), disagreement.detach(), margin.detach()


def select_clean_anchors(
    p_l: torch.Tensor,
    p_h: torch.Tensor,
    p_a: torch.Tensor,
    labels_obs: torch.Tensor,
    train_mask: torch.Tensor,
    clean_ratio: float,
    gamma: float,
    clean_ratio_0: Optional[float] = None,
    clean_ratio_1: Optional[float] = None,
) -> torch.Tensor:
    # 按观测类别分别选 clean_score 最高的一部分节点。
    # 0 类 anchor 依赖低频-全频一致性；1 类 anchor 依赖高频-全频一致性。
    if clean_ratio_0 is None:
        clean_ratio_0 = clean_ratio
    if clean_ratio_1 is None:
        clean_ratio_1 = clean_ratio
    if clean_ratio_0 <= 0 and clean_ratio_1 <= 0:
        return torch.zeros_like(train_mask, dtype=torch.bool)
    if clean_ratio_0 < 0 or clean_ratio_0 > 1:
        raise ValueError("clean_ratio_0 must be in [0, 1]")
    if clean_ratio_1 < 0 or clean_ratio_1 > 1:
        raise ValueError("clean_ratio_1 must be in [0, 1]")

    score_0, score_1 = class_aware_anchor_scores(p_l, p_h, p_a, labels_obs, gamma)
    label_idx = label_to_index(labels_obs)
    anchor_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    for cls in [0, 1]:
        cls_idx = (train_mask & (label_idx == cls)).nonzero(as_tuple=False).squeeze(1)
        if cls_idx.numel() == 0:
            continue
        ratio = clean_ratio_0 if cls == 0 else clean_ratio_1
        if ratio <= 0:
            continue
        k = max(1, int(torch.ceil(torch.tensor(float(cls_idx.numel()) * ratio)).item()))
        score = score_0 if cls == 0 else score_1
        top_pos = torch.topk(score[cls_idx], k=min(k, cls_idx.numel())).indices
        anchor_mask[cls_idx[top_pos]] = True
    return anchor_mask


def class_aware_anchor_scores(
    p_l: torch.Tensor,
    p_h: torch.Tensor,
    p_a: torch.Tensor,
    labels_obs: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 软标签版本的 clean score：观测标签越不确定，对应类别的 anchor 分数越低。
    # 类 0 依赖低频-全频一致性，类 1 依赖高频-全频一致性。
    p_l = p_l.detach().clamp(0.0, 1.0)
    p_h = p_h.detach().clamp(0.0, 1.0)
    p_a = p_a.detach().clamp(0.0, 1.0)
    if labels_obs.dim() == 1:
        label_prob = F.one_hot(labels_obs.long(), num_classes=2).float().to(p_a.device)
    else:
        label_prob = labels_obs.detach().float().to(p_a.device)
    y0 = label_prob[:, 0].clamp(0.0, 1.0)
    y1 = label_prob[:, 1].clamp(0.0, 1.0)

    score_0 = y0 * (1.0 - p_l) * (1.0 - p_a) * torch.exp(-gamma * (p_l - p_a).abs())
    score_1 = y1 * p_h * p_a * torch.exp(-gamma * (p_h - p_a).abs())
    return score_0, score_1


class PrototypeMemory:
    # 原型是慢变量：不通过反向传播训练，只通过 clean anchors 做 EMA 更新。
    def __init__(self, num_classes: int, hidden_dim: int, device: torch.device) -> None:
        self.prototypes = torch.zeros(num_classes, hidden_dim, device=device)
        self.ready = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @property
    def is_ready(self) -> bool:
        return bool(self.ready.all().item())

    @torch.no_grad()
    def update(
        self,
        z: torch.Tensor,
        labels_obs: torch.Tensor,
        anchor_mask: torch.Tensor,
        momentum: float,
        min_count: int,
    ) -> None:
        # 每个类别分别用 clean anchors 的 z 均值更新原型。
        label_idx = label_to_index(labels_obs)
        for cls in range(self.prototypes.shape[0]):
            cls_mask = anchor_mask & (label_idx == cls)
            if cls_mask.sum().item() < min_count:
                continue
            proto = z[cls_mask].detach().mean(dim=0)
            if self.ready[cls]:
                self.prototypes[cls] = momentum * self.prototypes[cls] + (1.0 - momentum) * proto
            else:
                self.prototypes[cls] = proto
                self.ready[cls] = True

    def probability(self, z: torch.Tensor, eta: float) -> Optional[torch.Tensor]:
        # 根据节点表示和两个类别原型的余弦相似度，得到两类原型软标签。
        if not self.is_ready:
            return None
        z_norm = F.normalize(z, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = z_norm @ proto_norm.t()
        return F.softmax(eta * sim, dim=1)


def build_corrected_soft_labels(
    labels_obs: torch.Tensor,
    q_proto: Optional[torch.Tensor],
    beta: float,
) -> torch.Tensor:
    # 两个原型都未就绪前，保留原始观测标签；否则按公式 16 直接融合。
    labels_obs = labels_obs.detach().float()
    if q_proto is None:
        return labels_obs
    q_proto = q_proto.detach().float().clamp(0.0, 1.0)
    beta = min(max(float(beta), 0.0), 1.0)
    return (1.0 - beta) * labels_obs + beta * q_proto


# 保留训练代码使用的兼容名称；当前唯一的 target 构造逻辑在上方实现。
build_train_targets = build_corrected_soft_labels


def estimate_corrected_prior(
    targets: torch.Tensor,
    train_mask: torch.Tensor,
    anchor_mask: Optional[torch.Tensor],
    clean_weight: float,
    eps: float,
) -> torch.Tensor:
    # 用 clean anchors 和软标签估计修正先验 pi_bar，避免直接相信含噪观测标签的类别比例。
    # clean_weight 越大，clean anchors 在先验估计中的权重越高。
    if targets.dim() == 1:
        probs = F.one_hot(targets.long(), num_classes=2).float()
    else:
        probs = targets.float()
    probs = probs.detach()
    mask = train_mask.detach().to(probs.device)
    weight = torch.ones(probs.shape[0], device=probs.device)
    if anchor_mask is not None and clean_weight > 0:
        anchors = anchor_mask.detach().to(probs.device)
        weight = weight + clean_weight * anchors.float()
    weight = weight * mask.float()

    counts = (probs * weight.unsqueeze(1)).sum(dim=0)
    prior = counts + eps
    prior = prior / prior.sum().clamp_min(eps)
    return prior.detach()


class PriorMemory:
    # 修正先验是慢变量：每轮用当前 y_tilde/clean anchors 估计，再用 EMA 平滑。
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.prior = torch.full((num_classes,), 1.0 / num_classes, device=device)
        self.ready = False

    @torch.no_grad()
    def update(self, estimate: torch.Tensor, momentum: float) -> torch.Tensor:
        estimate = estimate.detach().to(self.prior.device)
        if self.ready:
            self.prior = momentum * self.prior + (1.0 - momentum) * estimate
            self.prior = self.prior / self.prior.sum().clamp_min(1e-12)
        else:
            self.prior = estimate
            self.ready = True
        return self.prior.detach()


def apply_logit_adjustment(logits: torch.Tensor, prior: Optional[torch.Tensor], tau: float) -> torch.Tensor:
    # RCP-LA：用修正先验 pi_bar 调整训练 logits；tau=0 时等价于不调整。
    # 这里用于训练 loss，对应 logit-adjusted softmax CE：logits + tau * log(prior)。
    # 若用于 post-hoc 推理校正，方向才是 logits - tau * log(prior)。
    if prior is None or tau <= 0:
        return logits
    prior = prior.detach().to(logits.device).clamp_min(1e-12)
    return logits + tau * prior.log().view(1, -1)


def clean_anchor_stats(
    anchor_mask: torch.Tensor,
    labels_true: torch.Tensor,
    labels_obs: torch.Tensor,
    noise_mask: torch.Tensor,
) -> Dict:
    # 该统计只用于人工注入噪声实验，训练时不能使用 true label 信息。
    true_idx = label_to_index(labels_true).detach().cpu()
    obs_idx = label_to_index(labels_obs).detach().cpu()
    anchors = anchor_mask.detach().cpu()
    noise = noise_mask.detach().cpu()
    stats = {"total": int(anchors.sum().item()), "true_clean": int((anchors & ~noise).sum().item())}
    for cls in [0, 1]:
        cls_anchor = anchors & (obs_idx == cls)
        stats[f"class{cls}_count"] = int(cls_anchor.sum().item())
        stats[f"class{cls}_true_clean"] = int((cls_anchor & (true_idx == obs_idx)).sum().item())
    stats["precision"] = stats["true_clean"] / stats["total"] if stats["total"] > 0 else float("nan")
    return stats


def get_best_f1(labels: torch.Tensor, positive_probs: torch.Tensor) -> Tuple[float, float]:
    # 在验证集上扫描阈值，选择 macro-F1 最好的阈值。
    labels_np = label_to_index(labels).detach().cpu().numpy()
    probs_np = positive_probs.detach().cpu().numpy()

    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.05, 0.95, 19):
        preds = (probs_np > threshold).astype(np.int64)
        score = f1_score(labels_np, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_f1, best_threshold


def evaluate_binary(labels: torch.Tensor, positive_probs: torch.Tensor, mask: torch.Tensor, threshold: float) -> Dict:
    # 使用验证集选出的阈值，在测试集上计算指标。
    label_idx = label_to_index(labels)
    labels_np = label_idx[mask].detach().cpu().numpy()
    probs_np = positive_probs[mask].detach().cpu().numpy()
    preds = (probs_np > threshold).astype(np.int64)

    metrics = {
        "acc": accuracy_score(labels_np, preds),
        "macro_f1": f1_score(labels_np, preds, average="macro", zero_division=0),
        "recall": recall_score(labels_np, preds, zero_division=0),
        "precision": precision_score(labels_np, preds, zero_division=0),
        "threshold": threshold,
    }
    metrics["auc"] = roc_auc_score(labels_np, probs_np) if np.unique(labels_np).size > 1 else float("nan")
    return metrics
