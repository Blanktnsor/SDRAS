from pathlib import Path
import pickle
import warnings

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import FraudAmazonDataset, FraudYelpDataset


def _normalize_minmax(feature: torch.Tensor) -> torch.Tensor:
    # 沿用原加载代码的 min-max 特征归一化，避免不同量纲特征主导训练。
    feature = feature.float()
    min_value = feature.min(dim=0, keepdim=True).values
    max_value = feature.max(dim=0, keepdim=True).values
    denom = (max_value - min_value).clamp_min(1e-12)
    return (feature - min_value) / denom


def _binarize_labels(label) -> torch.Tensor:
    # 固定按原代码的二分类规则：{0,1} 保持，{-1,1} 转成 {0,1}，其他二值取较大值为 1。
    label_np = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else np.asarray(label)
    label_np = label_np.reshape(-1)
    unique = np.unique(label_np)
    unique_set = set(unique.tolist())
    if unique_set == {0, 1}:
        return torch.from_numpy(label_np.astype(np.int64)).long()
    if unique_set == {-1, 1}:
        return torch.from_numpy((label_np == 1).astype(np.int64)).long()
    if len(unique) == 2:
        return torch.from_numpy((label_np == unique.max()).astype(np.int64)).long()
    raise ValueError(f"Expect binary labels but got unique values: {unique[:10]}")


def _resolve_dat_path(name: str, data_dir: str) -> Path:
    # 本地数据固定读取 data_dir/<dataset>.dat，不再兼容子目录或 .pkl 路径。
    root = Path(data_dir)
    path = root / f"{name}.dat"
    if path.exists():
        return path
    raise FileNotFoundError(f"cannot find {name} dataset file: {path}")


def _extract_pickle_components(obj, dataset_name: str):
    # 与原先可运行代码保持一致：dict 里读 x/y/edge_index；或者对象属性 x/y/edge_index。
    if isinstance(obj, dict):
        x = obj.get("x", obj.get("node_features"))
        y = obj.get("y", obj.get("node_labels"))
        edge_index = obj.get("edge_index")
        if edge_index is None and "edges" in obj:
            src_list, dst_list = [], []
            for value in obj["edges"].values():
                edge_block = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                if edge_block.dim() != 2 or edge_block.size(0) != 2:
                    raise ValueError(f"Unsupported edge block shape in {dataset_name}: {tuple(edge_block.shape)}")
                src_list.append(edge_block[0].long())
                dst_list.append(edge_block[1].long())
            if src_list:
                edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
        return x, y, edge_index

    return getattr(obj, "x", None), getattr(obj, "y", None), getattr(obj, "edge_index", None)


def _load_dat_graph(name: str, data_dir: str):
    path = _resolve_dat_path(name, data_dir)
    with path.open("rb") as file:
        # 这些本地 .dat 文件里包含了 torch.Tensor。
        # 新版本 PyTorch 在反序列化 tensor storage 时会打印一个
        # “torch.load(weights_only=False)” 的 FutureWarning。
        # 这里的数据文件是用户本地自有数据，继续按原行为加载即可，
        # 仅把这条已知 warning 定向屏蔽，避免每次运行都刷屏。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*weights_only=False.*",
                category=FutureWarning,
            )
            obj = pickle.load(file)

    x, y, edge_index = _extract_pickle_components(obj, name)
    if x is None or y is None or edge_index is None:
        raise ValueError(
            f"{name} file missing one of x/y/edge_index. "
            f"Got types: x={type(x)}, y={type(y)}, edge_index={type(edge_index)}"
        )

    feature = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    label = _binarize_labels(y)
    edge_index = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index)
    edge_index = edge_index.long()
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{name} edge_index must be [2, E], got {tuple(edge_index.shape)}")

    num_nodes = int(feature.shape[0])
    src = edge_index[0].view(-1)
    dst = edge_index[1].view(-1)
    if src.numel() == 0:
        raise ValueError(f"{name} edge_index is empty")
    if int(src.min()) < 0 or int(dst.min()) < 0 or int(src.max()) >= num_nodes or int(dst.max()) >= num_nodes:
        raise ValueError(
            f"{name} edge_index contains out-of-range ids: "
            f"src in [{int(src.min())}, {int(src.max())}], "
            f"dst in [{int(dst.min())}, {int(dst.max())}], num_nodes={num_nodes}"
        )

    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    graph.ndata["feature"] = _normalize_minmax(feature)
    graph.ndata["label"] = F.one_hot(label, num_classes=2).float()
    return graph


def _make_homogeneous(graph, undirected: bool = True):
    # 如果是异构图或多关系图，先合并成同构图；节点特征和标签会被保留下来。
    if len(getattr(graph, "ntypes", [])) > 1 or len(getattr(graph, "etypes", [])) > 1:
        graph = dgl.to_homogeneous(
            graph,
            ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
        )

    # 先去掉原始自环，再转无向、去重，最后统一重新加自环。
    graph = dgl.remove_self_loop(graph)
    if undirected:
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.to_simple(graph, copy_ndata=True)
    return dgl.add_self_loop(graph)


def _prepare_node_data(graph):
    # 当前代码默认每个节点必须有 feature 和 label。
    if "feature" not in graph.ndata:
        raise KeyError("graph.ndata must contain 'feature'")
    if "label" not in graph.ndata:
        raise KeyError("graph.ndata must contain 'label'")

    # 清理特征中的 NaN/Inf，防止图传播后数值污染整个图。
    feature = graph.ndata["feature"].float()
    graph.ndata["feature"] = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

    # 标签统一转成 [N, 2] 的 one-hot/soft-label 形式，方便后续做软标签纠正。
    label = graph.ndata["label"]
    if label.dim() == 1:
        label = F.one_hot(label.long(), num_classes=2).float()
    elif label.dim() == 2 and label.shape[1] == 1:
        label = F.one_hot(label.squeeze(1).long(), num_classes=2).float()
    elif label.dim() == 2 and label.shape[1] == 2:
        label = label.float()
    else:
        raise ValueError("binary labels must be class ids [N], [N, 1], or one-hot/soft labels [N, 2]")

    graph.ndata["label"] = label
    return graph


def load_dataset(name: str, data_dir: str = "dataset", undirected: bool = True):
    # 统一的数据加载入口。返回的图已经完成：同构化、无向化、去重、加自环和标签格式整理。
    name = name.lower()
    if name == "yelp":
        graph = FraudYelpDataset()[0]
        graph = _make_homogeneous(graph, undirected=undirected)
    elif name == "amazon":
        graph = FraudAmazonDataset()[0]
        graph = _make_homogeneous(graph, undirected=undirected)
    elif name in {"elliptic", "tfinance"}:
        graph = _load_dat_graph(name, data_dir)
        graph = _make_homogeneous(graph, undirected=undirected)
    else:
        raise ValueError("dataset must be one of: yelp, amazon, elliptic, tfinance")

    graph = _prepare_node_data(graph)
    labels = graph.ndata["label"]
    # 第一版只支持二分类，所以标签必须是 [N, 2]。
    if labels.dim() != 2 or labels.shape[1] != 2:
        raise ValueError("this first version only supports binary node classification")
    return graph
