import argparse
import os
import shlex
import time
from pathlib import Path

# DGL 依赖的 outdated 包会启动后台联网检查线程，容易和 DGL/PyTorch 原生算子产生环境级崩溃。
os.environ.setdefault("OUTDATED_IGNORE", "1")
os.environ.setdefault("DGLBACKEND", "pytorch")

import numpy as np
import torch

from data import load_dataset
from model import BernsteinBinaryClassifier
from utils import (
    PriorMemory,
    PrototypeMemory,
    apply_logit_adjustment,
    build_train_targets,
    clean_anchor_stats,
    compute_class_weights,
    evaluate_binary,
    get_best_f1,
    inject_balanced_label_noise,
    label_to_index,
    parse_index_list,
    select_clean_anchors,
    set_seed,
    split_masks,
    estimate_corrected_prior,
    weighted_soft_cross_entropy,
)


def clean_anchor_branch_loss(
    logits_low: torch.Tensor,
    logits_high: torch.Tensor,
    labels_obs: torch.Tensor,
    anchor_mask: torch.Tensor,
    sample_size: int,
) -> torch.Tensor:
    # 从两类 clean anchors 中平衡随机抽样，同一批干净节点同时监督低频和高频分支。
    label_idx = label_to_index(labels_obs)
    clean0 = (anchor_mask & (label_idx == 0)).nonzero(as_tuple=False).squeeze(1)
    clean1 = (anchor_mask & (label_idx == 1)).nonzero(as_tuple=False).squeeze(1)
    per_class = min(clean0.numel(), clean1.numel(), sample_size)
    if per_class <= 0:
        return (logits_low.sum() + logits_high.sum()) * 0.0

    pick0 = torch.randperm(clean0.numel(), device=clean0.device)[:per_class]
    pick1 = torch.randperm(clean1.numel(), device=clean1.device)[:per_class]
    branch_idx = torch.cat([clean0[pick0], clean1[pick1]], dim=0)
    branch_targets = labels_obs[branch_idx]

    loss_low = weighted_soft_cross_entropy(logits_low[branch_idx], branch_targets)
    loss_high = weighted_soft_cross_entropy(logits_high[branch_idx], branch_targets)
    return 0.5 * (loss_low + loss_high)


def train_one_run(args, graph, run_id: int) -> dict:
    # 每个 run 使用独立 seed，使划分、噪声注入、初始化和抽样都能形成独立重复实验。
    seed = args.seed + run_id
    set_seed(seed)

    # 先划分 train / val / test，再只对 train_mask 内的标签注入噪声。
    train_mask, val_mask, test_mask = split_masks(
        graph.ndata["label"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=seed,
        dataset_name=args.dataset,
        amazon_start=args.amazon_start,
    )
    labels_obs_cpu, noise_mask_cpu = inject_balanced_label_noise(
        graph.ndata["label"],
        train_mask,
        noise_ratio=args.noise_ratio,
        seed=seed,
    )

    # 图和张量移动到指定设备；labels_true 只用于验证/测试和噪声统计。
    device = torch.device(args.device)
    graph = graph.to(device)
    features = graph.ndata["feature"]
    labels_true = graph.ndata["label"]
    labels_obs = labels_obs_cpu.to(device)
    noise_mask = noise_mask_cpu.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    if args.noise_ratio > 0:
        print(f"Injected train label noise: {int(noise_mask.sum().item())} flipped nodes")
    class_weight = compute_class_weights(
        labels_obs,
        train_mask,
        mode=args.class_weight_mode,
        beta=args.class_weight_beta,
        power=args.class_weight_power,
    )
    if class_weight is not None:
        print(f"class weights: {[round(item, 4) for item in class_weight.detach().cpu().tolist()]}")

    # Bernstein 主模型：低频、高频和全频三路输出。
    model = BernsteinBinaryClassifier(
        in_feats=features.shape[1],
        hidden_dim=args.hidden_dim,
        order=args.order,
        dropout=args.dropout,
        low_indices=parse_index_list(args.low_indices),
        high_indices=parse_index_list(args.high_indices),
    ).to(device)

    # 原型记忆只通过 clean anchors 的 z 表示做 EMA 更新，不参与反向传播。
    prototypes = PrototypeMemory(num_classes=2, hidden_dim=args.hidden_dim, device=device)
    # 修正先验用于 RCP-LA，只通过训练集的 y_tilde / clean anchors 做 EMA 更新。
    prior_memory = PriorMemory(num_classes=2, device=device)
    main_parameters = [
        *model.encoder.parameters(),
        *model.main_head.parameters(),
    ]
    optimizer = torch.optim.Adam(main_parameters, lr=args.lr, weight_decay=args.weight_decay)
    branch_parameters = [
        *model.low_proj.parameters(),
        *model.high_proj.parameters(),
        *model.low_head.parameters(),
        *model.high_head.parameters(),
    ]
    branch_optimizer = torch.optim.Adam(
        branch_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_val_f1 = -1.0
    best_test = None
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        out = model(graph, features)
        logits = out["logits"]
        if not torch.isfinite(logits).all():
            report_numerics("non-finite logits", out, labels_obs, train_mask)
            raise FloatingPointError("model logits contain NaN/Inf before loss computation")

        # warm-up 结束后才启用 clean anchors、原型和软标签纠正。
        loss_mask = train_mask
        sample_weight = None
        use_clean = bool(args.use_clean_anchor) and epoch >= args.warmup_epochs
        if use_clean:
            # clean anchors 每个 epoch 动态重选，不是固定集合。
            # 固定用观测标签筛选 clean anchors，避免软标签反过来污染 anchor 集合。
            anchor_source = labels_obs
            anchor_mask = select_clean_anchors(
                out["p_L"],
                out["p_H"],
                out["p_A"],
                anchor_source,
                train_mask,
                clean_ratio=args.clean_ratio,
                gamma=args.gamma,
                clean_ratio_0=args.clean_ratio_0,
                clean_ratio_1=args.clean_ratio_1,
            )
            prototypes.update(
                out["z"],
                anchor_source,
                anchor_mask,
                momentum=args.proto_momentum,
                min_count=args.min_proto_per_class,
            )
            q_proto = prototypes.probability(out["z"].detach(), eta=args.proto_eta)
            if args.target_mode == "soft":
                train_targets = build_train_targets(
                    labels_obs,
                    q_proto,
                    beta=args.correction_beta,
                )
                train_targets = torch.where(anchor_mask.unsqueeze(1), labels_obs, train_targets)
            elif args.target_mode == "anchor_weighted":
                train_targets = labels_obs
                sample_weight = torch.ones_like(out["p_A"])
                sample_weight = torch.where(anchor_mask, sample_weight * args.anchor_loss_weight, sample_weight)
            elif args.target_mode == "anchor_only":
                train_targets = labels_obs
                loss_mask = train_mask & anchor_mask
                if not bool(loss_mask.any()):
                    loss_mask = train_mask
            else:
                train_targets = labels_obs
            if args.log_clean_stats and (epoch % args.clean_log_every == 0 or epoch == args.warmup_epochs):
                stats = clean_anchor_stats(anchor_mask, labels_true, labels_obs, noise_mask)
                print(
                    f"Clean anchors epoch={epoch} "
                    f"total={stats['total']} precision={stats['precision']:.4f} "
                    f"c0={stats['class0_count']}/{stats['class0_true_clean']} "
                    f"c1={stats['class1_count']}/{stats['class1_true_clean']} "
                    f"proto_ready={prototypes.is_ready}"
                )
        else:
            # warm-up 或关闭 clean-anchor 时，直接使用观测标签训练。
            train_targets = labels_obs

        # corrected prior 只用训练集信息估计，但 RCP-LA 会作用到所有训练节点的 loss。
        use_rcp_la = bool(args.use_rcp_la) and use_clean and epoch >= args.la_start_epoch
        if use_rcp_la:
            prior_est = estimate_corrected_prior(
                train_targets,
                train_mask,
                anchor_mask,
                clean_weight=args.prior_clean_weight,
                eps=args.prior_eps,
            )
            corrected_prior = prior_memory.update(prior_est, momentum=args.prior_momentum)
        else:
            corrected_prior = None

        # 主分类分支始终在所有训练节点上训练，只是 targets 可由 y_obs 切换成 y_tilde。
        train_logits = apply_logit_adjustment(logits, corrected_prior, tau=args.la_tau)
        loss_cls = weighted_soft_cross_entropy(
            train_logits[loss_mask],
            train_targets[loss_mask],
            class_weight=class_weight,
            sample_weight=sample_weight[loss_mask] if sample_weight is not None else None,
        )
        loss_branch = logits.new_tensor(0.0)
        if use_clean and bool(args.use_clean_branch):
            branch_logits_low, branch_logits_high = model.branch_logits_from_filters(out["Z_all"].detach())
            loss_branch = clean_anchor_branch_loss(
                branch_logits_low,
                branch_logits_high,
                anchor_source,
                anchor_mask,
                sample_size=args.branch_sample_size,
            )
            if not torch.isfinite(loss_branch):
                print(f"loss_branch={loss_branch.detach().item()}")
                report_numerics("non-finite branch loss", out, labels_obs, train_mask)
                raise FloatingPointError("clean anchor branch loss became NaN/Inf")

        L_total = loss_cls + loss_branch
        if not torch.isfinite(L_total):
            print(f"loss_cls={loss_cls.detach().item()} loss_branch={loss_branch.detach().item()}")
            report_numerics("non-finite training loss", out, labels_obs, train_mask)
            raise FloatingPointError("training loss became NaN/Inf")

        optimizer.zero_grad()
        branch_optimizer.zero_grad()
        L_total.backward()
        if args.grad_clip > 0:
            # 梯度裁剪防止某一轮更新过大导致数值不稳定。
            torch.nn.utils.clip_grad_norm_(main_parameters, args.grad_clip)
        optimizer.step()

        if use_clean and bool(args.use_clean_branch):
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(branch_parameters, args.grad_clip)
            branch_optimizer.step()

        # 验证/测试始终使用真实标签 labels_true；训练噪声不能进入评估。
        model.eval()
        with torch.no_grad():
            out = model(graph, features)
            val_f1, threshold = get_best_f1(labels_true[val_mask], out["p_A"][val_mask])
            test_metrics = evaluate_binary(labels_true, out["p_A"], test_mask, threshold)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test = test_metrics

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            prior_text = ""
            if corrected_prior is not None:
                prior_cpu = corrected_prior.detach().cpu().tolist()
                prior_text = f" prior=[{prior_cpu[0]:.4f},{prior_cpu[1]:.4f}]"
            branch_text = ""
            if bool(args.use_clean_branch):
                branch_text = f" branch={loss_branch.item():.4f}"
            print(
                f"Epoch {epoch:03d} "
                f"L_total={L_total.item():.4f} "
                f"ce={loss_cls.item():.4f} "
                f"{branch_text} "
                f"val_mf1={val_f1:.4f} "
                f"best_val_mf1={best_val_f1:.4f}"
                f"{prior_text}"
            )

    elapsed = time.time() - start
    print(
        "Test "
        f"ACC {best_test['acc'] * 100:.2f} "
        f"REC {best_test['recall'] * 100:.2f} "
        f"PRE {best_test['precision'] * 100:.2f} "
        f"MF1 {best_test['macro_f1'] * 100:.2f} "
        f"AUC {best_test['auc'] * 100:.2f} "
        f"TH {best_test['threshold']:.2f} "
        f"time {elapsed:.1f}s"
    )
    return best_test


def report_numerics(name: str, out: dict, labels: torch.Tensor, train_mask: torch.Tensor) -> None:
    # 出现 NaN/Inf 时打印关键张量范围，方便定位数值问题。
    print(f"[numeric debug] {name}")
    print(f"[numeric debug] train samples={train_mask.sum().item()}")
    train_labels = label_to_index(labels[train_mask])
    if train_labels.numel() > 0:
        print(f"[numeric debug] train label counts={torch.bincount(train_labels.detach().cpu()).tolist()}")
    for key in ["Z_all", "z", "z_L", "z_H", "logits", "logits_L", "logits_H", "p_A", "p_L", "p_H"]:
        value = out.get(key)
        if value is None:
            continue
        finite = torch.isfinite(value)
        finite_count = finite.sum().item()
        total = value.numel()
        if finite_count > 0:
            finite_value = value[finite]
            min_value = finite_value.min().detach().item()
            max_value = finite_value.max().detach().item()
            mean_value = finite_value.mean().detach().item()
            print(
                f"[numeric debug] {key}: finite={finite_count}/{total} "
                f"min={min_value:.6g} max={max_value:.6g} mean={mean_value:.6g}"
            )
        else:
            print(f"[numeric debug] {key}: finite=0/{total}")


def build_parser() -> argparse.ArgumentParser:
    # 命令行参数集中在这里，便于实验时调整噪声、clean anchors 和原型相关超参。
    parser = argparse.ArgumentParser(description="Bernstein multi-spectrum binary node classifier")
    parser.add_argument("--config_dir", type=str, default="config", help="directory for dataset-specific config files")
    parser.add_argument("--gpu", type=int, default=6, help="GPU id to use; change this default in main.py if needed")
    parser.add_argument("--dataset", type=str, default="amazon", choices=["yelp", "amazon", "elliptic", "tfinance"])
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--hidden_dim", "--hid_dim", dest="hidden_dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--low_indices", type=str, default=None, help="comma-separated Bernstein indices, e.g. 0,1")
    parser.add_argument("--high_indices", type=str, default=None, help="comma-separated Bernstein indices, e.g. 1,2")
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--use_cleanliness", type=int, default=1)
    parser.add_argument("--noise_ratio", type=float, default=0.3)#噪声率
    parser.add_argument("--use_clean_anchor", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--clean_ratio", type=float, default=0.2)
    parser.add_argument("--clean_ratio_0", type=float, default=0.2)
    parser.add_argument("--clean_ratio_1", type=float, default=0.7)
    parser.add_argument(
        "--target_mode",
        type=str,
        default="soft",
        choices=["soft", "observed", "anchor_weighted", "anchor_only"],
    )
    parser.add_argument("--anchor_loss_weight", type=float, default=2.0)
    parser.add_argument("--use_clean_branch", type=int, default=1)
    parser.add_argument("--branch_sample_size", type=int, default=512)
    parser.add_argument("--proto_momentum", type=float, default=0.9)
    parser.add_argument("--proto_weight", type=float, default=0.5)
    parser.add_argument("--proto_eta", type=float, default=5.0)
    parser.add_argument("--correction_beta", type=float, default=0.5)
    parser.add_argument("--minority_protect", type=float, default=0.5)
    parser.add_argument("--class_weight_mode", type=str, default="none", choices=["none", "inverse", "effective"])
    parser.add_argument("--class_weight_beta", type=float, default=0.999)
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--use_rcp_la", type=int, default=0)#不平衡
    parser.add_argument("--la_start_epoch", type=int, default=20)
    parser.add_argument("--la_tau", type=float, default=1.0)
    parser.add_argument("--prior_momentum", type=float, default=0.9)
    parser.add_argument("--prior_clean_weight", type=float, default=1.0)
    parser.add_argument("--prior_eps", type=float, default=1e-6)
    parser.add_argument("--min_proto_per_class", type=int, default=1)
    parser.add_argument("--log_clean_stats", type=int, default=0)
    parser.add_argument("--clean_log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=100)
    parser.add_argument("--runs", "--run", dest="runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--amazon_start", type=int, default=3305)
    parser.add_argument("--undirected", type=int, default=1, help="convert graph to undirected before Bernstein filtering")
    return parser


def load_dataset_config_args(dataset: str, config_dir: str) -> list[str]:
    # 根据数据集名称读取 config/<dataset>.txt，把其中参数作为该数据集的默认参数。
    config_path = Path(config_dir) / f"{dataset.lower()}.txt"
    if not config_path.exists():
        return []

    command_lines = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command_lines.append(stripped.rstrip("\\"))
    config_args = shlex.split(" ".join(command_lines))
    print(f"Loaded dataset config: {config_path}")
    return config_args


def parse_args_with_dataset_config() -> argparse.Namespace:
    # 先只解析 dataset/config_dir，再加载对应配置文件，最后让命令行参数覆盖配置文件参数。
    parser = build_parser()
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--dataset", type=str, default=parser.get_default("dataset"))
    base_parser.add_argument("--config_dir", type=str, default=parser.get_default("config_dir"))
    preliminary, _ = base_parser.parse_known_args()

    config_args = load_dataset_config_args(preliminary.dataset, preliminary.config_dir)
    return parser.parse_args(config_args + os.sys.argv[1:])


def main() -> None:
    args = parse_args_with_dataset_config()
    if args.use_clean_branch not in (0, 1):
        raise ValueError("--use_clean_branch must be 0 or 1")
    if args.branch_sample_size <= 0:
        raise ValueError("--branch_sample_size must be positive")
    # GPU 选择不写进数据集 config；需要换卡时改这里的默认值，或运行时传 --gpu。
    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    elif args.device == "auto":
        args.device = "cpu"

    print(args)
    graph = load_dataset(args.dataset, args.data_dir, undirected=bool(args.undirected))
    print(graph)
    print("feature shape:", tuple(graph.ndata["feature"].shape))
    print("label counts:", torch.bincount(label_to_index(graph.ndata["label"])).tolist())

    results = []
    for run_id in range(args.runs):
        print(f"Run {run_id + 1}/{args.runs} seed={args.seed + run_id}")
        results.append(train_one_run(args, graph, run_id))

    if args.runs > 1:
        mf1 = np.array([item["macro_f1"] for item in results])
        auc = np.array([item["auc"] for item in results])
        print(
            f"MF1 mean {mf1.mean() * 100:.2f}, std {mf1.std() * 100:.2f}; "
            f"AUC mean {auc.mean() * 100:.2f}, std {auc.std() * 100:.2f}"
        )


if __name__ == "__main__":
    main()
