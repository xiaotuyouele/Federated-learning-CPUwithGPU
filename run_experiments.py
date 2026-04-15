import numpy as np
import torch
import random
import copy
import os
import csv

from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
from models.Nets import MLP
from utils.options import args_parser
from torch.utils.data import TensorDataset


# ===== 保存路径 =====
SAVE_DIR = r"D:\GraduationDocuments"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_once(alpha, run_id, args):
    print(f"\n--- Run {run_id} (alpha={alpha}) ---")

    set_seed(2026 + run_id)

    # ===== synthetic 数据 =====
    num_samples = 10000
    input_size = args.input_size
    num_classes = args.num_classes
    num_users = args.num_users

    X = torch.randn(num_samples, input_size)
    W = torch.randn(input_size, num_classes)
    logits = X @ W
    y = torch.argmax(logits, dim=1)

    dataset_train = TensorDataset(X, y)
    dataset_test = TensorDataset(X, y)

    # ===== non-IID 分配 =====
    dict_users = {i: [] for i in range(num_users)}

    labels = y.numpy()
    idxs = np.arange(num_samples)
    np.random.shuffle(idxs)

    # 保底
    min_samples = 10
    for i in range(num_users):
        dict_users[i].extend(idxs[i * min_samples:(i + 1) * min_samples])

    # 剩余
    remaining_idxs = idxs[num_users * min_samples:]
    remaining_labels = labels[remaining_idxs]

    idxs_by_class = [
        remaining_idxs[remaining_labels == i] for i in range(num_classes)
    ]

    # Dirichlet
    for c in range(num_classes):
        idx_c = idxs_by_class[c]
        if len(idx_c) == 0:
            continue

        np.random.shuffle(idx_c)

        proportions = np.random.dirichlet([alpha] * num_users)
        proportions = np.clip(proportions, 1e-2, None)
        proportions /= proportions.sum()

        proportions = (proportions * len(idx_c)).astype(int)

        # 修正误差
        diff = len(idx_c) - np.sum(proportions)
        for i in range(diff):
            proportions[i % num_users] += 1

        start = 0
        for user in range(num_users):
            num = proportions[user]
            if num > 0:
                dict_users[user] += idx_c[start:start + num].tolist()
            start += num

    # ✅ 检查数据是否完整
    total = sum(len(dict_users[i]) for i in range(num_users))
    print(f"Total assigned samples: {total}")

    # ===== 模型 =====
    net_glob = MLP(
        dim_in=input_size,
        dim_hidden=200,
        dim_out=num_classes
    ).to(args.device)

    net_glob.train()
    w_glob = net_glob.state_dict()

    # ===== 联邦训练 =====
    for epoch in range(args.epochs):
        w_locals = []

        m = max(int(args.frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))

        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

    # ===== 测试 =====
    net_glob.eval()
    acc_test, _ = test_img(net_glob, dataset_test, args)

    return acc_test


def main():
    args = args_parser()

    # ✅ 强制使用 CPU（你当前阶段推荐）
    args.device = torch.device('cpu')

    print(f"Using device: {args.device}")

    alphas = [0.01, 0.1, 0.5, 1, 10]
    runs = 5

    results = {}

    csv_path = os.path.join(SAVE_DIR, "results.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # 表头
        writer.writerow(["alpha", "run", "accuracy"])

        for alpha in alphas:
            print(f"\n===== Alpha = {alpha} =====")

            acc_list = []

            for r in range(runs):
                acc = run_once(alpha, r, args)
                print(f"Accuracy: {acc:.2f}")

                writer.writerow([alpha, r, acc])
                acc_list.append(acc)

            mean = np.mean(acc_list)
            std = np.std(acc_list)

            results[alpha] = (mean, std)

            # ✅ 写入统计结果（论文用）
            writer.writerow([alpha, "mean", mean])
            writer.writerow([alpha, "std", std])

            print(f"Alpha {alpha} -> Mean: {mean:.2f}, Std: {std:.2f}")

    # 保存 npy
    np.save(os.path.join(SAVE_DIR, "results.npy"), results)

    print(f"\n 全部完成！结果已保存到: {SAVE_DIR}")


if __name__ == "__main__":
    main()