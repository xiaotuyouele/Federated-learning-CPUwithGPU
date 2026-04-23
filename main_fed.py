#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import os
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, random_split

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.test import test_img


SAVE_DIR = "/content/save"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('./save', exist_ok=True)

FAIR_DIR = "/content/drive/MyDrive/fair_experiment"


# =========================================================
# 工具函数
# =========================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trusted_torch_load(path, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def find_existing_file(base_path_without_ext, exts):
    for ext in exts:
        full_path = base_path_without_ext + ext
        if os.path.exists(full_path):
            return full_path
    return None


def load_fair_dataset_pair(experiment_seed):
    train_path = find_existing_file(
        os.path.join(FAIR_DIR, f"dataset_train_seed{experiment_seed}"),
        [".pt", ".pth"]
    )
    test_path = find_existing_file(
        os.path.join(FAIR_DIR, f"dataset_test_seed{experiment_seed}"),
        [".pt", ".pth"]
    )

    if train_path is None:
        raise FileNotFoundError(
            f"未找到 dataset_train_seed{experiment_seed}.pt/.pth，请检查 {FAIR_DIR}"
        )
    if test_path is None:
        raise FileNotFoundError(
            f"未找到 dataset_test_seed{experiment_seed}.pt/.pth，请检查 {FAIR_DIR}"
        )

    dataset_train = trusted_torch_load(train_path)
    dataset_test = trusted_torch_load(test_path)
    return dataset_train, dataset_test


def load_fair_dict_users(experiment_seed, iid=False):
    candidate_names = []
    if iid:
        candidate_names = [
            f"dict_users_iid_seed{experiment_seed}.npy",
            f"dict_users_iid_seed{experiment_seed}.pt",
            f"dict_users_iid_seed{experiment_seed}.pth"
        ]
    else:
        candidate_names = [
            f"dict_users_seed{experiment_seed}.npy",
            f"dict_users_seed{experiment_seed}.pt",
            f"dict_users_seed{experiment_seed}.pth"
        ]

    for filename in candidate_names:
        path = os.path.join(FAIR_DIR, filename)
        if os.path.exists(path):
            if path.endswith(".npy"):
                return np.load(path, allow_pickle=True).item()
            return trusted_torch_load(path)

    raise FileNotFoundError(
        f"未找到 {'dict_users_iid' if iid else 'dict_users'}_seed{experiment_seed} 文件，请检查 {FAIR_DIR}"
    )


def load_fair_client_schedule(experiment_seed):
    npy_path = os.path.join(FAIR_DIR, f"client_schedule_seed{experiment_seed}.npy")
    pt_path = os.path.join(FAIR_DIR, f"client_schedule_seed{experiment_seed}.pt")
    pth_path = os.path.join(FAIR_DIR, f"client_schedule_seed{experiment_seed}.pth")

    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True)
    elif os.path.exists(pt_path):
        return trusted_torch_load(pt_path)
    elif os.path.exists(pth_path):
        return trusted_torch_load(pth_path)
    else:
        raise FileNotFoundError(
            f"未找到 client_schedule_seed{experiment_seed}.npy/.pt/.pth，请检查 {FAIR_DIR}"
        )


def load_fair_init_state(experiment_seed, map_location=None):
    init_path = find_existing_file(
        os.path.join(FAIR_DIR, f"init_model_seed{experiment_seed}"),
        [".pt", ".pth"]
    )
    if init_path is None:
        raise FileNotFoundError(
            f"未找到 init_model_seed{experiment_seed}.pt/.pth，请检查 {FAIR_DIR}"
        )
    return trusted_torch_load(init_path, map_location=map_location)


def generate_synthetic_dataset(args, seed=0):
    set_seed(seed)

    num_samples = 10000
    input_size = args.input_size
    num_classes = args.num_classes

    centers = torch.randn(num_classes, input_size) * 0.25

    labels = torch.randint(0, num_classes, (num_samples,))
    features = centers[labels] + 3.2 * torch.randn(num_samples, input_size)
    noise_ratio = 0.15
    noisy_mask = torch.rand(num_samples) < noise_ratio
    noisy_labels = torch.randint(0, num_classes, (num_samples,))
    labels[noisy_mask] = noisy_labels[noisy_mask]

    dataset = TensorDataset(features.float(), labels.long())

    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

    return dataset_train, dataset_test


def build_dirichlet_split(dataset_train, num_users, num_classes, alpha, min_samples=10, seed=0):
    """
    为训练集构造 Dirichlet 非IID划分
    """
    set_seed(seed)

    dict_users = {i: [] for i in range(num_users)}

    idxs = np.arange(len(dataset_train))
    labels = np.array([dataset_train[i][1].item() for i in range(len(dataset_train))])

    np.random.shuffle(idxs)

    total_min_need = num_users * min_samples
    if total_min_need > len(idxs):
        raise ValueError("min_samples * num_users 超过训练集大小，请调小 min_samples 或 num_users")

    for i in range(num_users):
        dict_users[i].extend(idxs[i * min_samples:(i + 1) * min_samples].tolist())

    remaining_idxs = idxs[total_min_need:]
    remaining_labels = labels[remaining_idxs]

    idxs_by_class = [
        remaining_idxs[remaining_labels == c] for c in range(num_classes)
    ]

    for c in range(num_classes):
        idx_c = idxs_by_class[c]
        if len(idx_c) == 0:
            continue

        np.random.shuffle(idx_c)

        proportions = np.random.dirichlet([alpha] * num_users)
        proportions = proportions / proportions.sum()
        proportions = (proportions * len(idx_c)).astype(int)

        diff = len(idx_c) - np.sum(proportions)
        for i in range(diff):
            proportions[i % num_users] += 1

        start = 0
        for user in range(num_users):
            num = proportions[user]
            if num > 0:
                dict_users[user].extend(idx_c[start:start + num].tolist())
            start += num

    return dict_users


def build_iid_split(dataset_train, num_users, seed=0):
    """
    synthetic 的 IID 划分
    """
    set_seed(seed)

    idxs = np.random.permutation(len(dataset_train))
    split_size = len(dataset_train) // num_users
    dict_users = {}

    for i in range(num_users):
        start = i * split_size
        end = (i + 1) * split_size if i < num_users - 1 else len(dataset_train)
        dict_users[i] = idxs[start:end].tolist()

    return dict_users


def build_model(args, img_size, device):
    """
    根据数据集和模型类型创建模型
    """
    if args.dataset == 'synthetic':
        return MLP(
            dim_in=args.input_size,
            dim_hidden1=200,
            dim_hidden2=100,
            dim_out=args.num_classes
        ).to(device)

    if args.model == 'cnn' and args.dataset == 'cifar':
        return CNNCifar(args=args).to(device)

    if args.model == 'cnn' and args.dataset == 'mnist':
        return CNNMnist(args=args).to(device)

    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        return MLP(
            dim_in=len_in,
            dim_hidden1=200,
            dim_hidden2=100,
            dim_out=args.num_classes
        ).to(device)

    raise ValueError('Error: unrecognized model')


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


# =========================================================
# 训练函数：方向一公平协同版（优化版）
# CPU：调度 + 控制 + 记录
# GPU：本地训练 + 聚合 + 全局模型保持
# 优化点：
# 1) local_net 循环外复用
# 2) 边训练边累加，不再保存整轮 w_locals
# =========================================================
def train_federated_cooperative(args, dataset_train, dataset_test, dict_users, img_size,
                                client_schedule, experiment_seed,
                                frac=None, all_clients_flag=None, verbose=True):
    """
    方向一公平协同版（优化版）：
    - CPU：调度、流程控制、日志统计
    - GPU：本地训练、参数聚合、全局模型常驻

    公平性保证：
    1) 与 CPU / GPU 版使用同一份 fair_experiment 数据、init_model、client_schedule
    2) 相同 epochs / frac / alpha / runs / seed 规则
    3) 相同模型结构、相同本地训练逻辑、相同 Dirichlet 划分逻辑
    """

    if frac is None:
        frac = args.frac
    if all_clients_flag is None:
        all_clients_flag = args.all_clients

    cpu_device = torch.device('cpu')
    gpu_device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )

    if verbose:
        print("========== Cooperative mode (Direction 1, optimized) ==========")
        print("调度设备: CPU")
        print("本地训练设备: {}".format(gpu_device))
        print("聚合设备: {}".format(gpu_device))
        print("全局模型更新设备: {}".format(gpu_device))

    # 全局模型常驻 GPU
    net_glob = build_model(args, img_size, gpu_device)
    init_state = load_fair_init_state(experiment_seed, map_location=gpu_device)
    net_glob.load_state_dict(init_state)
    net_glob.train()

    loss_train = []
    acc_curve = []
    epoch_times = []
    agg_times = []
    local_train_times = []
    schedule_times = []
    transfer_to_gpu_times = []
    transfer_to_cpu_times = []

    time_detail = {
        "mode": "CPU_schedule_GPU_train_GPU_agg_optimized",
        "schedule_device": "CPU",
        "local_train_device": str(gpu_device),
        "agg_device": str(gpu_device),
        "global_update_device": str(gpu_device),

        "epoch_times_sec": [],
        "agg_times_sec": [],
        "local_train_times_sec": [],
        "schedule_times_sec": [],
        "transfer_to_gpu_times_sec": [],
        "transfer_to_cpu_times_sec": [],
        "selected_clients_per_epoch": [],
    }

    old_global_device = args.device
    args.device = gpu_device

    train_start_time = time.time()

    # 复用本地模型，避免每个 client 都重新 build_model
    local_net = build_model(args, img_size, gpu_device)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        loss_locals = []

        # =====================================================
        # 1) 调度（CPU）
        # =====================================================
        schedule_start_time = time.time()

        if all_clients_flag:
            idxs_users = list(range(args.num_users))
        else:
            m = max(int(frac * args.num_users), 1)
            epoch_order = list(client_schedule[epoch])
            idxs_users = epoch_order[:m]

        schedule_time = time.time() - schedule_start_time
        schedule_times.append(schedule_time)

        # =====================================================
        # 2) 全局模型已在 GPU，保留口径
        # =====================================================
        transfer_to_gpu_start = time.time()
        if gpu_device.type == "cuda":
            torch.cuda.synchronize(gpu_device)
        transfer_to_gpu_time = time.time() - transfer_to_gpu_start
        transfer_to_gpu_times.append(transfer_to_gpu_time)

        # =====================================================
        # 3) 本地训练（GPU）
        #    优化：边训练边累计，不再保存整轮 w_locals
        # =====================================================
        local_train_start_time = time.time()

        global_state_gpu = clone_state_dict(net_glob.state_dict())
        w_sum = None

        for idx in idxs_users:
            # 复用模型，只重载参数
            local_net.load_state_dict(global_state_gpu)

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=local_net)

            if w_sum is None:
                w_sum = {k: v.detach().clone() for k, v in w.items()}
            else:
                for k in w_sum.keys():
                    w_sum[k] += w[k].detach()

            loss_locals.append(loss)

        if gpu_device.type == "cuda":
            torch.cuda.synchronize(gpu_device)

        local_train_time = time.time() - local_train_start_time
        local_train_times.append(local_train_time)

        # =====================================================
        # 4) GPU 聚合
        # =====================================================
        agg_start_time = time.time()

        if w_sum is None:
            raise ValueError("本轮没有客户端上传参数，无法聚合")

        w_glob_gpu = {}
        num_selected = len(idxs_users)
        for k in w_sum.keys():
            w_glob_gpu[k] = torch.div(w_sum[k], num_selected)

        net_glob.load_state_dict(w_glob_gpu)

        if gpu_device.type == "cuda":
            torch.cuda.synchronize(gpu_device)

        agg_time = time.time() - agg_start_time
        agg_times.append(agg_time)

        # =====================================================
        # 5) GPU -> CPU 回传（方向一中间过程不回传，记极小值）
        # =====================================================
        transfer_to_cpu_start = time.time()
        if gpu_device.type == "cuda":
            torch.cuda.synchronize(gpu_device)
        transfer_to_cpu_time = time.time() - transfer_to_cpu_start
        transfer_to_cpu_times.append(transfer_to_cpu_time)

        # =====================================================
        # 6) loss / acc
        # =====================================================
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        net_glob.eval()
        acc_test, _ = test_img(net_glob, dataset_test, args)
        acc_curve.append(acc_test)
        net_glob.train()

        # =====================================================
        # 7) 本轮总时间
        # =====================================================
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        time_detail["epoch_times_sec"].append(float(epoch_time))
        time_detail["agg_times_sec"].append(float(agg_time))
        time_detail["local_train_times_sec"].append(float(local_train_time))
        time_detail["schedule_times_sec"].append(float(schedule_time))
        time_detail["transfer_to_gpu_times_sec"].append(float(transfer_to_gpu_time))
        time_detail["transfer_to_cpu_times_sec"].append(float(transfer_to_cpu_time))
        time_detail["selected_clients_per_epoch"].append(int(len(idxs_users)))

        if verbose:
            print(
                "Round {:3d}, clients {:3d}, Average loss {:.4f}, Test accuracy {:.4f}, "
                "Epoch Time {:.2f}s, Local Train {:.2f}s, Schedule {:.6f}s, "
                "CPU->GPU {:.6f}s, GPU Agg {:.4f}s, GPU->CPU {:.6f}s".format(
                    epoch + 1,
                    len(idxs_users),
                    loss_avg,
                    acc_test,
                    epoch_time,
                    local_train_time,
                    schedule_time,
                    transfer_to_gpu_time,
                    agg_time,
                    transfer_to_cpu_time
                )
            )

    total_train_time = time.time() - train_start_time
    args.device = old_global_device

    time_detail["total_train_time_sec"] = float(total_train_time)

    time_detail["total_epoch_time_sec"] = float(np.sum(epoch_times))
    time_detail["total_local_train_time_sec"] = float(np.sum(local_train_times))
    time_detail["total_schedule_time_sec"] = float(np.sum(schedule_times))
    time_detail["total_agg_time_sec"] = float(np.sum(agg_times))
    time_detail["total_transfer_to_gpu_time_sec"] = float(np.sum(transfer_to_gpu_times))
    time_detail["total_transfer_to_cpu_time_sec"] = float(np.sum(transfer_to_cpu_times))

    time_detail["avg_epoch_time_sec"] = float(np.mean(epoch_times))
    time_detail["avg_local_train_time_sec"] = float(np.mean(local_train_times))
    time_detail["avg_schedule_time_sec"] = float(np.mean(schedule_times))
    time_detail["avg_agg_time_sec"] = float(np.mean(agg_times))
    time_detail["avg_transfer_to_gpu_time_sec"] = float(np.mean(transfer_to_gpu_times))
    time_detail["avg_transfer_to_cpu_time_sec"] = float(np.mean(transfer_to_cpu_times))

    time_detail["min_epoch_time_sec"] = float(np.min(epoch_times))
    time_detail["max_epoch_time_sec"] = float(np.max(epoch_times))
    time_detail["min_local_train_time_sec"] = float(np.min(local_train_times))
    time_detail["max_local_train_time_sec"] = float(np.max(local_train_times))
    time_detail["min_schedule_time_sec"] = float(np.min(schedule_times))
    time_detail["max_schedule_time_sec"] = float(np.max(schedule_times))
    time_detail["min_agg_time_sec"] = float(np.min(agg_times))
    time_detail["max_agg_time_sec"] = float(np.max(agg_times))
    time_detail["min_transfer_to_gpu_time_sec"] = float(np.min(transfer_to_gpu_times))
    time_detail["max_transfer_to_gpu_time_sec"] = float(np.max(transfer_to_gpu_times))
    time_detail["min_transfer_to_cpu_time_sec"] = float(np.min(transfer_to_cpu_times))
    time_detail["max_transfer_to_cpu_time_sec"] = float(np.max(transfer_to_cpu_times))

    return (
        net_glob,
        loss_train,
        acc_curve,
        epoch_times,
        agg_times,
        local_train_times,
        schedule_times,
        transfer_to_gpu_times,
        transfer_to_cpu_times,
        total_train_time,
        time_detail
    )


if __name__ == '__main__':
    program_start_time = time.time()

    args = args_parser()
    print("当前输入维度 input_size =", args.input_size)
    args.epochs = 20
    print("args.epochs =", args.epochs)

    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )

    device_type = "GPU" if torch.cuda.is_available() and args.gpu != -1 else "CPU"
    print("当前可用加速设备:", device_type)
    print("args.device =", args.device)

    cooperative_gpu_device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )
    print("========== 方向一公平协同模式（优化版） ==========")
    print("调度设备: CPU")
    print("本地训练设备:", cooperative_gpu_device)
    print("聚合设备:", cooperative_gpu_device)
    print("全局模型更新设备:", cooperative_gpu_device)

    # ===== 读取数据（synthetic only） =====
    data_prepare_start = time.time()

    if args.dataset != 'synthetic':
        raise ValueError("本版本只支持 synthetic 实验，请使用 --dataset synthetic")

    experiment_seed = 0

    dataset_train, dataset_test = load_fair_dataset_pair(experiment_seed)

    # =====================================================
    # 关键修正：IID 文件不存在时，自动生成并保存
    # =====================================================
    if args.iid:
        iid_path = os.path.join(FAIR_DIR, f"dict_users_iid_seed{experiment_seed}.npy")
        if os.path.exists(iid_path):
            dict_users = np.load(iid_path, allow_pickle=True).item()
            print("已读取 IID 划分文件:", iid_path)
        else:
            dict_users = build_iid_split(
                dataset_train=dataset_train,
                num_users=args.num_users,
                seed=experiment_seed
            )
            np.save(iid_path, dict_users, allow_pickle=True)
            print("未找到 IID 划分文件，已自动生成并保存:", iid_path)
    else:
        dict_users = load_fair_dict_users(experiment_seed, iid=False)

    client_schedule_fixed = load_fair_client_schedule(experiment_seed)
    fixed_experiment_seed = experiment_seed

    data_prepare_time = time.time() - data_prepare_start
    print("数据准备完成，用时: {:.2f} 秒".format(data_prepare_time))

    img_size = dataset_train[0][0].shape

    # ===== 打印客户端样本量 =====
    print("========== Client sample statistics ==========")
    for i in range(args.num_users):
        print("user {}: {} samples".format(i, len(dict_users[i])))

    # =========================================================
    # 0) 单次训练
    # =========================================================
    print("\n========== Single training (CPU schedule + GPU train + GPU agg, optimized) ==========")
    single_train_start = time.time()

    (
        net_glob,
        loss_curve,
        acc_curve,
        epoch_times,
        agg_times,
        local_train_times,
        schedule_times,
        transfer_to_gpu_times,
        transfer_to_cpu_times,
        total_train_time,
        single_time_detail
    ) = train_federated_cooperative(
        args=args,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        dict_users=dict_users,
        img_size=img_size,
        client_schedule=client_schedule_fixed,
        experiment_seed=fixed_experiment_seed,
        frac=args.frac,
        all_clients_flag=args.all_clients,
        verbose=True
    )

    single_train_total_time = time.time() - single_train_start

    print("\n========== Single training time statistics ==========")
    print("单次训练总时间: {:.2f} 秒".format(total_train_time))
    print("单次训练外层总耗时: {:.2f} 秒".format(single_train_total_time))

    print("平均每轮时间: {:.2f} 秒/epoch".format(np.mean(epoch_times)))
    print("最快轮次时间: {:.2f} 秒".format(np.min(epoch_times)))
    print("最慢轮次时间: {:.2f} 秒".format(np.max(epoch_times)))

    print("平均每轮本地训练时间: {:.2f} 秒/epoch".format(np.mean(local_train_times)))
    print("最快本地训练时间: {:.2f} 秒".format(np.min(local_train_times)))
    print("最慢本地训练时间: {:.2f} 秒".format(np.max(local_train_times)))

    print("平均每轮调度时间: {:.6f} 秒/epoch".format(np.mean(schedule_times)))
    print("最快调度时间: {:.6f} 秒".format(np.min(schedule_times)))
    print("最慢调度时间: {:.6f} 秒".format(np.max(schedule_times)))

    print("平均每轮 CPU->GPU 传输时间: {:.6f} 秒/epoch".format(np.mean(transfer_to_gpu_times)))
    print("平均每轮 GPU 聚合时间: {:.4f} 秒/epoch".format(np.mean(agg_times)))
    print("平均每轮 GPU->CPU 回传时间: {:.6f} 秒/epoch".format(np.mean(transfer_to_cpu_times)))

    np.save(os.path.join(SAVE_DIR, "single_epoch_times.npy"), np.array(epoch_times))
    print("single_epoch_times.npy 已保存")

    np.save(os.path.join(SAVE_DIR, "single_agg_times.npy"), np.array(agg_times))
    print("single_agg_times.npy 已保存")

    np.save(os.path.join(SAVE_DIR, "single_local_train_times.npy"), np.array(local_train_times))
    print("single_local_train_times.npy 已保存")

    np.save(os.path.join(SAVE_DIR, "single_schedule_times.npy"), np.array(schedule_times))
    print("single_schedule_times.npy 已保存")

    np.save(os.path.join(SAVE_DIR, "single_transfer_to_gpu_times.npy"), np.array(transfer_to_gpu_times))
    print("single_transfer_to_gpu_times.npy 已保存")

    np.save(os.path.join(SAVE_DIR, "single_transfer_to_cpu_times.npy"), np.array(transfer_to_cpu_times))
    print("single_transfer_to_cpu_times.npy 已保存")

    # loss 曲线
    plt.figure()
    plt.plot(range(1, len(loss_curve) + 1), loss_curve)
    plt.ylabel('train_loss')
    plt.xlabel('round')
    plt.tight_layout()
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid
    ))
    plt.close()

    net_glob.eval()
    old_device = args.device
    args.device = cooperative_gpu_device
    acc_train, loss_train_eval = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    args.device = old_device

    print("Training accuracy: {:.4f}".format(acc_train))
    print("Testing accuracy: {:.4f}".format(acc_test))

    np.save(os.path.join(SAVE_DIR, "single_convergence.npy"), np.array(acc_curve))
    print("single_convergence.npy 已保存")

    np.save(os.path.join(SAVE_DIR, "single_time_detail.npy"), single_time_detail, allow_pickle=True)
    print("single_time_detail.npy 已保存")

    single_time_summary = {
        "mode": "CPU_schedule_GPU_train_GPU_agg_optimized",
        "device": "Cooperative_Direction1_Optimized",

        "data_prepare_time_sec": float(data_prepare_time),

        "single_train_time_sec": float(total_train_time),
        "single_train_outer_time_sec": float(single_train_total_time),

        "avg_epoch_time_sec": float(np.mean(epoch_times)),
        "min_epoch_time_sec": float(np.min(epoch_times)),
        "max_epoch_time_sec": float(np.max(epoch_times)),
        "total_epoch_time_sec": float(np.sum(epoch_times)),

        "avg_local_train_time_sec": float(np.mean(local_train_times)),
        "min_local_train_time_sec": float(np.min(local_train_times)),
        "max_local_train_time_sec": float(np.max(local_train_times)),
        "total_local_train_time_sec": float(np.sum(local_train_times)),

        "avg_schedule_time_sec": float(np.mean(schedule_times)),
        "min_schedule_time_sec": float(np.min(schedule_times)),
        "max_schedule_time_sec": float(np.max(schedule_times)),
        "total_schedule_time_sec": float(np.sum(schedule_times)),

        "avg_transfer_to_gpu_time_sec": float(np.mean(transfer_to_gpu_times)),
        "total_transfer_to_gpu_time_sec": float(np.sum(transfer_to_gpu_times)),

        "avg_agg_time_sec": float(np.mean(agg_times)),
        "min_agg_time_sec": float(np.min(agg_times)),
        "max_agg_time_sec": float(np.max(agg_times)),
        "total_agg_time_sec": float(np.sum(agg_times)),

        "avg_transfer_to_cpu_time_sec": float(np.mean(transfer_to_cpu_times)),
        "total_transfer_to_cpu_time_sec": float(np.sum(transfer_to_cpu_times)),

        "final_train_acc": float(acc_train),
        "final_test_acc": float(acc_test)
    }
    np.save(os.path.join(SAVE_DIR, "single_time_summary.npy"), single_time_summary)
    print("single_time_summary.npy 已保存")

    # =====================================================
    # 关键修正：IID baseline 跑完后直接退出
    # =====================================================
    if args.iid:
        print("IID baseline 已完成，跳过多α和多frac实验。")
        exit()

    # =========================================================
    # 1) 多 α 实验
    # =========================================================
    print("\n========== Multi-alpha experiment ==========")
    multi_alpha_start = time.time()

    alphas = [0.01, 0.1, 0.3, 0.5, 0.8, 1, 10]
    runs_per_setting = 5

    convergence_curves = {}
    alpha_results = {}
    alpha_time_results = {}
    alpha_run_times = {}
    alpha_agg_time_results = {}
    alpha_agg_run_times = {}
    alpha_transfer_to_gpu_results = {}
    alpha_transfer_to_cpu_results = {}

    compare_all_clients_flag = False

    for alpha in alphas:
        print("\n===== α = {} =====".format(alpha))
        alpha_setting_start = time.time()

        acc_curves_alpha = []
        final_acc_list = []
        time_list_alpha = []
        agg_time_list_alpha = []
        transfer_to_gpu_list_alpha = []
        transfer_to_cpu_list_alpha = []

        for run in range(runs_per_setting):
            run_start = time.time()
            set_seed(1000 + run)

            if args.dataset == 'synthetic':
                dict_users_run = build_dirichlet_split(
                    dataset_train=dataset_train,
                    num_users=args.num_users,
                    num_classes=args.num_classes,
                    alpha=alpha,
                    min_samples=1,
                    seed=1000 + run
                )
            else:
                dict_users_run = copy.deepcopy(dict_users)

            (
                _,
                _,
                acc_curve_run,
                epoch_times_run,
                agg_times_run,
                local_train_times_run,
                schedule_times_run,
                transfer_to_gpu_times_run,
                transfer_to_cpu_times_run,
                total_time_run,
                time_detail_run
            ) = train_federated_cooperative(
                args=args,
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                dict_users=dict_users_run,
                img_size=img_size,
                client_schedule=client_schedule_fixed,
                experiment_seed=fixed_experiment_seed,
                frac=args.frac,
                all_clients_flag=compare_all_clients_flag,
                verbose=False
            )

            run_total_wall_time = time.time() - run_start

            acc_curves_alpha.append(acc_curve_run)
            final_acc_list.append(acc_curve_run[-1])
            time_list_alpha.append(total_time_run)
            agg_time_list_alpha.append(float(np.sum(agg_times_run)))
            transfer_to_gpu_list_alpha.append(float(np.sum(transfer_to_gpu_times_run)))
            transfer_to_cpu_list_alpha.append(float(np.sum(transfer_to_cpu_times_run)))

            print("run {:2d}/{:2d}, final acc = {:.4f}, train_time = {:.2f}s, wall_time = {:.2f}s".format(
                run + 1, runs_per_setting, acc_curve_run[-1], total_time_run, run_total_wall_time
            ))

        mean_curve = np.mean(np.array(acc_curves_alpha), axis=0)
        convergence_curves[alpha] = mean_curve
        alpha_results[alpha] = (
            float(np.mean(final_acc_list)),
            float(np.std(final_acc_list))
        )

        alpha_time_results[alpha] = (
            float(np.mean(time_list_alpha)),
            float(np.std(time_list_alpha))
        )
        alpha_run_times[alpha] = time_list_alpha

        alpha_agg_time_results[alpha] = (
            float(np.mean(agg_time_list_alpha)),
            float(np.std(agg_time_list_alpha))
        )
        alpha_agg_run_times[alpha] = agg_time_list_alpha

        alpha_transfer_to_gpu_results[alpha] = (
            float(np.mean(transfer_to_gpu_list_alpha)),
            float(np.std(transfer_to_gpu_list_alpha))
        )
        alpha_transfer_to_cpu_results[alpha] = (
            float(np.mean(transfer_to_cpu_list_alpha)),
            float(np.std(transfer_to_cpu_list_alpha))
        )

        alpha_setting_total_time = time.time() - alpha_setting_start

        print("α={}, mean_acc={:.4f}, std_acc={:.4f}".format(
            alpha, alpha_results[alpha][0], alpha_results[alpha][1]
        ))
        print("α={}, mean_time={:.2f}s, std_time={:.2f}s, setting_total_wall_time={:.2f}s".format(
            alpha, alpha_time_results[alpha][0], alpha_time_results[alpha][1], alpha_setting_total_time
        ))
        print("α={}, mean_agg_time={:.4f}s, std_agg_time={:.4f}s".format(
            alpha, alpha_agg_time_results[alpha][0], alpha_agg_time_results[alpha][1]
        ))
        print("α={}, mean_cpu_to_gpu={:.6f}s, mean_gpu_to_cpu={:.6f}s".format(
            alpha, alpha_transfer_to_gpu_results[alpha][0], alpha_transfer_to_cpu_results[alpha][0]
        ))

    np.save(os.path.join(SAVE_DIR, "convergence.npy"), convergence_curves)
    np.save(os.path.join(SAVE_DIR, "results.npy"), alpha_results)
    np.save(os.path.join(SAVE_DIR, "alpha_time_results.npy"), alpha_time_results)
    np.save(os.path.join(SAVE_DIR, "alpha_run_times.npy"), alpha_run_times)
    np.save(os.path.join(SAVE_DIR, "alpha_agg_time_results.npy"), alpha_agg_time_results)
    np.save(os.path.join(SAVE_DIR, "alpha_agg_run_times.npy"), alpha_agg_run_times)
    np.save(os.path.join(SAVE_DIR, "alpha_transfer_to_gpu_results.npy"), alpha_transfer_to_gpu_results)
    np.save(os.path.join(SAVE_DIR, "alpha_transfer_to_cpu_results.npy"), alpha_transfer_to_cpu_results)
    print("convergence.npy、results.npy、alpha_time_results.npy、alpha_run_times.npy、alpha_agg_time_results.npy、alpha_agg_run_times.npy、alpha_transfer_to_gpu_results.npy、alpha_transfer_to_cpu_results.npy 已保存")

    multi_alpha_total_time = time.time() - multi_alpha_start
    print("Multi-alpha experiment 总时间: {:.2f} 秒".format(multi_alpha_total_time))

    # =========================================================
    # 2) 多 α + 多 frac 实验
    # =========================================================
    print("\n========== Multi-alpha + multi-frac experiment ==========")
    multi_alpha_frac_start = time.time()

    alphas_for_frac = [0.1, 0.3, 0.5, 0.8, 1, 10]
    fracs = [0.1, 0.3, 0.5, 0.8]
    runs_per_setting = 5

    frac_results = {}
    convergence_curves_frac = {}
    frac_time_results = {}
    frac_run_times = {}
    frac_agg_time_results = {}
    frac_agg_run_times = {}
    frac_transfer_to_gpu_results = {}
    frac_transfer_to_cpu_results = {}

    compare_all_clients_flag = False

    pair_final_accs = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_curves = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_times = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_agg_times = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_transfer_to_gpu_times = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_transfer_to_cpu_times = {(a, f): [] for a in alphas_for_frac for f in fracs}

    for alpha_fixed in alphas_for_frac:
        print("\n===== α = {} =====".format(alpha_fixed))
        alpha_fixed_start = time.time()

        for run in range(runs_per_setting):
            set_seed(1000 + run)

            if args.dataset == 'synthetic':
                dict_users_run = build_dirichlet_split(
                    dataset_train=dataset_train,
                    num_users=args.num_users,
                    num_classes=args.num_classes,
                    alpha=alpha_fixed,
                    min_samples=1,
                    seed=1000 + run
                )
            else:
                dict_users_run = copy.deepcopy(dict_users)

            for frac in fracs:
                pair_start = time.time()
                print("alpha={}, frac={}, run={}".format(alpha_fixed, frac, run + 1))

                (
                    _,
                    _,
                    acc_curve_run,
                    epoch_times_run,
                    agg_times_run,
                    local_train_times_run,
                    schedule_times_run,
                    transfer_to_gpu_times_run,
                    transfer_to_cpu_times_run,
                    total_time_run,
                    time_detail_run
                ) = train_federated_cooperative(
                    args=args,
                    dataset_train=dataset_train,
                    dataset_test=dataset_test,
                    dict_users=dict_users_run,
                    img_size=img_size,
                    client_schedule=client_schedule_fixed,
                    experiment_seed=fixed_experiment_seed,
                    frac=frac,
                    all_clients_flag=compare_all_clients_flag,
                    verbose=False
                )

                pair_wall_time = time.time() - pair_start

                pair_final_accs[(alpha_fixed, frac)].append(acc_curve_run[-1])
                pair_curves[(alpha_fixed, frac)].append(acc_curve_run)
                pair_times[(alpha_fixed, frac)].append(total_time_run)
                pair_agg_times[(alpha_fixed, frac)].append(float(np.sum(agg_times_run)))
                pair_transfer_to_gpu_times[(alpha_fixed, frac)].append(float(np.sum(transfer_to_gpu_times_run)))
                pair_transfer_to_cpu_times[(alpha_fixed, frac)].append(float(np.sum(transfer_to_cpu_times_run)))

                print("final acc = {:.4f}, train_time = {:.2f}s, wall_time = {:.2f}s".format(
                    acc_curve_run[-1], total_time_run, pair_wall_time
                ))

        alpha_fixed_total_time = time.time() - alpha_fixed_start
        print("α={} 的所有 frac + run 总耗时: {:.2f} 秒".format(alpha_fixed, alpha_fixed_total_time))

    for alpha_fixed in alphas_for_frac:
        for frac in fracs:
            final_accs = pair_final_accs[(alpha_fixed, frac)]
            curves = pair_curves[(alpha_fixed, frac)]
            times_list = pair_times[(alpha_fixed, frac)]
            agg_times_list = pair_agg_times[(alpha_fixed, frac)]
            transfer_to_gpu_list = pair_transfer_to_gpu_times[(alpha_fixed, frac)]
            transfer_to_cpu_list = pair_transfer_to_cpu_times[(alpha_fixed, frac)]

            frac_results[(alpha_fixed, frac)] = (
                float(np.mean(final_accs)),
                float(np.std(final_accs))
            )

            convergence_curves_frac[(alpha_fixed, frac)] = np.mean(np.array(curves), axis=0)
            frac_time_results[(alpha_fixed, frac)] = (
                float(np.mean(times_list)),
                float(np.std(times_list))
            )
            frac_run_times[(alpha_fixed, frac)] = times_list

            frac_agg_time_results[(alpha_fixed, frac)] = (
                float(np.mean(agg_times_list)),
                float(np.std(agg_times_list))
            )
            frac_agg_run_times[(alpha_fixed, frac)] = agg_times_list

            frac_transfer_to_gpu_results[(alpha_fixed, frac)] = (
                float(np.mean(transfer_to_gpu_list)),
                float(np.std(transfer_to_gpu_list))
            )
            frac_transfer_to_cpu_results[(alpha_fixed, frac)] = (
                float(np.mean(transfer_to_cpu_list)),
                float(np.std(transfer_to_cpu_list))
            )

            print("α={}, frac={}, mean_acc={:.4f}, std_acc={:.4f}, mean_time={:.2f}s, std_time={:.2f}s, mean_agg_time={:.4f}s, std_agg_time={:.4f}s".format(
                alpha_fixed,
                frac,
                frac_results[(alpha_fixed, frac)][0],
                frac_results[(alpha_fixed, frac)][1],
                frac_time_results[(alpha_fixed, frac)][0],
                frac_time_results[(alpha_fixed, frac)][1],
                frac_agg_time_results[(alpha_fixed, frac)][0],
                frac_agg_time_results[(alpha_fixed, frac)][1]
            ))

    np.save(os.path.join(SAVE_DIR, "frac_results.npy"), frac_results)
    np.save(os.path.join(SAVE_DIR, "convergence_frac.npy"), convergence_curves_frac)
    np.save(os.path.join(SAVE_DIR, "frac_time_results.npy"), frac_time_results)
    np.save(os.path.join(SAVE_DIR, "frac_run_times.npy"), frac_run_times)
    np.save(os.path.join(SAVE_DIR, "frac_agg_time_results.npy"), frac_agg_time_results)
    np.save(os.path.join(SAVE_DIR, "frac_agg_run_times.npy"), frac_agg_run_times)
    np.save(os.path.join(SAVE_DIR, "frac_transfer_to_gpu_results.npy"), frac_transfer_to_gpu_results)
    np.save(os.path.join(SAVE_DIR, "frac_transfer_to_cpu_results.npy"), frac_transfer_to_cpu_results)
    print("frac_results.npy、convergence_frac.npy、frac_time_results.npy、frac_run_times.npy、frac_agg_time_results.npy、frac_agg_run_times.npy、frac_transfer_to_gpu_results.npy、frac_transfer_to_cpu_results.npy 已保存")

    multi_alpha_frac_total_time = time.time() - multi_alpha_frac_start
    print("Multi-alpha + multi-frac experiment 总时间: {:.2f} 秒".format(multi_alpha_frac_total_time))

    program_total_time = time.time() - program_start_time
    overall_time_summary = {
        "mode": "CPU_schedule_GPU_train_GPU_agg_optimized",
        "device": "Cooperative_Direction1_Optimized",

        "data_prepare_time_sec": float(data_prepare_time),
        "single_training_block_time_sec": float(single_train_total_time),
        "multi_alpha_time_sec": float(multi_alpha_total_time),
        "multi_alpha_frac_time_sec": float(multi_alpha_frac_total_time),
        "program_total_time_sec": float(program_total_time),

        "single_total_epoch_time_sec": float(np.sum(epoch_times)),
        "single_total_local_train_time_sec": float(np.sum(local_train_times)),
        "single_total_schedule_time_sec": float(np.sum(schedule_times)),
        "single_total_transfer_to_gpu_time_sec": float(np.sum(transfer_to_gpu_times)),
        "single_total_agg_time_sec": float(np.sum(agg_times)),
        "single_total_transfer_to_cpu_time_sec": float(np.sum(transfer_to_cpu_times))
    }
    np.save(os.path.join(SAVE_DIR, "overall_time_summary.npy"), overall_time_summary)

    print("\n========== Overall time summary ==========")
    print("数据准备时间: {:.2f} 秒".format(data_prepare_time))
    print("单次训练模块时间: {:.2f} 秒".format(single_train_total_time))
    print("多 α 实验时间: {:.2f} 秒".format(multi_alpha_total_time))
    print("多 α + 多 frac 实验时间: {:.2f} 秒".format(multi_alpha_frac_total_time))
    print("程序总运行时间: {:.2f} 秒 ({:.2f} 分钟)".format(
        program_total_time, program_total_time / 60.0
    ))
    print("overall_time_summary.npy 已保存")
