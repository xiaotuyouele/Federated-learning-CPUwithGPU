#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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
from models.Fed import FedAvg
from models.test import test_img


SAVE_DIR = "/content/save"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('./save', exist_ok=True)

FAIR_DIR = "/content/drive/MyDrive/fair_experiment"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_model(args, img_size):
    """
    根据数据集和模型类型创建模型
    """
    if args.dataset == 'synthetic':
        return MLP(
            dim_in=args.input_size,
            dim_hidden1=200,
            dim_hidden2=100,
            dim_out=args.num_classes
        ).to(args.device)

    if args.model == 'cnn' and args.dataset == 'cifar':
        return CNNCifar(args=args).to(args.device)

    if args.model == 'cnn' and args.dataset == 'mnist':
        return CNNMnist(args=args).to(args.device)

    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        return MLP(
            dim_in=len_in,
            dim_hidden1=200,
            dim_hidden2=100,
            dim_out=args.num_classes
        ).to(args.device)

    raise ValueError('Error: unrecognized model')


def train_federated(args, dataset_train, dataset_test, dict_users, img_size,
                    client_schedule, experiment_seed,
                    frac=None, all_clients_flag=None, verbose=True):
    """
    执行一次联邦训练，返回：
    - 最终模型
    - loss 曲线
    - acc 曲线
    - 每轮时间列表
    - 每轮聚合时间列表
    - 总训练时间
    """
    if frac is None:
        frac = args.frac
    if all_clients_flag is None:
        all_clients_flag = args.all_clients

    net_glob = build_model(args, img_size)

    init_state = torch.load(
        os.path.join(FAIR_DIR, f"init_model_seed{experiment_seed}.pth"),
        map_location=args.device
    )
    net_glob.load_state_dict(init_state)

    net_glob.train()
    w_glob = copy.deepcopy(net_glob.state_dict())

    loss_train = []
    acc_curve = []
    epoch_times = []
    agg_times = []

    if all_clients_flag:
        if verbose:
            print("Aggregation over all clients")
        w_locals = [copy.deepcopy(w_glob) for _ in range(args.num_users)]

    train_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        loss_locals = []

        if not all_clients_flag:
            w_locals = []

        if all_clients_flag:
            idxs_users = list(range(args.num_users))
        else:
            m = max(int(frac * args.num_users), 1)
            epoch_order = list(client_schedule[epoch])
            idxs_users = epoch_order[:m]

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            if all_clients_flag:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))

        agg_start_time = time.time()
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        agg_time = time.time() - agg_start_time
        agg_times.append(agg_time)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        net_glob.eval()
        acc_test, _ = test_img(net_glob, dataset_test, args)
        acc_curve.append(acc_test)
        net_glob.train()

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if verbose:
            print(
                'Round {:3d}, clients {:3d}, Average loss {:.4f}, Test accuracy {:.4f}, Epoch Time {:.2f}s, Agg Time {:.4f}s'.format(
                    epoch + 1, len(idxs_users), loss_avg, acc_test, epoch_time, agg_time
                )
            )

    total_train_time = time.time() - train_start_time

    return net_glob, loss_train, acc_curve, epoch_times, agg_times, total_train_time


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
    print("当前设备:", device_type)
    print("device =", args.device)

    # ===== 读取数据 =====
    data_prepare_start = time.time()

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            raise ValueError("当前代码未为 CIFAR 写 Dirichlet 非IID 划分，可先用 synthetic 或 mnist")

    elif args.dataset == 'synthetic':
        experiment_seed = 0

        dataset_train = torch.load(
            os.path.join(FAIR_DIR, f"dataset_train_seed{experiment_seed}.pt"),
            weights_only=False
        )
        dataset_test = torch.load(
            os.path.join(FAIR_DIR, f"dataset_test_seed{experiment_seed}.pt"),
            weights_only=False
        )

        if args.iid:
            dict_users = np.load(
                os.path.join(FAIR_DIR, f"dict_users_iid_seed{experiment_seed}.npy"),
                allow_pickle=True
            ).item()
        else:
            dict_users = np.load(
                os.path.join(FAIR_DIR, f"dict_users_seed{experiment_seed}.npy"),
                allow_pickle=True
            ).item()

    else:
        exit('Error: unrecognized dataset')

    data_prepare_time = time.time() - data_prepare_start
    print("数据准备完成，用时: {:.2f} 秒".format(data_prepare_time))

    img_size = dataset_train[0][0].shape

    # ===== 固定公平实验资源 =====
    fixed_experiment_seed = 0
    client_schedule_fixed = np.load(
        os.path.join(FAIR_DIR, "client_schedule_seed0.npy"),
        allow_pickle=True
    )

    # ===== 打印客户端样本量 =====
    print("========== Client sample statistics ==========")
    for i in range(args.num_users):
        print("user {}: {} samples".format(i, len(dict_users[i])))

    # =========================================================
    # 0) 单次训练
    # =========================================================
    print("\n========== Single training ==========")
    single_train_start = time.time()

    net_glob, loss_curve, acc_curve, epoch_times, agg_times, total_train_time = train_federated(
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
    print("平均每轮聚合时间: {:.4f} 秒/epoch".format(np.mean(agg_times)))
    print("最快聚合时间: {:.4f} 秒".format(np.min(agg_times)))
    print("最慢聚合时间: {:.4f} 秒".format(np.max(agg_times)))

    np.save(os.path.join(SAVE_DIR, "single_epoch_times.npy"), np.array(epoch_times))
    print("single_epoch_times.npy 已保存")

    np.save(os.path.join(SAVE_DIR, "single_agg_times.npy"), np.array(agg_times))
    print("single_agg_times.npy 已保存")

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

    # 测试
    net_glob.eval()
    acc_train, loss_train_eval = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.4f}".format(acc_train))
    print("Testing accuracy: {:.4f}".format(acc_test))

    np.save(os.path.join(SAVE_DIR, "single_convergence.npy"), np.array(acc_curve))
    print("single_convergence.npy 已保存")

    single_time_summary = {
        "device": device_type,
        "data_prepare_time_sec": float(data_prepare_time),
        "single_train_time_sec": float(total_train_time),
        "single_train_outer_time_sec": float(single_train_total_time),
        "avg_epoch_time_sec": float(np.mean(epoch_times)),
        "min_epoch_time_sec": float(np.min(epoch_times)),
        "max_epoch_time_sec": float(np.max(epoch_times)),
        "avg_agg_time_sec": float(np.mean(agg_times)),
        "min_agg_time_sec": float(np.min(agg_times)),
        "max_agg_time_sec": float(np.max(agg_times)),
        "total_agg_time_sec": float(np.sum(agg_times)),
        "final_train_acc": float(acc_train),
        "final_test_acc": float(acc_test)
    }
    np.save(os.path.join(SAVE_DIR, "single_time_summary.npy"), single_time_summary)
    print("single_time_summary.npy 已保存")

    # =========================================================
    # 1) 多 α 实验：固定 frac，看 α 对最终精度和收敛曲线的影响
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

    compare_all_clients_flag = False

    for alpha in alphas:
        print("\n===== α = {} =====".format(alpha))
        alpha_setting_start = time.time()

        acc_curves_alpha = []
        final_acc_list = []
        time_list_alpha = []
        agg_time_list_alpha = []

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

            _, _, acc_curve_run, epoch_times_run, agg_times_run, total_time_run = train_federated(
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

    np.save(os.path.join(SAVE_DIR, "convergence.npy"), convergence_curves)
    np.save(os.path.join(SAVE_DIR, "results.npy"), alpha_results)
    np.save(os.path.join(SAVE_DIR, "alpha_time_results.npy"), alpha_time_results)
    np.save(os.path.join(SAVE_DIR, "alpha_run_times.npy"), alpha_run_times)
    np.save(os.path.join(SAVE_DIR, "alpha_agg_time_results.npy"), alpha_agg_time_results)
    np.save(os.path.join(SAVE_DIR, "alpha_agg_run_times.npy"), alpha_agg_run_times)
    print("convergence.npy、results.npy、alpha_time_results.npy、alpha_run_times.npy、alpha_agg_time_results.npy、alpha_agg_run_times.npy 已保存")

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

    compare_all_clients_flag = False

    pair_final_accs = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_curves = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_times = {(a, f): [] for a in alphas_for_frac for f in fracs}
    pair_agg_times = {(a, f): [] for a in alphas_for_frac for f in fracs}

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

                _, _, acc_curve_run, epoch_times_run, agg_times_run, total_time_run = train_federated(
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
    print("frac_results.npy、convergence_frac.npy、frac_time_results.npy、frac_run_times.npy、frac_agg_time_results.npy、frac_agg_run_times.npy 已保存")

    multi_alpha_frac_total_time = time.time() - multi_alpha_frac_start
    print("Multi-alpha + multi-frac experiment 总时间: {:.2f} 秒".format(multi_alpha_frac_total_time))

    program_total_time = time.time() - program_start_time
    overall_time_summary = {
        "device": device_type,
        "data_prepare_time_sec": float(data_prepare_time),
        "single_training_block_time_sec": float(single_train_total_time),
        "multi_alpha_time_sec": float(multi_alpha_total_time),
        "multi_alpha_frac_time_sec": float(multi_alpha_frac_total_time),
        "program_total_time_sec": float(program_total_time),
        "single_total_agg_time_sec": float(np.sum(agg_times))
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
