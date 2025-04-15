import argparse
import csv
from copy import copy

import torch
import torch_geometric.transforms as T
import numpy as np

from torch_geometric import seed_everything
from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from torch_geometric.utils import sort_edge_index

from hetero2net.dataset import load_dataset
from hetero2net.logger import setup_logger
from hetero2net.models import HeteroGNN
#from hetero2net.models_layers3 import HeteroGNN
from hetero2net.lr_scheduler import get_cosine_schedule_with_warmup
from hetero2net.utils import (
    evaluate_full_batch,
    evaluate_mini_batch,
    train_full_batch,
    train_mini_batch,
    tab_printer,
)

def main(seed_override=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=["micro-f1", "macro-f1"])
    parser.add_argument("--monitor", type=str, default="metric")
    parser.add_argument('--mask_lp', action='store_true')
    parser.add_argument('--p', type=float, default=0.7)
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 10])
    parser.add_argument('--batch_size', type=int, default=1024)

    try:
        args = parser.parse_args()
        table = tab_printer(args)
    except:
        parser.print_help()
        exit(0)

    if seed_override is not None:
        args.seed = seed_override

    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    logger = setup_logger(output=f'log2.txt')
    #logger.info(f'\n{tab_printer(args)}')

    metrics = args.metrics
    monitor = args.monitor
    dataset = args.dataset
    hidden = args.hidden
    lr = args.lr
    epochs = args.epochs
    dropout = args.dropout
    r = args.r
    alpha = args.alpha
    beta = args.beta
    p = args.p
    batch_size = args.batch_size
    num_neighbors = args.num_neighbors
    mask_lp = args.mask_lp
    project = False if dataset in ['ACM', 'IMDB'] else True
    mini_batch = dataset in ['MAG', 'RCDD']

    data, metapaths = load_dataset(dataset)
    #logger.info(data)

    for et in data.edge_types:
        data[et].edge_index = sort_edge_index(data[et].edge_index)

    node_types = [t for t in data.node_types if data[t].get('y') is not None]
    assert len(node_types) == 1
    node_type = node_types[0]

    if dataset == 'FreeBase':
        data = data.to(device, 'y')
    else:
        data = data.to(device, 'x', 'y')

    if data[node_type].y.squeeze().ndim == 1:
        num_classes = data[node_type].y.max().item() + 1
    else:
        num_classes = data[node_type].y.size(-1)

    train_mask = data[node_type].train_mask
    test_mask = data[node_type].test_mask
    val_mask = data[node_type].val_mask

    train_data = copy(data)
    test_data = copy(data)

    if mask_lp:
        y = train_data[node_type].y.clone()
        if y.ndim > 1:
            y = y.long()
            row, col = y.nonzero().T
            y[y == 0] = num_classes
            y[row, col] = col
        y[y == -1] = num_classes
        y[test_mask] = num_classes
        test_data[node_type].y_emb = y.clone()
        if val_mask is not None:
            y[val_mask] = num_classes
        train_data[node_type].y_emb = y.clone()
        del y

    if mini_batch:
        train_loader = NeighborLoader(
            train_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=(node_type, train_mask),
            num_workers=6,
            shuffle=True,
        )

        test_loader = NeighborLoader(
            test_data,
            num_neighbors=[-1, -1] if dataset in ['ACM', 'DBLP', 'IMDB'] else num_neighbors,
            batch_size=batch_size,
            input_nodes=(node_type, test_mask),
            num_workers=6,
            shuffle=False,
        )

        if val_mask is not None:
            val_loader = NeighborLoader(
                train_data,
                num_neighbors=[-1, -1] if dataset in ['ACM', 'DBLP', 'IMDB'] else num_neighbors,
                batch_size=batch_size,
                input_nodes=(node_type, val_mask),
                num_workers=6,
                shuffle=False,
            )

    model = HeteroGNN(data.metadata(),
                      in_channels=data[node_type].x.size(1),
                      hidden_channels=hidden,
                      out_channels=num_classes,
                      dropout=dropout,
                      project=project).to(device)
    model.node_type = node_type
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = None

    best_metrics = None
    best_val_loss = 1e5
    best_val_metric = 0

    for epoch in range(1, epochs + 1):
        if mini_batch:
            loss = train_mini_batch(model, optimizer, train_loader, device,
                                    p=p, metapaths=metapaths, r=r,
                                    alpha=alpha, beta=beta, schedule=schedule)
            test_loss, test_metrics = evaluate_mini_batch(model, test_loader, device, metrics=metrics)
            if val_mask is not None:
                val_loss, val_metrics = evaluate_mini_batch(model, val_loader, device, metrics=metrics)
            else:
                val_loss, val_metrics = test_loss, test_metrics
        else:
            loss = train_full_batch(model, optimizer, train_data, train_mask, device,
                                    p=p, metapaths=metapaths, r=r,
                                    alpha=alpha, beta=beta, schedule=schedule)
            test_loss, test_metrics = evaluate_full_batch(model, test_data, test_mask, device, metrics=metrics)
            if val_mask is not None:
                val_loss, val_metrics = evaluate_full_batch(model, train_data, val_mask, device, metrics=metrics)
            else:
                val_loss, val_metrics = test_loss, test_metrics

        if monitor == 'loss':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = test_metrics
        else:
            if val_metrics[0] > best_val_metric:
                best_val_metric = val_metrics[0]
                best_metrics = test_metrics

    metric_dict = dict(zip(metrics, best_metrics))
    return metric_dict


def run_experiments(runs=10, csv_file='results.csv'):
    micro_f1s = []
    macro_f1s = []

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Micro-F1', 'Macro-F1'])

        for i in range(runs):
            print(f"\n========== Run {i + 1} ==========")
            result = main(seed_override=2024 + i)
            micro_f1 = result.get("micro-f1", None)
            macro_f1 = result.get("macro-f1", None)

            # ✅ 打印当前实验的结果
            print(f"Run {i + 1} - Micro-F1: {micro_f1 * 100:.2f}%, Macro-F1: {macro_f1 * 100:.2f}%")
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)

            writer.writerow([i + 1, f'{micro_f1 * 100:.2f}', f'{macro_f1 * 100:.2f}'])

        # Write summary
        writer.writerow([])
        writer.writerow(['Mean', f'{np.mean(micro_f1s) * 100:.2f}', f'{np.mean(macro_f1s) * 100:.2f}'])
        writer.writerow(['Std', f'{np.std(micro_f1s) * 100:.2f}', f'{np.std(macro_f1s) * 100:.2f}'])

    print("\n===== Summary over 10 runs =====")
    print(f"Micro-F1: mean = {np.mean(micro_f1s)*100:.2f}%, std = {np.std(micro_f1s)*100:.2f}%")
    print(f"Macro-F1: mean = {np.mean(macro_f1s)*100:.2f}%, std = {np.std(macro_f1s)*100:.2f}%")


if __name__ == "__main__":
    # 获取 dataset 名称
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ACM")
    # 加入一些默认参数（为了兼容 argparse）
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=["micro-f1", "macro-f1"])
    parser.add_argument("--monitor", type=str, default="metric")
    parser.add_argument('--mask_lp', action='store_true')
    parser.add_argument('--p', type=float, default=0.7)
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 10])
    parser.add_argument('--batch_size', type=int, default=1024)
    args, _ = parser.parse_known_args()
    dataset = args.dataset

    csv_file_name = f"result_{dataset}_layers_3_diff.csv"
    run_experiments(runs=10, csv_file=csv_file_name)

