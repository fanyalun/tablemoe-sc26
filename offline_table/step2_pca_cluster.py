import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_table.common import (
    COMPRESSED_DIM,
    DEFAULT_CLUSTER_SIZE,
    DEFAULT_DTYPE_NAME,
    DTYPE_CHOICES,
    MODEL_CHOICES,
    get_threshold,
    get_torch_dtype,
)


MAX_ITER = 30
N_INIT = 1
INIT_METHOD = "kmeans++"


def load_expert_data(exp_dir, modality, max_vectors=50000):
    modality_dir = os.path.join(exp_dir, modality)
    files = sorted(glob.glob(os.path.join(modality_dir, "vectors_part_*.npy")))
    if not files:
        return None

    data_list = []
    total = 0
    for file_path in files:
        try:
            arr = np.load(file_path)
        except Exception:
            continue
        data_list.append(arr)
        total += arr.shape[0]
        if total > max_vectors:
            break

    if not data_list:
        return None

    data = np.concatenate(data_list, axis=0)
    if data.shape[0] > max_vectors:
        indices = np.random.choice(data.shape[0], max_vectors, replace=False)
        data = data[indices]
    return torch.from_numpy(data).float()


def train_pca_layer(layer_data_list, compressed_dim=COMPRESSED_DIM):
    samples = []
    for tensor in layer_data_list:
        sample_count = min(tensor.shape[0], 2000)
        indices = torch.randperm(tensor.shape[0])[:sample_count]
        samples.append(tensor[indices])

    if not samples:
        return None, None

    stacked = torch.cat(samples, dim=0)
    mean = stacked.mean(dim=0)
    centered = stacked - mean
    _, _, proj = torch.pca_lowrank(centered, q=compressed_dim, center=False, niter=4)
    return mean, proj


def fast_kmeans(x, k, max_iter=30, tol=1e-4, n_init=1, init_method="random"):
    num_rows, dim = x.shape
    if num_rows <= k:
        return x, torch.arange(num_rows, device=x.device), 0.0

    x = x.contiguous()
    best_centroids = None
    best_labels = None
    best_inertia = float("inf")

    for _ in range(n_init):
        if init_method == "random":
            indices = torch.randperm(num_rows, device=x.device)[:k]
            centroids = x[indices].clone()
        else:
            centroids = torch.empty((k, dim), device=x.device, dtype=x.dtype)
            first_idx = torch.randint(0, num_rows, (1,)).item()
            centroids[0] = x[first_idx]
            dist_sq = torch.norm(x - centroids[0], dim=1) ** 2
            for cluster_idx in range(1, k):
                probs = dist_sq / (dist_sq.sum() + 1e-10)
                cumulative = torch.cumsum(probs, dim=0)
                random_value = torch.rand(1, device=x.device).item()
                next_idx = torch.searchsorted(cumulative, random_value).item()
                next_idx = min(next_idx, num_rows - 1)
                centroids[cluster_idx] = x[next_idx]
                dist_sq = torch.minimum(dist_sq, torch.norm(x - centroids[cluster_idx], dim=1) ** 2)

        current_labels = None
        last_inertia = float("inf")
        current_inertia = float("inf")

        for _ in range(max_iter):
            distances = torch.cdist(x, centroids)
            min_dists, labels = torch.min(distances, dim=1)
            current_inertia = min_dists.sum().item()
            current_labels = labels

            if abs(last_inertia - current_inertia) < tol:
                break
            last_inertia = current_inertia

            new_centroids = []
            for cluster_idx in range(k):
                mask = labels == cluster_idx
                if mask.any():
                    new_centroids.append(x[mask].mean(dim=0))
                else:
                    worst_idx = torch.argmax(min_dists)
                    new_centroids.append(x[worst_idx])
                    min_dists[worst_idx] = 0
            centroids = torch.stack(new_centroids)

        if current_inertia < best_inertia:
            best_inertia = current_inertia
            best_centroids = centroids
            best_labels = current_labels

    return best_centroids, best_labels, best_inertia


def load_reused_pca_model(reuse_pca_dir, layer_idx, modality):
    if not reuse_pca_dir:
        return None
    pca_path = os.path.join(reuse_pca_dir, f"layer_{layer_idx}", f"L{layer_idx}_{modality}_pca.pt")
    if not os.path.isfile(pca_path):
        return None
    return torch.load(pca_path, map_location="cpu")


def plot_layer_cdf(model_name, layer_idx, num_layers, errs_vis, errs_txt, save_path, cluster_size):
    plt.figure(figsize=(10, 6))

    if len(errs_vis) > 0:
        sorted_vis = np.sort(errs_vis)
        plt.plot(sorted_vis, np.arange(len(sorted_vis)) / len(sorted_vis), label="Vision", color="blue", linewidth=2)
        thresh_v = get_threshold(model_name, "vision", layer_idx, num_layers=num_layers)
        plt.axvline(thresh_v, color="blue", linestyle="--", alpha=0.6, label=f"Vis Thresh {thresh_v}")

    if len(errs_txt) > 0:
        sorted_txt = np.sort(errs_txt)
        plt.plot(sorted_txt, np.arange(len(sorted_txt)) / len(sorted_txt), label="Text", color="orange", linewidth=2)
        thresh_t = get_threshold(model_name, "text", layer_idx, num_layers=num_layers)
        plt.axvline(thresh_t, color="orange", linestyle="--", alpha=0.6, label=f"Txt Thresh {thresh_t}")

    plt.xlabel("Relative Error (to Nearest Centroid)")
    plt.ylabel("CDF (Cumulative Proportion)")
    plt.title(f"Layer {layer_idx} Clustering Quality CDF (K={cluster_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.2)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Unified Step2 PCA + K-Means clustering")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default=DEFAULT_DTYPE_NAME)
    parser.add_argument("--cluster-size", type=int, default=DEFAULT_CLUSTER_SIZE)
    parser.add_argument("--min-samples-per-expert", type=int, default=None)
    parser.add_argument("--reuse-pca-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if args.min_samples_per_expert is None:
        args.min_samples_per_expert = args.cluster_size
    save_dtype = get_torch_dtype(args.dtype)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"=== Fixed K={args.cluster_size} Clustering ({args.model}, dtype={args.dtype}, init={INIT_METHOD}) ===")

    layer_dirs = sorted(glob.glob(os.path.join(args.data_dir, "layer_*")))
    layer_indices = []
    for layer_dir in layer_dirs:
        try:
            layer_indices.append(int(os.path.basename(layer_dir).split("_")[1]))
        except Exception:
            continue
    num_layers = (max(layer_indices) + 1) if layer_indices else 0

    for layer_dir in tqdm(layer_dirs, desc="Processing Layers"):
        try:
            layer_idx = int(os.path.basename(layer_dir).split("_")[1])
        except Exception:
            continue

        layer_save_dir = os.path.join(args.save_dir, f"layer_{layer_idx}")
        os.makedirs(layer_save_dir, exist_ok=True)
        expert_dirs = sorted(glob.glob(os.path.join(layer_dir, "expert_*")))

        vis_buffer = []
        txt_buffer = []
        for expert_dir in expert_dirs:
            vis_data = load_expert_data(expert_dir, "vision", max_vectors=1000)
            if vis_data is not None:
                vis_buffer.append(vis_data)
            txt_data = load_expert_data(expert_dir, "text", max_vectors=1000)
            if txt_data is not None:
                txt_buffer.append(txt_data)

        pca_models = {}
        for modality, modality_buffer in (("vision", vis_buffer), ("text", txt_buffer)):
            reused_pca = load_reused_pca_model(args.reuse_pca_dir, layer_idx, modality)
            if reused_pca is not None:
                mean = reused_pca["mean"].float()
                proj = reused_pca["proj"].float()
            elif modality_buffer:
                mean, proj = train_pca_layer(modality_buffer)
            else:
                continue
            pca_models[modality] = (mean.to(device), proj.to(device))
            torch.save(
                {"mean": mean.to(save_dtype).cpu(), "proj": proj.to(save_dtype).cpu()},
                os.path.join(layer_save_dir, f"L{layer_idx}_{modality}_pca.pt"),
            )

        del vis_buffer, txt_buffer
        layer_errs = {"vision": [], "text": []}

        for expert_dir in expert_dirs:
            expert_id = int(os.path.basename(expert_dir).split("_")[1])
            for modality in ("vision", "text"):
                if modality not in pca_models:
                    continue
                x_orig = load_expert_data(expert_dir, modality, max_vectors=30000)
                if x_orig is None or x_orig.shape[0] < args.min_samples_per_expert:
                    continue

                x_orig = x_orig.to(device)
                mean, proj = pca_models[modality]
                x_low = torch.matmul(x_orig - mean, proj)
                _, labels, _ = fast_kmeans(
                    x_low,
                    args.cluster_size,
                    max_iter=MAX_ITER,
                    n_init=N_INIT,
                    init_method=INIT_METHOD,
                )

                centroids_high = []
                centroids_low = []
                for cluster_idx in range(args.cluster_size):
                    mask = labels == cluster_idx
                    if mask.any():
                        centroids_high.append(x_orig[mask].mean(dim=0))
                        centroids_low.append(x_low[mask].mean(dim=0))
                    else:
                        centroids_high.append(torch.zeros(x_orig.shape[1], device=device))
                        centroids_low.append(torch.zeros(COMPRESSED_DIM, device=device))

                centroids_high_final = torch.stack(centroids_high)
                centroids_low_final = torch.stack(centroids_low)
                rel_errs = torch.norm(x_orig - centroids_high_final[labels], dim=1) / (torch.norm(x_orig, dim=1) + 1e-6)
                if rel_errs.numel() > 2000:
                    indices = torch.randperm(rel_errs.numel())[:2000]
                    layer_errs[modality].append(rel_errs[indices].cpu().numpy())
                else:
                    layer_errs[modality].append(rel_errs.cpu().numpy())

                save_path = os.path.join(layer_save_dir, f"L{layer_idx}_E{expert_id}_{modality}_clusters.pt")
                torch.save(
                    {
                        "key": centroids_low_final.to(save_dtype).cpu(),
                        "value": centroids_high_final.to(save_dtype).cpu(),
                    },
                    save_path,
                )

        if layer_errs["vision"] or layer_errs["text"]:
            vis_all = np.concatenate(layer_errs["vision"]) if layer_errs["vision"] else []
            txt_all = np.concatenate(layer_errs["text"]) if layer_errs["text"] else []
            plot_layer_cdf(
                args.model,
                layer_idx,
                num_layers,
                vis_all,
                txt_all,
                os.path.join(layer_save_dir, f"L{layer_idx}_cdf.png"),
                args.cluster_size,
            )

    print(f"\nAll Done! Results saved in {args.save_dir}/layer_X/")


if __name__ == "__main__":
    main()
