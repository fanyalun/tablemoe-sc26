import torch


def build_fixed_keep_mask(selected_experts: torch.Tensor, keep_k: int | None) -> torch.Tensor:
    keep_mask = torch.zeros_like(selected_experts, dtype=torch.bool)
    if selected_experts.numel() == 0:
        return keep_mask

    if keep_k is None:
        keep_k = selected_experts.shape[1]

    keep_k = max(0, min(int(keep_k), selected_experts.shape[1]))
    if keep_k > 0:
        keep_mask[:, :keep_k] = True
    return keep_mask


def build_prefill_keep_mask(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    attn_scores: torch.Tensor | None,
    num_experts: int,
    keep_rate: float,
) -> torch.Tensor:
    keep_mask = torch.zeros_like(selected_experts, dtype=torch.bool)
    if selected_experts.numel() == 0:
        return keep_mask

    if attn_scores is not None and attn_scores.numel() == selected_experts.shape[0]:
        combined_scores = attn_scores.unsqueeze(1).to(routing_weights.device) * routing_weights
    else:
        combined_scores = routing_weights / max(selected_experts.shape[0], 1)

    expert_score_sum = torch.zeros(num_experts, device=routing_weights.device, dtype=torch.float32)
    expert_score_sum.scatter_add_(0, selected_experts.reshape(-1), combined_scores.reshape(-1).float())

    activated_experts = torch.unique(selected_experts)
    if activated_experts.numel() == 0:
        return keep_mask

    activated_scores = expert_score_sum[activated_experts]
    sorted_indices = torch.argsort(activated_scores, descending=True)
    sorted_experts = activated_experts[sorted_indices]

    num_keep = max(1, int(sorted_experts.numel() * keep_rate))
    num_keep = min(num_keep, sorted_experts.numel())
    kept_experts = sorted_experts[:num_keep]
    return torch.isin(selected_experts, kept_experts)


def build_decode_keep_mask(selected_experts: torch.Tensor, keep_k: int | None) -> torch.Tensor:
    return build_fixed_keep_mask(selected_experts, keep_k)


def renormalize_surviving_weights(
    routing_weights: torch.Tensor,
    keep_mask: torch.Tensor,
    target_row_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    pruned_weights = routing_weights * keep_mask.to(routing_weights.dtype)
    kept_row_sum = pruned_weights.sum(dim=-1, keepdim=True)
    if target_row_sum is None:
        target_row_sum = torch.ones_like(kept_row_sum)
    else:
        target_row_sum = target_row_sum.to(routing_weights.dtype)

    renorm_weights = torch.zeros_like(routing_weights)
    valid_rows = kept_row_sum.squeeze(-1) > 0
    if valid_rows.any():
        renorm_weights[valid_rows] = (
            pruned_weights[valid_rows] / kept_row_sum[valid_rows]
        ) * target_row_sum[valid_rows]
    return renorm_weights
