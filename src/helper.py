import torch
import numpy as np


def _to_tensor(x):
    """Convert various types to torch.Tensor (CPU)."""
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty list")
        if all(isinstance(el, torch.Tensor) for el in x):
            return torch.stack([el.cpu() for el in x])
        if all(isinstance(el, np.ndarray) for el in x):
            return torch.from_numpy(np.stack(x))
        converted = []
        for el in x:
            if isinstance(el, torch.Tensor):
                converted.append(el.cpu())
            elif isinstance(el, np.ndarray):
                converted.append(torch.from_numpy(el))
            else:
                converted.append(torch.tensor(el))
        return torch.stack(converted)
    return torch.tensor(x)


def _extract_context_target_mask(batch):
    """Extract (context, target, mask) from batch."""
    if isinstance(batch, dict):
        ctx_keys = ("past_target", "past_targets", "past", "history", "x")
        tgt_keys = ("future_target", "future_targets", "future", "y", "target")
        mask_keys = ("future_mask", "mask", "future_masks")
        context = next((batch[k] for k in ctx_keys if k in batch), None)
        target = next((batch[k] for k in tgt_keys if k in batch), None)
        future_mask = next((batch[k] for k in mask_keys if k in batch), None)
        return context, target, future_mask
    
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) >= 3:
            return batch[0], batch[2] if len(batch) > 2 else None, batch[3] if len(batch) > 3 else None
    
    raise TypeError(f"Unsupported batch format: {type(batch)}")


def _to_torch(x):
    """Convert pipeline outputs to torch.Tensor (CPU)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, list):
        if len(x) == 0:
            return None
        if all(isinstance(el, np.ndarray) for el in x):
            return torch.from_numpy(np.stack(x))
        if all(isinstance(el, torch.Tensor) for el in x):
            return torch.stack([el.cpu() for el in x])
        return torch.tensor(x)
    raise TypeError(f"Unsupported type: {type(x)}")


def align_forecast_to_target(q_out, mean_out, n_features, pred_len, quantile_levels):
    """
    Align forecast output to [B, pred_len, n_features] format.
    Returns CPU tensor or None.
    """
    median_idx = len(quantile_levels) // 2
    
    if q_out is not None:
        if q_out.ndim == 4:
            B, a, b, c = q_out.shape
            # Case: [B, n_var, pred_len, n_q]
            if a == n_features and b == pred_len and c == len(quantile_levels):
                med = q_out[..., median_idx]  # [B, n_var, pred_len]
                return med.permute(0, 2, 1).contiguous()
            # Case: [B, pred_len, n_q, n_var]
            if a == pred_len and b == len(quantile_levels) and c == n_features:
                return q_out[:, :, median_idx, :].contiguous()
            # Try permutations
            for perm in [(0,1,2,3),(0,2,1,3),(0,3,1,2),(0,2,3,1)]:
                try:
                    cand = q_out.permute(*perm)
                    if (cand.ndim == 4 and cand.shape[1] == pred_len and 
                        cand.shape[3] == n_features and cand.shape[2] == len(quantile_levels)):
                        return cand[:, :, median_idx, :].contiguous()
                except:
                    pass
        
        elif q_out.ndim == 3:
            s = q_out.shape
            if s[1] == pred_len and s[2] == n_features:
                return q_out.contiguous()
            if s[1] == n_features and s[2] == pred_len:
                return q_out.permute(0, 2, 1).contiguous()
            if s[1] == n_features and s[2] == len(quantile_levels):
                med = q_out[:, :, median_idx]  # [B, n_var]
                return med.unsqueeze(1).expand(-1, pred_len, -1).contiguous()
    
    # Fallback to mean_out
    if mean_out is not None and isinstance(mean_out, torch.Tensor):
        if mean_out.ndim == 3:
            if mean_out.shape[1] == n_features and mean_out.shape[2] == pred_len:
                return mean_out.permute(0, 2, 1).contiguous()
            if mean_out.shape[1] == pred_len and mean_out.shape[2] == n_features:
                return mean_out.contiguous()
    
    return None

print("Helper functions loaded successfully")