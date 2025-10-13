import torch


def logits_mean_over_time(model, x):
    """x: [B,C,T,H,W] eller list med T tensors [B,C,H,W]"""
    if isinstance(x, (list, tuple)):
        # liste av T frames: [B,C,H,W] hver
        logits_t = [model(f) for f in x]  # T x [B,K]
        return torch.stack(logits_t, dim=1).mean(1)  # [B,K]
    else:
        # stacket: [B,C,T,H,W] -> [B*T, C, H, W]
        B, C, T, H, W = x.shape
        x_bt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        logits_bt = model(x_bt)  # [B*T,K]
        return logits_bt.view(B, T, -1).mean(1)  # [B,K]
