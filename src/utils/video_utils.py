# src/utils/video_utils.py
import torch

# kun for smoke_test, kan slette denne etterhvert


@torch.no_grad()
def logits_mean_over_time(model, x, *, batch_size: int = 16):
    """
    Returnerer [B,K]. Støtter:
      - list/tuple av T frames [C,H,W] eller [B,C,H,W]
      - 3D: [C,H,W] (enkeltbilde)
      - 4D: [T,C,H,W] ELLER [C,T,H,W] ELLER [B,C,H,W]
      - 5D: [B,C,T,H,W]
    """

    def _as_batched_frames(t):
        # t: [C,H,W] -> [1,C,H,W]
        if isinstance(t, torch.Tensor) and t.dim() == 3:
            return t.unsqueeze(0)
        return t  # antas allerede [B,C,H,W]

    # 1) Liste/tuple av T frames
    if isinstance(x, (list, tuple)):
        frames = [_as_batched_frames(f) for f in x]  # hver [B?,C,H,W]
        # Samle til [sumB, C, H, W] for effektiv fwd:
        fb = (
            torch.cat(frames, dim=0)
            if frames and frames[0].dim() == 4
            else torch.stack(frames, dim=0)
        )
        # Hvis stack via stack([...]) ga [T,1,C,H,W] -> gjør til [T,C,H,W]
        if fb.dim() == 5 and fb.shape[1] == 1:
            fb = fb.squeeze(1)
        # Nå forventer vi enten [T,C,H,W] eller [B,C,H,W]
        if fb.dim() == 4 and fb.shape[0] != frames[0].shape[0]:  # [T,C,H,W]
            # kjør i minibatcher over T
            T = fb.shape[0]
            logits_list = []
            for s in range(0, T, batch_size):
                logits_list.append(model(fb[s : s + batch_size]))  # [t,C,H,W] -> [t,K]
            logits_t = torch.cat(logits_list, dim=0).unsqueeze(0)  # [1,T,K]
            return logits_t.mean(1)  # [1,K]
        else:
            # antas [B,C,H,W] (allerede batchet); behandle som enkelt “frame batch”
            return model(fb)

    # 2) 5D: [B,C,T,H,W]
    if isinstance(x, torch.Tensor) and x.dim() == 5:
        B, C, T, H, W = x.shape
        x_bt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        logits_bt = []
        for s in range(0, B * T, batch_size):
            logits_bt.append(model(x_bt[s : s + batch_size]))  # [bs,K]
        logits_bt = torch.cat(logits_bt, dim=0).view(B, T, -1)  # [B,T,K]
        return logits_bt.mean(1)  # [B,K]

    # 3) 4D: kan være [B,C,H,W] ELLER [T,C,H,W] ELLER [C,T,H,W]
    if isinstance(x, torch.Tensor) and x.dim() == 4:
        B_or_T, C, H, W = x.shape
        # Heuristikk: hvis B_or_T er lite (typ <= 10) og C i {1,3}, behandle som T
        if C in (1, 3) and B_or_T > 1 and B_or_T <= 64:
            # anta [T,C,H,W]
            T = B_or_T
            logits_list = []
            for s in range(0, T, batch_size):
                logits_list.append(model(x[s : s + batch_size]))  # [bs,K]
            logits_t = torch.cat(logits_list, dim=0).unsqueeze(0)  # [1,T,K]
            return logits_t.mean(1)  # [1,K]
        # ellers anta [B,C,H,W]
        return model(x)  # [B,K]

    # 4) 3D: [C,H,W] -> legg til batch
    if isinstance(x, torch.Tensor) and x.dim() == 3:
        return model(x.unsqueeze(0))  # [1,K]

    raise ValueError(
        f"Ukjent input-shape til logits_mean_over_time: {getattr(x, 'shape', type(x))}"
    )
