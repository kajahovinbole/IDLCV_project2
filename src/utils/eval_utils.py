import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


@torch.no_grad()
def evaluate_loader(
    model, loader, device, *, compute_confusion=False, class_names=None
):
    """
    Felles eval for både val (i trening) og test (i eval.py).
    Returnerer alltid loss/acc/f1_macro. Hvis compute_confusion=True,
    returneres også confusion_matrix, per_class_acc og class_names.
    """
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    y_true, y_pred = [], []

    for x, y in loader:
        # støtt både list/tuple av T frames og samlet tensor
        if isinstance(x, (list, tuple)):
            x = [t.to(device, non_blocking=True) for t in x]
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(1)

        loss_sum += float(loss.item())
        total += y.numel()
        correct += int((pred == y).sum().item())

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    f1m = float(f1_score(y_true, y_pred, average="macro"))

    out = {
        "loss": loss_sum / max(1, total),
        "acc": acc,
        "f1_macro": f1m,
    }

    if compute_confusion:
        K = len(class_names) if class_names is not None else None
        cm = confusion_matrix(y_true, y_pred, labels=list(range(K)) if K else None)
        per_class_acc = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
        out.update(
            {
                "confusion_matrix": cm.tolist(),
                "per_class_acc": per_class_acc,
                "class_names": class_names,
            }
        )

    return out
