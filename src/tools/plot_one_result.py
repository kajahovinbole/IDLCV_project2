#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
MODEL_NAME = "per_frame"
HIST_PATH  = Path(f"results/history/{MODEL_NAME}.json")           # fra treningen din
EVAL_PATH  = Path(f"results/eval/{MODEL_NAME}_test.json")         # fra eval
OUT_DIR    = Path(f"results/plots/{MODEL_NAME}")

def load_json(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Fant ikke {p}")
    with open(p, "r") as f:
        return json.load(f)

def plot_valF1_vs_testF1(history: dict, eval_res: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs   = [e["epoch"] for e in history["epochs"]]
    val_f1   = [e["val_f1"] for e in history["epochs"]]
    test_f1  = float(eval_res["f1_macro"])

    plt.figure()
    plt.plot(epochs, val_f1, "-", linewidth=2, label="Val F1 (macro)")
    plt.axhline(test_f1, linestyle="--", label=f"Test F1 = {test_f1:.3f}")
    plt.xlabel("Epoch"); plt.ylabel("F1 (macro)")
    plt.title(f"Validation F1 vs Test F1 – {history['model']}")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{MODEL_NAME}_valF1_vs_testF1.png", dpi=200)
    plt.close()

def plot_confusion_matrix(eval_res: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.array(eval_res["confusion_matrix"])
    classes = eval_res["class_names"]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion matrix – {eval_res['model']}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{MODEL_NAME}_confusion_matrix.png", dpi=200)
    plt.close()

def plot_per_class_accuracy(eval_res: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    accs = eval_res["per_class_acc"]
    classes = eval_res["class_names"]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(accs)), accs)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylim(0,1); plt.xlabel("Class"); plt.ylabel("Accuracy")
    plt.title(f"Per-class accuracy – {eval_res['model']}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{MODEL_NAME}_per_class_acc.png", dpi=200)
    plt.close()

def plot_acc_loss_from_history(history: dict):
    """Valgfritt: fortsatt nyttig å ha acc/loss–kurver fra history.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ep   = [e["epoch"] for e in history["epochs"]]
    tr_a = [e["train_acc"] for e in history["epochs"]]
    va_a = [e["val_acc"]   for e in history["epochs"]]
    tr_l = [e["train_loss"] for e in history["epochs"]]
    va_l = [e["val_loss"]   for e in history["epochs"]]

    # Acc
    plt.figure()
    plt.plot(ep, tr_a, "--", alpha=0.8, label="Train acc")
    plt.plot(ep, va_a, "-",  alpha=0.9, label="Val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"Train/Val Accuracy – {history['model']}")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{MODEL_NAME}_acc_vs_epoch.png", dpi=200)
    plt.close()

    # Loss
    plt.figure()
    plt.plot(ep, tr_l, "--", alpha=0.8, label="Train loss")
    plt.plot(ep, va_l, "-",  alpha=0.9, label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Train/Val Loss – {history['model']}")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{MODEL_NAME}_loss_vs_epoch.png", dpi=200)
    plt.close()

def main():
    hist = load_json(HIST_PATH)   # inneholder val_f1 per epoch (du lagrer det allerede)
    eva  = load_json(EVAL_PATH)   # inneholder test f1/acc/loss + CM + per_class_acc

    plot_valF1_vs_testF1(hist, eva)     # <- F1-fokus
    plot_confusion_matrix(eva)
    plot_per_class_accuracy(eva)
    plot_acc_loss_from_history(hist)    # (valgfritt, men ofte kjekt)

    print(f"[INFO] Ferdig – figurer lagret i {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()