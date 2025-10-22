import re
import glob
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LOG_DIR = Path("logs")
OUT_DIR = Path("results/plots")
EVAL_DIR = Path("results/eval")


# -------------------- eksisterende kode --------------------
def find_latest_out_file():
    files = glob.glob(str(LOG_DIR / "*.out"))
    if not files:
        raise FileNotFoundError("Ingen .out-filer i logs/")
    latest = max(files, key=os.path.getmtime)
    print(f"[INFO] Bruker nyeste loggfil: {latest}")
    return latest


def parse_models_from_log(fp):
    model_pat = re.compile(r"\[INFO\]\s+Starter trening for modell:\s*(.+)")
    epoch_pat = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\(([\d.]+)\)\s+\|\s+val\s+([\d.]+)\s+\(([\d.]+)\)"
    )
    data = {}
    cur = None
    with open(fp, "r") as f:
        for line in f:
            m_model = model_pat.search(line)
            if m_model:
                name = m_model.group(1).strip()
                cur = {
                    "epochs": [],
                    "train_acc": [],
                    "val_acc": [],
                    "train_loss": [],
                    "val_loss": [],
                }
                data[name] = cur
                continue
            m_ep = epoch_pat.search(line)
            if m_ep and cur is not None:
                e, tr_a, tr_l, va_a, va_l = m_ep.groups()
                cur["epochs"].append(int(e))
                cur["train_acc"].append(float(tr_a))
                cur["train_loss"].append(float(tr_l))
                cur["val_acc"].append(float(va_a))
                cur["val_loss"].append(float(va_l))
    data = {k: v for k, v in data.items() if v["epochs"]}
    if not data:
        raise ValueError("Fant ingen epoch-data i loggfilen.")
    return data


def plot_curves(models_dict, outdir=OUT_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    model_names = list(models_dict.keys())
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(model_names)}

    # --- ACCURACY ---
    plt.figure()
    for name, h in models_dict.items():
        c = color_map[name]
        plt.plot(h["epochs"], h["train_acc"], linestyle="--", color=c, alpha=0.8)
        plt.plot(
            h["epochs"],
            h["val_acc"],
            linestyle="-",
            color=c,
            linewidth=2.2,
            alpha=0.9,
            label=name,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(outdir / "acc_vs_epoch.png", dpi=200)

    # --- LOSS ---
    plt.figure()
    for name, h in models_dict.items():
        c = color_map[name]
        plt.plot(h["epochs"], h["train_loss"], linestyle="--", color=c, alpha=0.8)
        plt.plot(
            h["epochs"],
            h["val_loss"],
            linestyle="-",
            color=c,
            linewidth=2.2,
            alpha=0.9,
            label=name,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(outdir / "loss_vs_epoch.png", dpi=200)

    print("\n[RESULTATER]")
    for name, h in models_dict.items():
        va = h["val_acc"]
        ep = h["epochs"]
        best_idx = max(range(len(va)), key=lambda i: va[i])
        print(f"- {name:12s} | best val_acc={va[best_idx]:.3f} @ epoch {ep[best_idx]}")


# -------------------- NY DEL: eval-visualisering --------------------


def load_eval_results(eval_dir=EVAL_DIR):
    files = glob.glob(str(eval_dir / "*_test.json"))
    if not files:
        raise FileNotFoundError(f"Ingen eval-filer i {eval_dir}")
    results = []
    for f in files:
        with open(f, "r") as jf:
            results.append(json.load(jf))
    return results


def plot_eval_summary(results, outdir=OUT_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in results]
    accs = [r["acc"] for r in results]
    f1s = [r["f1_macro"] for r in results]
    losses = [r["loss"] for r in results]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, accs, width, label="Accuracy")
    plt.bar(x, f1s, width, label="F1-macro")
    plt.bar(x + width, losses, width, label="Loss")
    plt.xticks(x, models, rotation=30)
    plt.ylabel("Score / Loss")
    plt.title("Eval: Accuracy, F1, and Loss per model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "eval_summary_barplot.png", dpi=200)


def plot_per_class_accuracy(results, outdir=OUT_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if "per_class_acc" not in r:
            continue
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(r["per_class_acc"])), r["per_class_acc"])
        plt.xticks(range(len(r["class_names"])), r["class_names"], rotation=45)
        plt.ylim(0, 1)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title(f"Per-class accuracy: {r['model']}")
        plt.tight_layout()
        plt.savefig(outdir / f"{r['model']}_per_class_acc.png", dpi=200)


def plot_confusion_matrix(result, outdir=OUT_DIR):
    """Plotter confusion matrix for én modell."""
    cm = np.array(result["confusion_matrix"])
    classes = result["class_names"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix: {result['model']}")
    plt.tight_layout()
    plt.savefig(outdir / f"{result['model']}_confusion_matrix.png", dpi=200)
    plt.close()


# -------------------- MAIN --------------------
# bruk denne for å teste for flere modeller samtidig
# if __name__ == "__main__":
#     # treningskurver
#     latest = find_latest_out_file()
#     models = parse_models_from_log(latest)
#     plot_curves(models)

#     # eval plots
#     eval_results = load_eval_results()
#     plot_eval_summary(eval_results)
#     plot_per_class_accuracy(eval_results)

#     # 🔹 Lag confusion matrix for alle modeller
#     for r in eval_results:
#         if "confusion_matrix" in r:
#             plot_confusion_matrix(r)
#     print(f"[INFO] Ferdig – figurer lagret i {OUT_DIR}")


# bruk denne for å teste for én modell av gangen (TARGET_MODEL)
if __name__ == "__main__":
    TARGET_MODEL = "per_frame_agg"  # <-- skriv inn navnet på modellen du trente

    # treningskurver
    latest = find_latest_out_file()
    models = parse_models_from_log(latest)
    if TARGET_MODEL in models:
        models = {TARGET_MODEL: models[TARGET_MODEL]}
    else:
        print(f"[WARN] Fant ikke {TARGET_MODEL} i loggfilen – plotter alle modeller.")

    plot_curves(models)

    # eval plots
    eval_results = load_eval_results()
    eval_results = [r for r in eval_results if r["model"] == TARGET_MODEL]

    if not eval_results:
        print(f"[WARN] Ingen eval-resultater for {TARGET_MODEL}")
    else:
        plot_eval_summary(eval_results)
        plot_per_class_accuracy(eval_results)
        for r in eval_results:
            if "confusion_matrix" in r:
                plot_confusion_matrix(r)

    print(f"[INFO] Ferdig – figurer lagret i {OUT_DIR}")
