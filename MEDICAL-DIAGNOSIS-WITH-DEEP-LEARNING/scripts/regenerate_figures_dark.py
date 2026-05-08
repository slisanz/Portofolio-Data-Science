"""Regenerate all report figures with the dark theme used by the Streamlit app.

Reads existing artefacts in `reports/` plus images from `chest_xray/` and re-renders
each figure with a palette aligned to `.streamlit/config.toml`. Run once after
changing the Streamlit theme so the dashboard images match the live app.

    python scripts/regenerate_figures_dark.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "chest_xray"

BG = "#0E1117"
PANEL = "#1E2530"
FG = "#FAFAFA"
PRIMARY = "#2E86AB"
ACCENT = "#E63946"
GRID = "#2A3441"
PALETTE = [PRIMARY, ACCENT, "#F4A261", "#90BE6D"]

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "text.color": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": FG,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "savefig.bbox": "tight",
        "savefig.dpi": 120,
        "font.family": "sans-serif",
    }
)
sns.set_palette(PALETTE)

random.seed(42)
np.random.seed(42)


def _save(name: str) -> None:
    out = FIG_DIR / name
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def class_distribution(manifest: pd.DataFrame) -> None:
    print("class distribution")
    pivot = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    pivot = pivot[["NORMAL", "PNEUMONIA"]].loc[["train", "val", "test"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax, color=[PRIMARY, ACCENT], width=0.7, edgecolor="none")
    ax.set_title("Class distribution per split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of images")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Class", loc="upper right")
    for container in ax.containers:
        ax.bar_label(container, padding=2, color=FG, fontsize=9)
    _save("class_distribution.png")


def subtype_distribution(manifest: pd.DataFrame) -> None:
    print("subtype distribution")
    train = manifest[manifest["split"] == "train"].copy()
    counts = train["subtype"].value_counts().reindex(["normal", "bacteria", "virus"]).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[PRIMARY, ACCENT, "#F4A261"], edgecolor="none")
    ax.set_title("Pneumonia subtype distribution (train split)")
    ax.set_xlabel("Subtype")
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{int(bar.get_height())}", ha="center", color=FG, fontsize=10)
    _save("subtype_distribution.png")


def image_dimensions(manifest: pd.DataFrame, sample: int = 600) -> None:
    print(f"image dimensions (sampling {sample})")
    sub = manifest.sample(min(sample, len(manifest)), random_state=42)
    widths, heights = [], []
    for fp in sub["filepath"]:
        try:
            with Image.open(fp) as im:
                widths.append(im.width)
                heights.append(im.height)
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(widths, heights, alpha=0.55, s=18, color=PRIMARY, edgecolor="none")
    ax.axvline(224, color=ACCENT, linestyle="--", linewidth=1, label="DenseNet input (224)")
    ax.axhline(224, color=ACCENT, linestyle="--", linewidth=1)
    ax.set_title(f"Image dimensions ({len(widths)} sampled)")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    _save("image_dimensions.png")


def intensity_distribution(manifest: pd.DataFrame, per_class: int = 200) -> None:
    print(f"intensity distribution (per_class={per_class})")
    means = {"NORMAL": [], "PNEUMONIA": []}
    for label in means:
        sub = manifest[manifest["label"] == label].sample(per_class, random_state=42)
        for fp in sub["filepath"]:
            try:
                with Image.open(fp) as im:
                    arr = np.asarray(im.convert("L"), dtype=np.float32)
                    means[label].append(arr.mean())
            except Exception:
                continue

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(means["NORMAL"], bins=35, color=PRIMARY, alpha=0.75, label="NORMAL", edgecolor="none")
    ax.hist(means["PNEUMONIA"], bins=35, color=ACCENT, alpha=0.65, label="PNEUMONIA", edgecolor="none")
    ax.set_title("Mean pixel intensity by class")
    ax.set_xlabel("Mean intensity (0-255)")
    ax.set_ylabel("Number of images")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    _save("intensity_distribution.png")


def sample_grid(manifest: pd.DataFrame, label: str, subtype: str | None, out_name: str, title: str) -> None:
    print(f"sample grid {out_name}")
    sub = manifest[manifest["label"] == label]
    if subtype is not None:
        sub = sub[sub["subtype"] == subtype]
    if len(sub) == 0:
        print(f"  skip {out_name}: empty subset")
        return
    sub = sub.sample(min(9, len(sub)), random_state=42)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(title, color=FG, fontsize=14, y=0.98)
    for ax, fp in zip(axes.ravel(), sub["filepath"]):
        try:
            with Image.open(fp) as im:
                ax.imshow(np.asarray(im.convert("L")), cmap="gray")
        except Exception:
            ax.set_facecolor(PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(out_name)


def training_curves(history_csv: Path, title: str, out_name: str) -> None:
    print(f"training curves {out_name}")
    if not history_csv.exists():
        print(f"  skip {out_name}: {history_csv.name} not found")
        return
    df = pd.read_csv(history_csv)
    epochs = np.arange(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, color=FG, fontsize=14)

    if "loss" in df:
        axes[0].plot(epochs, df["loss"], color=PRIMARY, label="train", linewidth=2)
    if "val_loss" in df:
        axes[0].plot(epochs, df["val_loss"], color=ACCENT, label="val", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].legend()

    metric_col = "auc" if "auc" in df else ("accuracy" if "accuracy" in df else None)
    val_col = "val_" + metric_col if metric_col else None
    if metric_col:
        axes[1].plot(epochs, df[metric_col], color=PRIMARY, label="train", linewidth=2)
    if val_col and val_col in df:
        axes[1].plot(epochs, df[val_col], color=ACCENT, label="val", linewidth=2)
    axes[1].set_title((metric_col or "metric").upper())
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_col or "metric")
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(out_name)


def confusion_matrix_fig(metrics: dict) -> None:
    print("confusion matrix")
    cm = np.array(metrics["confusion_matrix_densenet"])
    labels = ["NORMAL", "PNEUMONIA"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="cividis")
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (threshold {metrics['chosen_threshold']:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] > cm.max() / 2 else FG, fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=FG)
    _save("confusion_matrix.png")


def model_curves() -> None:
    print("model curves (ROC, PR, threshold tuning) — running test inference")
    try:
        import tensorflow as tf
        from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
    except ImportError as e:
        print(f"  skip: {e}")
        return

    model_path = PROJECT_ROOT / "models" / "best_model.keras"
    if not model_path.exists():
        print("  skip: best_model.keras not found")
        return

    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        print("  skip: test data not found")
        return

    print("  loading model")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("  building test pipeline")
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
    )

    def prep(x, y):
        x = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
        x = tf.keras.applications.densenet.preprocess_input(x)
        return x, y

    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    print("  predicting test set")
    y_true, y_score = [], []
    for x, y in ds:
        y_score.append(model.predict(x, verbose=0).ravel())
        y_true.append(y.numpy().ravel())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(np.trapz(tpr, fpr))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PRIMARY, linewidth=2, label=f"DenseNet121 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    _save("roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=PRIMARY, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.grid(linestyle="--", alpha=0.4)
    _save("pr_curve.png")

    thresholds = np.linspace(0.05, 0.95, 91)
    f1s, recalls, precisions = [], [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        recalls.append(tp / (tp + fn) if (tp + fn) else 0)
        precisions.append(tp / (tp + fp) if (tp + fp) else 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresholds, recalls, color=PRIMARY, linewidth=2, label="Recall")
    ax.plot(thresholds, precisions, color=ACCENT, linewidth=2, label="Precision")
    ax.plot(thresholds, f1s, color="#F4A261", linewidth=2, label="F1")
    ax.axvline(0.72, color=FG, linestyle=":", linewidth=1, label="Chosen threshold (0.72)")
    ax.axhline(0.95, color=GRID, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold tuning")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    _save("threshold_tuning.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = REPORTS_DIR / "image_manifest.csv"
    metrics_path = REPORTS_DIR / "metrics.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found at {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    class_distribution(manifest)
    subtype_distribution(manifest)
    image_dimensions(manifest)
    intensity_distribution(manifest)
    sample_grid(manifest, "NORMAL", "normal", "samples_normal.png", "NORMAL samples")
    sample_grid(manifest, "PNEUMONIA", "bacteria", "samples_bacteria.png", "Bacterial pneumonia samples")
    sample_grid(manifest, "PNEUMONIA", "virus", "samples_virus.png", "Viral pneumonia samples")

    training_curves(REPORTS_DIR / "baseline_history.csv",
                    "Baseline custom CNN training", "baseline_training_curves.png")
    training_curves(REPORTS_DIR / "densenet_history.csv",
                    "DenseNet121 transfer learning", "densenet_training_curves.png")

    if metrics:
        confusion_matrix_fig(metrics)

    model_curves()


if __name__ == "__main__":
    main()
