import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.calibration import calibration_curve
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from scipy import stats


def plot_calibration_curve(
    y_true: np.array,
    y_pred_proba: np.array,
    model_name: str,
    path: str = None,
    n_bins=10,
) -> plt.figure:
    """
    Plot calibration curve.

    Args:
        y_pred_proba (np.array): Predicted probabilities for the positive class (default).
        y_true (np.array): True binary labels (0 for not default, 1 for default).
        model_name (str): Name of the model for labeling the plot.
        figsize (Tuple[int, int]): Size of the plot.
        path (str): Path to store plot image.
        n_bins (int): Number of bins to use for calibration curve.
    Return:
        plt.Axes: Matplotlib axis object.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration plot")
    ax.legend()
    ax.grid(True)

    image = fig

    if path is not None:
        # Save figure
        fig.savefig(path)

    # Close plot
    plt.close(fig)

    return image


def plot_pred_proba_distribution(
    y_true: np.array, y_pred_proba: np.array
) -> plt.Axes:
    """
    Plot the predicted probability distributions for the default and non-default classes.

    Args:
        y_pred_proba (np.array): Predicted probabilities for the positive class (default).
        y_true (np.array): True binary labels (0 for not default, 1 for default).
        figsize (Tuple[int, int]): size of the plot.
    Return:
        plt.Axes: Matplotlib axis object.
    """
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()

    sns.histplot(
        y_pred_proba[y_true == 0],
        label="Not Default",
        kde=True,
        color="blue",
        alpha=0.6,
        bins=30,
        ax=ax,
    )
    sns.histplot(
        y_pred_proba[y_true == 1],
        label="Default",
        kde=True,
        color="red",
        alpha=0.6,
        bins=30,
        ax=ax,
    )

    ax.set_title(
        "Predicted Probability Distributions for Default and Not Default",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted Probability", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.legend(title="Label", fontsize=12, title_fontsize="13")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    image = fig

    return image


def plot_roc_auc_curve(
    y_true: np.array, y_pred_proba: np.array
) -> plt.Axes:
    """
    Plot the ROC curve and calculate the AUC.

    Args:
        y_pred_proba (np.array): Predicted probabilities for the positive class (default).
        y_true (np.array): True binary labels (0 for not default, 1 for default).
        figsize (Tuple[int, int]): size of the plot.
    Return:
        plt.Axes: Matplotlib axis object.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    ax.set_title(
        "Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(axis="both", linestyle="--", alpha=0.7)

    image = fig

    return image


def plot_precision_recall_curve(
    y_true: np.array, y_pred_proba: np.array
) -> plt.Axes:
    """
    Plot the Precision-Recall curve and calculate the Average Precision (AP).

    Args:
        y_pred_proba (np.array): Predicted probabilities for the positive class (default).
        y_true (np.array): True binary labels (0 for not default, 1 for default).
        figsize (Tuple[int, int]): size of the plot.
    Return:
        plt.Axes: Matplotlib axis object.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.plot(recall, precision, lw=2, label=f"PR curve (AP = {pr_auc:.2f})")

    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(axis="both", linestyle="--", alpha=0.7)

    image = fig

    return image


def plot_ks(
    y_true: np.array, y_pred_proba: np.array
) -> plt.Axes:
    """
    Plot the Kolmogorov-Smirnov (KS) statistic.

    Args:
        y_pred_proba (np.array): Predicted probabilities for the positive class (default).
        y_true (np.array): True binary labels (0 for not default, 1 for default).
        figsize (Tuple[int, int]): size of the plot.
    Return:
        plt.Axes: Matplotlib axis object.
    """
    y_pred_proba_not_default = y_pred_proba[y_true == 0]
    y_pred_proba_default = y_pred_proba[y_true == 1]

    ks_stat, p_value = stats.ks_2samp(y_pred_proba_not_default, y_pred_proba_default)

    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.hist(
        y_pred_proba_not_default,
        bins=50,
        density=True,
        histtype="step",
        cumulative=True,
        label="Not Default",
        linewidth=2,
    )
    ax.hist(
        y_pred_proba_default,
        bins=50,
        density=True,
        histtype="step",
        cumulative=True,
        label="Default",
        linewidth=2,
    )
    ax.set_title("KS Plot")
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")
    ax.legend(title=f"KS Statistic: {ks_stat:.3f}, P-value: {p_value:.3f}")

    image = fig

    return image


def plot_calibration_curve(
    y_true: np.array,
    y_pred_proba: np.array,
    model_name: str,
    n_bins=10,
) -> plt.Axes:
    """
    Plot calibration curve.

    Args:
        y_pred_proba (np.array): Predicted probabilities for the positive class (default).
        y_true (np.array): True binary labels (0 for not default, 1 for default).
        model_name (str): Name of the model for labeling the plot.
        figsize (Tuple[int, int]): size of the plot.
        n_bins (int): Number of bins to use for calibration curve.
    Return:
        plt.Axes: Matplotlib axis object.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration plot")
    ax.legend()
    ax.grid(True)

    image = fig

    return image
