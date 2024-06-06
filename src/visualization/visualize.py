import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.calibration import calibration_curve


def plot_calibration_curve(
    y_true: np.array,
    y_pred_proba: np.array,
    model_name: str,
    figsize: Tuple[int, int],
    path: str,
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration plot")
    ax.legend()
    ax.grid(True)

    image = fig

    # Save figure
    fig.savefig(path)

    # Close plot
    plt.close(fig)

    return image
