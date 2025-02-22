"""Classes and functions which help with evaluation of models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def get_confusion_matrix(y_true, y_pred):
    """Output confusion matrix formatted as a dataframe with column and row labels"""

    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred))
    matrix.columns.name = "Predicted"
    matrix.index.name = "Actual"

    return matrix


def evaluate_model(model, splits, threshold=0.5, include_train=False):
    """Standard evaluation metrics for binary classifier

    Parameters
    ----------
    model: estimator
    splits: [X_train, X_test, y_train, y_test]
    threshold: decision function threshold (between 0 and 1)
    include_train: bool
    """

    X_train = splits[0]
    X_test = splits[1]
    y_train = splits[2]
    y_test = splits[3]

    if include_train:
        Xs = [X_train, X_test]
        ys = [y_train, y_test]
        names = ["Train", "Test"]
    else:
        Xs = [X_test]
        ys = [y_test]
        names = ["Test"]

    for X, y, name in zip(Xs, ys, names):

        pred_proba = model.predict_proba(X)[:, 1]
        pred = (pred_proba >= threshold).astype(np.int8)

        roc_score = roc_auc_score(y, pred_proba)

        print("******************************************************")
        print(f"{name} Metrics, threshold =", threshold)
        print("******************************************************")
        print("Confusion Matrix")
        display(get_confusion_matrix(y, pred))

        print("******************************************************")
        print("Classification report")
        print(classification_report(y, pred, digits=3))

        print("******************************************************")
        print(f"ROC Curve, roc_auc = {roc_score:.4f}")
        fig, ax = plt.subplots()

        RocCurveDisplay.from_estimator(model, X, y, ax=ax)
        ax.plot([0, 1], [0, 1], "r--")

        # Add lines showing where on the roc curve the model is at specified threshold
        fpr, tpr, thresholds = roc_curve(y, pred_proba)
        df2 = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
        idx = (df2["thresholds"] - threshold).abs().idxmin()
        fpr1 = df2.loc[idx, "fpr"]
        tpr1 = df2.loc[idx, "tpr"]

        ax.plot([fpr1, fpr1], [-0.05, tpr1], "--", color="black")
        ax.plot([-0.05, fpr1], [tpr1, tpr1], "--", color="black")

        ax.set_ylabel("True Positive Rate (Recall)")

        plt.show()
        print(f"At threshold = {threshold}")
        print(f"tpr = {tpr1:.4f}, fpr = {fpr1:.4f}")
        print("******************************************************")

        fig2, ax2 = plt.subplots()

        PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax2)

        precision, recall, thresholds2 = precision_recall_curve(y, pred_proba)

        # Add lines showing where on the precision-recall curve the model is at specified threshold
        thresholds2 = np.append(thresholds2, [1])
        df3 = pd.DataFrame(
            {"precision": precision, "recall": recall, "thresholds": thresholds2}
        )
        idx2 = (df3["thresholds"] - threshold).abs().idxmin()
        precision1 = df3.loc[idx2, "precision"]
        recall1 = df3.loc[idx2, "recall"]
        min_precision = df3["precision"].iloc[0]
        min_recall = df3["recall"].iloc[-1]

        ax2.plot(
            [recall1, recall1], [min_precision - 0.05, precision1], "--", color="black"
        )
        ax2.plot(
            [min_recall - 0.05, recall1], [precision1, precision1], "--", color="black"
        )

        ax2.set_xlabel("Recall (True Positive Rate)")

        ax2.legend(loc="upper right")

        plt.show()
        print(f"At threshold = {threshold}")
        print(f"precision = {precision1:.4f}, recall = {recall1:.4f}")
        print("******************************************************")

    return
