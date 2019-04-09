import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc(y_true, y_pred, figname="roc.png"):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print("roc auc = {}".format(roc_auc))

    plt.rcParams.update({"font.size": 24})

    fig = plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figname, bbox_inches="tight")
    plt.close(fig)
