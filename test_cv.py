import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve

from data import fold_data, fold_pids
from focal_loss import focal_loss
from plots import plot_roc

checkpoints_dir = "/data/checkpoints/<FOLD>/"
weights_file = "weights.h5"
batch_size = 128
nb_categories = 1


def predict(fold):
    fold_checkpoints_dir = checkpoints_dir.replace("<FOLD>", str(fold))
    weights_path = os.path.join(fold_checkpoints_dir, weights_file)

    net = load_model(weights_path, custom_objects={"focal_loss_fixed": focal_loss()})

    x_train, y_train, x_test, y_test = fold_data(fold)

    preds = net.predict(x_test, batch_size=batch_size, verbose=1)
    y = y_test[0]

    return preds[0], y


def test(folds=10):
    pids = []
    predictions = np.zeros((0, nb_categories))
    targets = []
    pid_fold = []

    for f in range(folds):
        preds, t = predict(f)
        predictions = np.vstack((predictions, preds))
        pids.extend(fold_pids(f))
        targets.extend(t)
        pid_fold.extend([f] * len(t))

    print("{} images".format(len(pids)))

    cases_predictions = {}
    cases_targets = {}
    cases_folds = {}
    for i in range(len(pids)):
        pid = pids[i]
        prev_pred = cases_predictions.get(pid, np.zeros(nb_categories))
        preds = predictions[i]
        cases_predictions[pid] = prev_pred + preds
        cases_targets[pid] = targets[i]
        cases_folds[pid] = pid_fold[i]

    print("{} cases".format(len(cases_predictions)))

    y_pred = []
    y_true = []
    y_id = []
    y_fold = []
    for pid in cases_predictions:
        y_pred.append(cases_predictions[pid][0])
        y_true.append(cases_targets[pid])
        y_id.append(pid)
        y_fold.append(cases_folds[pid])

    with open("./predictions_cv.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["ID", "Prediction", "Cancer", "Fold"])
        for pid, prediction, gt, f in zip(y_id, y_pred, y_true, y_fold):
            pid = pid.lstrip("0")
            csvwriter.writerow([pid, prediction, gt[0], f])

    plot_roc(y_true, y_pred, figname="roc_cv.png")


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    device = "/gpu:" + sys.argv[1]
    with tf.device(device):
        test()
