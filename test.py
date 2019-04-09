import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve

from data import test_data, test_pids
from focal_loss import focal_loss
from plots import plot_roc

checkpoints_dir = "/data/test/checkpoints/"
batch_size = 128
nb_categories = 1


def predict():
    weights_path = os.path.join(checkpoints_dir, "weights.h5")

    net = load_model(weights_path, custom_objects={"focal_loss_fixed": focal_loss()})

    X_test, y_test = test_data()

    preds = net.predict(X_test, batch_size=batch_size, verbose=1)

    return preds[0], y_test[0]


def test():
    predictions, targets = predict()

    cases_predictions = {}
    cases_targets = {}
    pids = test_pids()
    for i in range(len(pids)):
        pid = pids[i]
        prev_pred = cases_predictions.get(pid, np.zeros(nb_categories))
        preds = predictions[i]
        cases_predictions[pid] = prev_pred + preds
        cases_targets[pid] = targets[i]

    y_pred = []
    y_true = []
    y_id = []
    for pid in cases_predictions:
        y_pred.append(cases_predictions[pid][0])
        y_true.append(cases_targets[pid])
        y_id.append(pid)

    with open("./predictions_test.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["ID", "Prediction", "Cancer"])
        for pid, prediction, gt in zip(y_id, y_pred, y_true):
            pid = pid.lstrip("0")
            csvwriter.writerow([pid, prediction, gt[0]])

    plot_roc(y_true, y_pred, figname="roc_test.png")


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    device = "/gpu:" + sys.argv[1]
    with tf.device(device):
        test()
