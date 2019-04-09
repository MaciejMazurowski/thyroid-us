import numpy as np
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from sklearn.metrics import roc_auc_score

from data import fold_data, augment
from model import multitask_cnn, loss_dict, loss_weights_dict

checkpoints_dir = "/data/checkpoints/<FOLD>/"
logs_dir = "/data/logs/<FOLD>/"

batch_size = 128
epochs = 250
base_lr = 0.001


def train(fold):
    fold_checkpoints_dir = checkpoints_dir.replace("<FOLD>", str(fold))
    fold_logs_dir = logs_dir.replace("<FOLD>", str(fold))

    if not os.path.exists(fold_checkpoints_dir):
        os.makedirs(fold_checkpoints_dir)
    if not os.path.exists(fold_logs_dir):
        os.makedirs(fold_logs_dir)

    x_train, y_train, x_test, y_test = fold_data(fold)

    print("Training and validation data processed.")
    print("Training data shape: {}".format(len(x_train)))
    print("Test data shape: {}".format(len(x_test)))

    model = multitask_cnn()

    optimizer = RMSprop(lr=base_lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=["accuracy"],
    )

    training_log = TensorBoard(
        log_dir=os.path.join(fold_logs_dir, "log"), write_graph=False
    )

    callbacks = [training_log]

    y_train_cancer = y_train["out_cancer"]
    y_test_cancer = y_test[0]

    for e in range(epochs):
        x_train_augmented = augment(x_train)
        model.fit(
            x={"thyroid_input": x_train_augmented},
            y=y_train,
            validation_data=(x_test, y_test),
            batch_size=batch_size,
            epochs=e + 1,
            initial_epoch=e,
            shuffle=True,
            callbacks=callbacks,
        )

        if np.mod(e + 1, 10) == 0:
            y_pred = model.predict(x_train, batch_size=batch_size, verbose=1)
            auc_train = roc_auc_score(y_train_cancer, y_pred[0])
            y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
            auc_test = roc_auc_score(y_test_cancer, y_pred[0])
            with open(os.path.join(fold_logs_dir, "auc.txt"), "a") as auc_file:
                auc_file.write("{},{}\n".format(auc_train, auc_test))

    model.save(os.path.join(fold_checkpoints_dir, "weights.h5"))

    print("Training fold {} completed.".format(fold))


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    device = "/gpu:" + sys.argv[1]
    with tf.device(device):
        train(int(sys.argv[2]))
