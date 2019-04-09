import numpy as np
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from sklearn.metrics import roc_auc_score

from data import augment, train_data, test_data
from model import multitask_cnn, loss_dict, loss_weights_dict

checkpoints_dir = "/data/test/checkpoints/"
logs_dir = "/data/test/logs/"

batch_size = 128
epochs = 250
base_lr = 0.001


def train():
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    X_train, y_train = train_data()
    X_test, y_test = test_data()

    print("Training and validation data processed.")

    model = multitask_cnn()

    optimizer = RMSprop(lr=base_lr)
    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=["accuracy"],
    )

    training_log = TensorBoard(log_dir=os.path.join(logs_dir, "log"), write_graph=False)

    callbacks = [training_log]

    for e in range(epochs):
        X_train_augmented = augment(X_train)
        model.fit(
            {"thyroid_input": X_train_augmented},
            y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=e + 1,
            initial_epoch=e,
            shuffle=True,
            callbacks=callbacks,
        )

        if np.mod(e + 1, 10) == 0:
            y_pred = model.predict(X_train, batch_size=batch_size, verbose=1)
            auc_train = roc_auc_score(y_train["out_cancer"], y_pred[0])
            y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
            auc_test = roc_auc_score(y_test[0], y_pred[0])
            with open(os.path.join(logs_dir, "auc.txt"), "a") as auc_file:
                auc_file.write("{},{}\n".format(auc_train, auc_test))

    model.save(os.path.join(checkpoints_dir, "weights.h5"))

    print("Training completed.")


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    device = "/gpu:" + sys.argv[1]
    with tf.device(device):
        train()
