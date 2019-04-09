from keras.initializers import Constant
from keras.layers import Input, Conv2D, Flatten, Activation, MaxPool2D, Dropout
from keras.models import Model

from focal_loss import focal_loss

img_width, img_height = 160, 160

loss_dict = {
    "out_cancer": focal_loss(),
    "out_compos": focal_loss(),
    "out_echo": focal_loss(),
    "out_shape": focal_loss(),
    "out_calcs": focal_loss(),
    "out_margin": focal_loss(),
}

loss_weights_dict = {
    "out_cancer": 1.0,
    "out_compos": 1.0,
    "out_echo": 1.0,
    "out_shape": 1.0,
    "out_calcs": 1.0,
    "out_margin": 1.0,
}


def multitask_cnn():
    # 160x160x1
    input_tensor = Input(shape=(img_height, img_width, 1), name="thyroid_input")
    # 160x160x8
    x = Conv2D(8, (3, 3), padding="same", activation="relu")(input_tensor)
    # 80x80x8
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 80x80x12
    x = Conv2D(12, (3, 3), padding="same", activation="relu")(x)
    # 40x40x12
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 40x40x16
    x = Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    # 20x20x16
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 20x20x24
    x = Conv2D(24, (3, 3), padding="same", activation="relu")(x)
    # 10x10x24
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 10x10x32
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    # 5x5x32
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 5x5x48
    x = Conv2D(48, (3, 3), padding="same", activation="relu")(x)
    # 5x5x48
    x = Dropout(0.5)(x)

    y_cancer = Conv2D(
        filters=1,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)
    y_cancer = Flatten()(y_cancer)
    y_cancer = Activation("sigmoid", name="out_cancer")(y_cancer)

    y_compos = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)
    y_compos = Flatten()(y_compos)
    y_compos = Activation("softmax", name="out_compos")(y_compos)

    y_echo = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)
    y_echo = Flatten()(y_echo)
    y_echo = Activation("softmax", name="out_echo")(y_echo)

    y_shape = Conv2D(
        filters=1,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)
    y_shape = Flatten()(y_shape)
    y_shape = Activation("sigmoid", name="out_shape")(y_shape)

    y_calcs = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)
    y_calcs = Flatten()(y_calcs)
    y_calcs = Activation("softmax", name="out_calcs")(y_calcs)

    y_margin = Conv2D(
        filters=4,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)
    y_margin = Flatten()(y_margin)
    y_margin = Activation("softmax", name="out_margin")(y_margin)

    return Model(
        inputs=[input_tensor],
        outputs=[y_cancer, y_compos, y_echo, y_shape, y_calcs, y_margin],
    )
