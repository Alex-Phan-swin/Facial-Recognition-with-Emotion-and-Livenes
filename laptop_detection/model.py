import tensorflow as tf
from tensorflow import keras


def build_laptop_model(img_size: int = 224, dropout: float = 0.3) -> keras.Model:
    # load MobileNetV2 with pretrained ImageNet weights, freeze backbone for phase 1
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="frame_input")

    # MobileNetV2 needs pixels in [-1, 1] range
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout / 2)(x)

    # single output: probability a laptop is in the frame
    outputs = keras.layers.Dense(1, activation="sigmoid", name="laptop")(x)

    return keras.Model(inputs, outputs, name="LaptopDetector")


def unfreeze_backbone(model: keras.Model) -> keras.Model:
    # unfreeze so the full model trains together in phase 2
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            layer.trainable = True
    return model
