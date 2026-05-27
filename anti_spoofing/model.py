import tensorflow as tf
from tensorflow import keras


def build_liveness_model(img_size: int = 224, dropout: float = 0.4) -> keras.Model:
    # EfficientNetB0 pre-trained on ImageNet, backbone frozen for phase 1
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="face_input")
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(dropout / 2)(x)

    # single sigmoid output: 0 = spoof, 1 = live
    outputs = keras.layers.Dense(1, activation="sigmoid", name="liveness")(x)

    return keras.Model(inputs, outputs, name="LivenessNet")


def unfreeze_backbone(model: keras.Model) -> keras.Model:
    # allow the full network to train in phase 2
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            layer.trainable = True
    return model
