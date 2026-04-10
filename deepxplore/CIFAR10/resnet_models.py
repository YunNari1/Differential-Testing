import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


def build_resnet50_model():
    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_shape=(32, 32, 3)
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=x)
    return model