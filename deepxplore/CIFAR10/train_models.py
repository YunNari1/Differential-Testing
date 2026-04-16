import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ResNet50용 resize
x_train = tf.image.resize(x_train, (224, 224)).numpy()
x_test = tf.image.resize(x_test, (224, 224)).numpy()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 모델 생성
def build_model(seed=None):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    base_model = ResNet50(
        weights=None,              # CIFAR용 → pretrained 사용 X
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model


model1 = build_model(seed=42)
model2 = build_model(seed=1234)


# 3. 서로 다르게 학습 
# 모델 1: Adam + 기본 학습
model1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 2: SGD + momentum + augmentation
model2.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# -------------------------
# 5. 데이터 증강 (model2만 다르게)
# -------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
# 4. 학습
print("Training Model 1 (Adam)...")
model1.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

print("Training Model 2 (SGD + Augmentation)...")
model2.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=15,
    validation_data=(x_test, y_test)
)

loss1, acc1 = model1.evaluate(x_test, y_test, verbose=0)
loss2, acc2 = model2.evaluate(x_test, y_test, verbose=0)

print(f"Model1 Accuracy: {acc1:.4f}")
print(f"Model2 Accuracy: {acc2:.4f}")

# 5. 저장 
model1.save("model1.h5")
model2.save("model2.h5")

print("Models saved!")