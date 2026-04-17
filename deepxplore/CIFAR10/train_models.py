import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# -------------------------
# 1. 데이터 로드
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------
# 2. tf.data로 변경 
# -------------------------
def preprocess(x, y):
    x = tf.image.resize(x, (224, 224)) 
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(10000).batch(8).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(preprocess).batch(8).prefetch(tf.data.AUTOTUNE)

# -------------------------
# 3. 모델 생성
# -------------------------
def build_model(seed=None):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)  
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    return models.Model(inputs=base_model.input, outputs=output)

model1 = build_model(seed=42)
model2 = build_model(seed=1234)

# -------------------------
# 4. 컴파일
# -------------------------
model1.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=SGD(0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# 5. 콜백
# -------------------------
checkpoint1 = ModelCheckpoint("model1_best.h5", monitor='val_accuracy', save_best_only=True)
checkpoint2 = ModelCheckpoint("model2_best.h5", monitor='val_accuracy', save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# -------------------------
# 6. 학습
# -------------------------
print("Training Model 1 (Adam)...")
model1.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds,
    callbacks=[checkpoint1, early_stop]
)

print("Training Model 2 (SGD)...")
model2.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds,
    callbacks=[checkpoint2, early_stop]
)

# -------------------------
# 7. 평가
# -------------------------
loss1, acc1 = model1.evaluate(test_ds)
loss2, acc2 = model2.evaluate(test_ds)

print(f"Model1 Accuracy: {acc1:.4f}")
print(f"Model2 Accuracy: {acc2:.4f}")

# ------------------------- 
# 8. 최종 저장 (DeepXplore용) 
# -------------------------
model1.save("model1.h5")
model2.save("model2.h5")

print("Models saved!")