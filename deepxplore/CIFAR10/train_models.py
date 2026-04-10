from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from resnet_models import build_resnet50_model

# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 모델 생성
model1 = build_resnet50_model()
model2 = build_resnet50_model()

# 3. 서로 다르게 학습 (중요!)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 학습
print("Training model 1...")
model1.fit(x_train, y_train, epochs=3, batch_size=64)

print("Training model 2...")
model2.fit(x_train, y_train, epochs=5, batch_size=64)

# 5. 저장 (🔥 필수)
model1.save("model1.h5")
model2.save("model2.h5")

print("Models saved!")