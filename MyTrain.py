import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, keras:
    print(module.__name__, module.__version__)

train_dir = './input/training'
validation_dir = './input/validation'

# 读取图片
height = 128
width = 128
channels = 3
batch_size = 64
num_classes = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(height, width),
    batch_size=batch_size, seed=7,
    shuffle=True, class_mode='categorical'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
valid_generator = valid_datagen.flow_from_directory(
    validation_dir, target_size=(height, width),
    batch_size=batch_size, seed=7,
    shuffle=False, class_mode='categorical'
)

train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)

for i in range(2):
    x, y = train_generator.next()
    print(x.shape, y.shape)
    #print(y)

# 构建模型
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                        activation='relu',
                        input_shape=[width, height, channels]),
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# 训练
epochs = 300
history = model.fit_generator(
    train_generator, steps_per_epoch= train_num // batch_size,
    epochs=epochs, validation_data=valid_generator,
    validation_steps= valid_num // batch_size
)


def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8,5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


#print(history.history.keys())

plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 5)

print('完毕')
