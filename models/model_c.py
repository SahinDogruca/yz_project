import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generation import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import matplotlib.pyplot as plt



def create_model():
  model = Sequential([
    Conv2D(48, (3,3), activation='relu', input_shape=(25,25,1), padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0023865)),
    Dropout(0.2),
    Dense(1, activation='linear')
  ])

  model.compile(optimizer=Adam(learning_rate=0.0018456), loss='mse', metrics=['mae'])
  return model


X, y = generate_C()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model()
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1
)

# Modeli değerlendir
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}, Test MAE: {mae:.4f}")


# def build_model(hp):
#     """Hiperparametre optimizasyonu için model builder fonksiyonu"""
#     model = Sequential()

#     # 1. Konvolüsyonel Katman (hp parametresi artık doğru kullanılıyor)
#     model.add(Conv2D(
#         filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),
#         kernel_size=(3,3),
#         activation=hp.Choice('activation', ['relu', 'leaky_relu']),
#         padding='same',
#         input_shape=(25,25,1)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2,2)))

#     # 2. Konvolüsyonel Katman
#     model.add(Conv2D(
#         filters=hp.Int('conv2_filters', min_value=32, max_value=128, step=32),
#         kernel_size=(3,3),
#         activation=hp.Choice('activation', ['relu', 'leaky_relu']),
#         padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2,2)))

#     # 3. Konvolüsyonel Katman (opsiyonel)
#     if hp.Boolean('add_third_conv'):
#         model.add(Conv2D(
#             filters=hp.Int('conv3_filters', min_value=64, max_value=256, step=64),
#             kernel_size=(3,3),
#             activation=hp.Choice('activation', ['relu', 'leaky_relu']),
#             padding='same'))
#         model.add(BatchNormalization())

#     # Tam Bağlı Katmanlar
#     model.add(Flatten())
#     model.add(Dense(
#         units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
#         activation='relu',
#         kernel_regularizer=l2(hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log'))))
#     model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
#     model.add(Dense(1, activation='linear'))

#     # Model Derleme
#     model.compile(
#         optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
#         loss='mse',
#         metrics=['mae'])

#     return model

# # Tuner'ı başlatma
# tuner = kt.RandomSearch(
#     build_model,
#     objective='val_mae',
#     max_trials=30,
#     executions_per_trial=2,
#     directory='tuning_dir',
#     project_name='max_euclidean_distance_tuning')

# # Tuner'ı çalıştırma (X_train, y_train, X_val, y_val tanımlı olmalı)
# tuner.search(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=50,
#     batch_size=32,
#     verbose=1)