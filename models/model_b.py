import os
import sys


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generation import *
from utils.regression_utils import *


def create_model():
    model = Sequential(
        [
            Input((25, 25, 1)),
            Conv2D(48, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Flatten(),
            Dense(128, activation="relu", kernel_regularizer=l2(0.0023865)),
            Dropout(0.2),
            Dense(1, activation="linear"),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.0018456), loss="mse", metrics=["mae"])
    return model


X, y = generate_B(3000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


subset_results = train_with_subsets(X_train, y_train, X_test, y_test, create_model)


full_model = create_model()
full_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
analysis_results = analyze_predictions(full_model, X_test, y_test)
