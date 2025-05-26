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
from utils.regression_utils import *
from data.generation import *
from tuning.tuner import *


def model_a():
    # Veri oluşturma ve bölme
    X, y = generate_A(3000)
    X = X[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Tuner dizin ayarları
    project_dir = "keras_tuner"
    project_name = "regression_model_a"

    # Daha önceki tuner varsa yükle, yoksa oluştur
    tuner = get_or_run_tuner(
        RegressionTuner,
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        project_dir=project_dir,
        project_name=project_name,
        epochs=100,
    )

    # En iyi hiperparametreleri al
    best_hps = tuner.get_best_hyperparameters()[0]

    # Tespit edilen en iyi parametrelerle model oluştur
    def tuned_model_a():
        model = Sequential(
            [
                Input((25, 25, 1)),
                Conv2D(
                    best_hps.get("conv1_units"),
                    (3, 3),
                    activation="relu",
                    padding="same",
                ),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(
                    best_hps.get("conv2_units"),
                    (3, 3),
                    activation="relu",
                    padding="same",
                ),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(
                    best_hps.get("conv3_units"),
                    (3, 3),
                    activation="relu",
                    padding="same",
                ),
                BatchNormalization(),
                Flatten(),
                Dense(
                    best_hps.get("dense_units"),
                    activation="relu",
                    kernel_regularizer=l2(best_hps.get("l2")),
                ),
                Dropout(best_hps.get("dropout")),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=best_hps.get("learning_rate")),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # Subset analizleri
    subset_results = train_with_subsets(X_train, y_train, X_test, y_test, tuned_model_a)

    # Tam eğitim
    full_model = tuned_model_a()
    full_model.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=best_hps.get("batch_size"),
        validation_data=(X_test, y_test),
        verbose=0,
    )

    return analyze_predictions(full_model, X_test, y_test)


model_a()
