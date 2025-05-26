import os
import keras_tuner as kt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


def build_regression_model(hp):
    # Konvolüsyon hiperparametreleri
    model = Sequential(
        [
            Input((25, 25, 1)),
            Conv2D(
                hp.Int("conv1_units", 16, 64, step=16),
                (3, 3),
                activation="relu",
                padding="same",
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(
                hp.Int("conv2_units", 64, 192, step=64),
                (3, 3),
                activation="relu",
                padding="same",
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(
                hp.Int("conv3_units", 64, 256, step=64),
                (3, 3),
                activation="relu",
                padding="same",
            ),
            BatchNormalization(),
            Flatten(),
            Dense(
                hp.Int("dense_units", 128, 512, step=128),
                activation="relu",
                kernel_regularizer=l2(hp.Float("l2", 1e-5, 1e-2, sampling="log")),
            ),
            Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)),
            Dense(1, activation="linear"),
        ]
    )

    # Optimizasyon hiperparametreleri
    optimizer = Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"))
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def build_classification_model(hp):
    model = Sequential(
        [
            Input((25, 25, 1)),
            Conv2D(
                hp.Int("conv1_units", 16, 64, step=16),
                (3, 3),
                activation="relu",
                padding="same",
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(
                hp.Int("conv2_units", 64, 192, step=32),
                (3, 3),
                activation="relu",
                padding="same",
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            GlobalAveragePooling2D(),
            Dense(
                hp.Int("dense_units", 128, 512, step=128),
                activation="relu",
                kernel_regularizer=l2(hp.Float("l2", 1e-5, 1e-2, sampling="log")),
            ),
            Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)),
            Dense(10, activation="softmax"),
        ]
    )

    optimizer = Nadam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"))
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class BaseTuner(kt.RandomSearch):
    def __init__(self, *args, **kwargs):
        self.objective_name = kwargs.get("objective", "val_loss")
        super().__init__(*args, **kwargs)

    def run_trial(self, trial, X, y, validation_data, **kwargs):
        hp = trial.hyperparameters
        early_stop = EarlyStopping(
            monitor=self.objective_name, patience=10, restore_best_weights=True
        )
        return super().run_trial(
            trial,
            X,
            y,
            validation_data=validation_data,
            batch_size=hp.Int("batch_size", 16, 256, step=16),
            callbacks=[early_stop],
            **kwargs,
        )


class RegressionTuner(BaseTuner):
    def __init__(self, **kwargs):
        super().__init__(
            build_regression_model,
            objective="val_loss",
            max_trials=30,
            executions_per_trial=2,
            directory="keras_tuner",
            project_name="regression_models",
            **kwargs,
        )


class ClassificationTuner(BaseTuner):
    def __init__(self, **kwargs):
        super().__init__(
            build_classification_model,
            objective="val_accuracy",
            max_trials=30,
            executions_per_trial=2,
            directory="keras_tuner",
            project_name="classification_models",
            **kwargs,
        )


def ensemble_top_models(tuner, X, y, n_models=5):
    best_models = tuner.get_best_models(num_models=n_models)
    predictions = [model.predict(X) for model in best_models]
    return np.mean(predictions, axis=0)


# Yeni fonksiyon: Var olan tuner varsa yükle, yoksa eğit
def get_or_run_tuner(
    tuner_class, X, y, validation_data, project_dir, project_name, epochs=100
):
    tuner_path = os.path.join(project_dir, project_name)
    if (
        os.path.exists(tuner_path)
        and os.path.isdir(tuner_path)
        and len(os.listdir(tuner_path)) > 0
    ):
        print(f"{project_name} tuner already exists. Loading existing tuner.")
        tuner = tuner_class()
        tuner.reload()  # Mevcut denemeleri yükler
    else:
        print(f"{project_name} tuner not found. Running search...")
        tuner = tuner_class()
        tuner.search(X, y, validation_data=validation_data, epochs=epochs)
    return tuner
