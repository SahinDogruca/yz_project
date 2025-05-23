from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def train_with_subsets(X_train, y_train, X_test, y_test, create_model):
    results = {}

    for subset_size in [0.25, 0.5, 1.0]:
        if subset_size < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=subset_size, random_state=42
            )
        else:
            X_sub, y_sub = X_train, y_train

        model = create_model()
        history = model.fit(
            X_sub,
            y_sub,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        # En iyi ve son metrikleri kaydet
        best_val_mae = min(history.history["val_mae"])
        final_test_mae = model.evaluate(X_test, y_test, verbose=0)[1]

        results[f"{subset_size*100}%"] = {
            "best_val_mae": best_val_mae,
            "final_test_mae": final_test_mae,
        }

    print("\nEğitim Kümesi Boyutu | Best Val MAE | Test MAE")
    for size, metrics in results.items():
        print(
            f"{size:15} | {metrics['best_val_mae']:.4f}     | {metrics['final_test_mae']:.4f}"
        )

    return results


def analyze_predictions(model, X_test, y_test, threshold=1.0, num_samples=5):
    y_pred = model.predict(X_test).flatten()
    absolute_errors = np.abs(y_pred - y_test)

    # Threshold bazlı doğru/yanlış hesaplama
    correct_mask = absolute_errors <= threshold
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    # İstatistikler
    print(f"\nTest MAE: {np.mean(absolute_errors):.4f}")
    print(f"Threshold ({threshold} birim) Doğruluk: {correct_mask.mean():.2%}")
    print(f"Maksimum Hata: {absolute_errors.max():.4f}")
    print(f"Std Hata: {absolute_errors.std():.4f}")

    # Hata dağılımı görselleştirme
    plt.figure(figsize=(15, 6))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, c=absolute_errors, cmap="viridis")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.colorbar(label="Mutlak Hata")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahminler")
    plt.title("Gerçek vs Tahmin")

    # Hata histogramı
    plt.subplot(1, 2, 2)
    plt.hist(absolute_errors, bins=20, edgecolor="black")
    plt.axvline(threshold, color="r", linestyle="--", label=f"Threshold ({threshold})")
    plt.xlabel("Mutlak Hata")
    plt.ylabel("Frekans")
    plt.title("Hata Dağılımı")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Örnek görüntüler
    plt.figure(figsize=(15, 6))

    # En iyi tahminler
    best_indices = np.argsort(absolute_errors)[:num_samples]
    for i, idx in enumerate(best_indices):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        plt.title(
            f"Gerçek: {y_test[idx]:.1f}\nTahmin: {y_pred[idx]:.1f}\nHata: {absolute_errors[idx]:.1f}"
        )
        plt.axis("off")

    # En kötü tahminler
    worst_indices = np.argsort(-absolute_errors)[:num_samples]
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        plt.title(
            f"Gerçek: {y_test[idx]:.1f}\nTahmin: {y_pred[idx]:.1f}\nHata: {absolute_errors[idx]:.1f}"
        )
        plt.axis("off")

    plt.suptitle(f"En İyi ve En Kötü Tahmin Örnekleri (Threshold: {threshold})")
    plt.tight_layout()
    plt.show()

    return {
        "y_pred": y_pred,
        "absolute_errors": absolute_errors,
        "correct_indices": correct_indices,
        "incorrect_indices": incorrect_indices,
    }
