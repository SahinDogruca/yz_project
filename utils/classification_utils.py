from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def train_with_subsets(X_train, y_train, X_test, y_test, create_model):
    results = {}

    for subset_size in [0.25, 0.5, 1.0]:
        # Veri alt kümesini oluştur
        if subset_size < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train,
                y_train,
                train_size=subset_size,
                stratify=y_train,
                random_state=42,
            )
        else:
            X_sub, y_sub = X_train, y_train

        # Modeli oluştur ve eğit
        model = create_model()
        history = model.fit(
            X_sub,
            y_sub,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        # En yüksek val_acc'yi bul
        best_val_acc = max(history.history["val_accuracy"])
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        results[f"{subset_size*100}%"] = {
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
        }

    # Sonuçları raporla
    print("Eğitim Kümesi Boyutu | En İyi Val Acc | Test Acc")
    for size, metrics in results.items():
        print(
            f"{size:15} | {metrics['best_val_acc']:.4f}       | {metrics['test_acc']:.4f}"
        )

    return results


def analyze_predictions(model, X_test, y_test, num_samples=5):
    # Tahminleri yap
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    # Doğru ve yanlış tahminleri ayır
    correct_mask = y_pred == y_test
    incorrect_mask = ~correct_mask

    # İstatistikleri hesapla
    print(f"\nToplam Test Örneği: {len(y_test)}")
    print(f"Doğru Tahmin Oranı: {np.mean(correct_mask):.2%}")
    print(f"Yanlış Tahmin Oranı: {np.mean(incorrect_mask):.2%}")

    # Yanlış tahminlerin detaylı analizi
    incorrect_indices = np.where(incorrect_mask)[0]
    confusion = {}
    for true, pred in zip(y_test[incorrect_mask], y_pred[incorrect_mask]):
        confusion[(true, pred)] = confusion.get((true, pred), 0) + 1

    # En sık karıştırılan sınıflar
    print("\nEn Sık Karıştırılan Sınıflar (Gerçek → Tahmin):")
    for (true, pred), count in sorted(confusion.items(), key=lambda x: -x[1])[:5]:
        print(f"{true} → {pred}: {count} kez")

    # Görsel örnekler için yeni düzen
    plt.figure(figsize=(15, 6))

    # Doğru örnekler için subplot
    plt.subplot(2, 1, 1)  # 2 satır 1 sütun, üst kısım
    plt.title("Doğru Tahmin Örnekleri")
    for i in range(num_samples):
        idx = np.where(correct_mask)[0][i]
        plt.subplot(2, num_samples, i + 1)  # 2 satır, num_samples sütun
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        plt.title(f"Gerçek: {y_test[idx]}\nTahmin: {y_pred[idx]}")
        plt.axis("off")

    # Yanlış örnekler için subplot
    plt.subplot(2, 1, 2)  # 2 satır 1 sütun, alt kısım
    plt.title("Yanlış Tahmin Örnekleri")
    for i in range(num_samples):
        idx = incorrect_indices[i]
        plt.subplot(2, num_samples, num_samples + i + 1)  # 2. satır
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        plt.title(f"Gerçek: {y_test[idx]}\nTahmin: {y_pred[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "correct_indices": np.where(correct_mask)[0],
        "incorrect_indices": incorrect_indices,
    }
