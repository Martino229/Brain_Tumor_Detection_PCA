import os
import time
import numpy as np

from preprocessing import preprocessing
from KNN_algorithm import KNN
from confusion_matrix import results, report
import plotter


def stratified_k_fold(y, k=5):
    y = np.array(y)
    classes = np.unique(y)
    folds = [[] for _ in range(k)]

    for c in classes:
        c_indices = np.where(y == c)[0]
        np.random.shuffle(c_indices)
        c_splits = np.array_split(c_indices, k)
        for i in range(k):
            folds[i].extend(c_splits[i])

    fold_indices = []
    for i in range(k):
        test_idx = np.array(folds[i])
        train_idx = np.array([idx for j in range(k) if j != i for idx in folds[j]])
        fold_indices.append((train_idx, test_idx))

    return fold_indices


def alternative_main():
    print("=== AVVIO TEST ALTERNATIVO: PIPELINE COMPLETA SENZA PCA ===")

    print("--- 1. CARICAMENTO TRAINING SET ---")
    X_train_val_raw, y_train_val = preprocessing(base_dir="./train")

    print("--- 2. CARICAMENTO TEST SET (Blindato) ---")
    X_test_final_raw, y_test_final = preprocessing(base_dir="./test")

    if len(X_train_val_raw) == 0 or len(X_test_final_raw) == 0:
        print("\n[ERRORE] Nessuna immagine caricata.")
        return

    print("\n[Preprocessing Matematico] Sottrazione della media globale...")
    mean_vector = np.mean(X_train_val_raw, axis=0)
    X_train_val = X_train_val_raw - mean_vector
    X_test_final = X_test_final_raw - mean_vector

    print(f"\n--- 3. CROSS-VALIDATION A 5 FOLD (SENZA PCA, 16.384 dimensioni) ---")
    np.random.seed(42)
    fold_indices = stratified_k_fold(y_train_val, k=5)

    cv_metrics = {"Accuracy": [], "Recall": [], "Precision": [], "Specificity": [], "F1_score": []}

    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        X_train, X_test = X_train_val[train_idx], X_train_val[test_idx]
        y_train, y_test = y_train_val[train_idx], y_train_val[test_idx]

        knn = KNN(n=3)
        knn.fit(X_train, y_train)
        y_pred, _, _ = knn.predict(X_test)

        metrics = results(y_test, y_pred)
        if metrics:
            for key in ["Accuracy", "Recall", "Precision", "Specificity", "F1_score"]:
                cv_metrics[key].append(metrics[key])
            print(f"Fold {fold + 1} -> Acc: {metrics['Accuracy'] * 100:.1f}%")

    print("\n--- 4. TEST FINALE SULLA CARTELLA TEST BLINDATA ---")

    distanze_train = np.sqrt(np.sum((X_train_val - np.mean(X_train_val, axis=0)) ** 2, axis=1))
    soglia_anomalia = np.max(distanze_train) * 1.5

    knn_final = KNN(unknown_threshold=soglia_anomalia)

    best_n = knn_final.optimize_n(X_train_val, y_train_val, n_values=[1, 3, 5])
    knn_final.n = best_n

    print("Avvio classificazione (Alta Dimensionalità). Misurazione tempo in corso...")
    start_time = time.perf_counter()

    knn_final.fit(X_train_val, y_train_val)
    y_pred_final, distances_final, confidences_final = knn_final.predict(X_test_final)

    end_time = time.perf_counter()
    tempo_esecuzione_no_pca = end_time - start_time

    print(f"\n[BENCHMARK] Tempo di esecuzione K-NN (Senza PCA): {tempo_esecuzione_no_pca:.4f} secondi")

    print("\n--- RISULTATI DIAGNOSTICI FINALI (SENZA PCA) ---")
    report(y_test_final, y_pred_final)

    print("\n--- 5. GENERAZIONE GRAFICO ---")
    plotter.plot_confusion_matrix(y_test_final, y_pred_final, save_name="grafico_matrice_confusione_NO_PCA.png")

    print("\n=== TEST ALTERNATIVO COMPLETATO ===")


if __name__ == "__main__":
    alternative_main()