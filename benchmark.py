import time
import numpy as np
from sklearn.decomposition import PCA as Sklearn_PCA
from sklearn.neighbors import KNeighborsClassifier as Sklearn_KNN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

from preprocessing import preprocessing
from PCA_algorithm import PCA_SVD
from KNN_algorithm import KNN


def evaluate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred) * 100
    sens = recall_score(y_true, y_pred) * 100
    spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
    prec = precision_score(y_true, y_pred) * 100
    return acc, sens, spec, prec


def main():
    print("\n--- CARICAMENTO DATI ---")
    X_train, y_train = preprocessing(base_dir="./train")
    X_test, y_test = preprocessing(base_dir="./test")

    if len(X_train) == 0 or len(X_test) == 0:
        print("Errore nel caricamento dei dati.")
        return

    print("\n[1/2] AVVIO PIPELINE CUSTOM (Mio Codice)...")
    start_custom = time.perf_counter()

    # PCA Custom
    pca_custom = PCA_SVD()
    pca_custom.fit(X_train)
    k_custom = pca_custom.get_k_for_variance(90.0)
    X_train_custom = pca_custom.transform(X_train, k_custom)
    X_test_custom = pca_custom.transform(X_test, k_custom)

    # KNN Custom (Tuning Dinamico)
    knn_custom = KNN()
    best_n_custom = knn_custom.optimize_n(X_train_custom, y_train, n_values=[1, 3, 5, 7, 9])
    y_pred_custom, _, _ = knn_custom.predict(X_test_custom)

    time_custom = time.perf_counter() - start_custom
    acc_c, sens_c, spec_c, prec_c = evaluate_metrics(y_test, y_pred_custom)

    print("\n[2/2] AVVIO PIPELINE LIBRERIA (Scikit-learn)...")
    start_sk = time.perf_counter()

    # PCA Sklearn
    pca_sk = Sklearn_PCA(n_components=0.90, svd_solver='full')
    X_train_sk = pca_sk.fit_transform(X_train)
    X_test_sk = pca_sk.transform(X_test)
    k_sk = pca_sk.n_components_

    # KNN Sklearn (Tuning Dinamico tramite GridSearchCV)
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
    grid_sk = GridSearchCV(Sklearn_KNN(metric='euclidean'), param_grid, cv=5)
    grid_sk.fit(X_train_sk, y_train)

    best_n_sk = grid_sk.best_params_['n_neighbors']
    knn_sk = grid_sk.best_estimator_
    y_pred_sk = knn_sk.predict(X_test_sk)

    time_sk = time.perf_counter() - start_sk
    acc_sk, sens_sk, spec_sk, prec_sk = evaluate_metrics(y_test, y_pred_sk)

    print("\n" + "=" * 50)
    print("                 RISULTATI BENCHMARK")
    print("=" * 50)
    print(f"{'Metrica':<20} | {'Mio Codice':<12} | {'Scikit-learn':<12}")
    print("-" * 50)
    print(f"{'Componenti PCA (K)':<20} | {k_custom:<12} | {k_sk:<12}")
    print(f"{'Miglior N (KNN)':<20} | {best_n_custom:<12} | {best_n_sk:<12}")
    print(f"{'Tempo Esecuzione':<20} | {time_custom:.4f}s     | {time_sk:.4f}s")
    print("-" * 50)
    print(f"{'Accuratezza':<20} | {acc_c:>6.2f}%     | {acc_sk:>6.2f}%")
    print(f"{'Sensibilità (Recall)':<20} | {sens_c:>6.2f}%     | {sens_sk:>6.2f}%")
    print(f"{'Specificità':<20} | {spec_c:>6.2f}%     | {spec_sk:>6.2f}%")
    print(f"{'Precisione':<20} | {prec_c:>6.2f}%     | {prec_sk:>6.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()