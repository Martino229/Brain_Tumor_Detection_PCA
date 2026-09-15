import os
import numpy as np

from preprocessing import preprocessing
from PCA_algorithm import PCA_SVD
from KNN_algorithm import KNN
from confusion_matrix import results, report
from excel_export import excel_export
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


def main():
    # --- CONFIGURAZIONE IPERPARAMETRI ---
    img_size = (128, 128)
    n_splits = 5
    pca_variance_target = 90.0

    print("--- 1. CARICAMENTO TRAINING SET ---")
    X_train_val, y_train_val = preprocessing(base_dir="./train")

    print("--- 2. CARICAMENTO TEST SET (Blindato) ---")
    X_test_final, y_test_final = preprocessing(base_dir="./test")

    if len(X_train_val) == 0 or len(X_test_final) == 0:
        print("\n[ERRORE] Nessuna immagine caricata.")
        return

    print(f"\n--- 3. CROSS-VALIDATION A {n_splits} FOLD ---")
    np.random.seed(42)
    fold_indices = stratified_k_fold(y_train_val, k=n_splits)

    cv_metrics = {"Accuracy": [], "Recall": [], "Precision": [], "Specificity": [], "F1_score": [], "K_PCA": []}

    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        X_train, X_test = X_train_val[train_idx], X_train_val[test_idx]
        y_train, y_test = y_train_val[train_idx], y_train_val[test_idx]

        # PCA Dinamica
        pca = PCA_SVD()
        pca.fit(X_train)
        k_opt = pca.get_k_for_variance(pca_variance_target)
        X_train_pca = pca.transform(X_train, k_opt)
        X_test_pca = pca.transform(X_test, k_opt)

        # Classificazione KNN
        knn = KNN(n=3)
        knn.fit(X_train_pca, y_train)
        y_pred, _, _ = knn.predict(X_test_pca)

        metrics = results(y_test, y_pred)
        if metrics:
            for key in ["Accuracy", "Recall", "Precision", "Specificity", "F1_score"]:
                cv_metrics[key].append(metrics[key])
            cv_metrics["K_PCA"].append(k_opt)
            print(f"Fold {fold + 1} -> Acc: {metrics['Accuracy'] * 100:.1f}% | K Componenti usate: {k_opt}")

    print("\n--- 4. TEST FINALE SULLA CARTELLA TEST BLINDATA ---")
    pca_final = PCA_SVD()
    pca_final.fit(X_train_val)
    k_final = pca_final.get_k_for_variance(pca_variance_target)

    X_train_val_scores = pca_final.transform(X_train_val, k_final)
    X_test_final_scores = pca_final.transform(X_test_final, k_final)

    # Calcolo soglia anomalia sul Train
    distanze_train = np.sqrt(np.sum((X_train_val_scores - np.mean(X_train_val_scores, axis=0)) ** 2, axis=1))
    soglia_anomalia = np.max(distanze_train) * 1.5

    knn_final = KNN(unknown_threshold=soglia_anomalia)
    best_n = knn_final.optimize_n(X_train_val_scores, y_train_val)

    y_pred_final, distances_final, confidences_final = knn_final.predict(X_test_final_scores)

    print("\n--- RISULTATI DIAGNOSTICI FINALI ---")
    report(y_test_final, y_pred_final)
    final_metrics = results(y_test_final, y_pred_final)

    print("\n--- 5. GENERAZIONE CRUSCOTTO GRAFICO ---")
    # Plotting standard 2D
    plotter.plot_mean_image(pca_final.mean_vector, img_shape=img_size, save_name="grafico_mean_image.png")
    plotter.eigenimages(pca_final.components, img_shape=img_size, n_images=4, save_name="grafico_eigenimages.png")
    plotter.plot_variance(pca_final.cumulative_variance, save_name="grafico_varianza.png")
    plotter.plot_pca(X_train_val_scores, y_train_val, save_name="grafico_pca.png")
    plotter.plot_distance_distribution(distances_final, threshold=soglia_anomalia,
                                       save_name="grafico_distribuzione_distanze.png")
    plotter.plot_confusion_matrix(y_test_final, y_pred_final, save_name="grafico_matrice_confusione.png")

    print("\n--- 6. ESPORTAZIONE REPORT EXCEL ---")
    if final_metrics is not None:
        tot_train = len(X_train_val)
        tot_test = len(X_test_final)
        scartati = int(np.sum(y_pred_final == -1))
        validi = tot_test - scartati

        mat = final_metrics["Matrix"]
        tn_perc = mat[0, 0] / validi if validi > 0 else 0.0
        fp_perc = mat[0, 1] / validi if validi > 0 else 0.0
        fn_perc = mat[1, 0] / validi if validi > 0 else 0.0
        tp_perc = mat[1, 1] / validi if validi > 0 else 0.0

        export_data = [
            {"Analisi": "Totale Campioni Addestramento", "CV (Media)": tot_train, "CV (Dev. Std)": "-",
             "Test Blindato": "-"},
            {"Analisi": " - Training Sani (Classe 0)", "CV (Media)": int(np.sum(y_train_val == 0)),
             "CV (Dev. Std)": "-", "Test Blindato": "-"},
            {"Analisi": " - Training Malati (Classe 1)", "CV (Media)": int(np.sum(y_train_val == 1)),
             "CV (Dev. Std)": "-", "Test Blindato": "-"},
            {"Analisi": "Totale Campioni Test (Iniziali)", "CV (Media)": "-", "CV (Dev. Std)": "-",
             "Test Blindato": tot_test},
            {"Analisi": " - Test Sani (Classe 0)", "CV (Media)": "-", "CV (Dev. Std)": "-",
             "Test Blindato": int(np.sum(y_test_final == 0))},
            {"Analisi": " - Test Malati (Classe 1)", "CV (Media)": "-", "CV (Dev. Std)": "-",
             "Test Blindato": int(np.sum(y_test_final == 1))},
            {"Analisi": "Anomalie Scartate dal KNN (-1)", "CV (Media)": "-", "CV (Dev. Std)": "-",
             "Test Blindato": scartati},
            {"Analisi": "Campioni Test Validi Analizzati", "CV (Media)": "-", "CV (Dev. Std)": "-",
             "Test Blindato": validi},

            # --- DETTAGLI PCA ---
            {"Analisi": "Componenti PCA Utilizzate (K)", "CV (Media)": int(np.mean(cv_metrics['K_PCA'])),
             "CV (Dev. Std)": "-", "Test Blindato": k_final},
            {"Analisi": "Varianza Spiegata PCA", "CV (Media)": "-", "CV (Dev. Std)": "-",
             "Test Blindato": pca_final.cumulative_variance[k_final - 1] / 100.0},

            # --- METRICHE CLINICHE ---
            {"Analisi": "Accuratezza", "CV (Media)": np.mean(cv_metrics['Accuracy']),
             "CV (Dev. Std)": np.std(cv_metrics['Accuracy']), "Test Blindato": final_metrics['Accuracy']},
            {"Analisi": "Sensibilità (Recall)", "CV (Media)": np.mean(cv_metrics['Recall']),
             "CV (Dev. Std)": np.std(cv_metrics['Recall']), "Test Blindato": final_metrics['Recall']},
            {"Analisi": "Specificità", "CV (Media)": np.mean(cv_metrics['Specificity']),
             "CV (Dev. Std)": np.std(cv_metrics['Specificity']), "Test Blindato": final_metrics['Specificity']},
            {"Analisi": "Precisione", "CV (Media)": np.mean(cv_metrics['Precision']),
             "CV (Dev. Std)": np.std(cv_metrics['Precision']), "Test Blindato": final_metrics['Precision']},
            {"Analisi": "F1-Score", "CV (Media)": np.mean(cv_metrics['F1_score']),
             "CV (Dev. Std)": np.std(cv_metrics['F1_score']), "Test Blindato": final_metrics['F1_score']},

            # --- MATRICE DI CONFUSIONE IN PERCENTUALE ---
            {"Analisi": "Veri Positivi (TP)", "CV (Media)": "-", "CV (Dev. Std)": "-", "Test Blindato": tp_perc},
            {"Analisi": "Veri Negativi (TN)", "CV (Media)": "-", "CV (Dev. Std)": "-", "Test Blindato": tn_perc},
            {"Analisi": "Falsi Positivi (FP)", "CV (Media)": "-", "CV (Dev. Std)": "-", "Test Blindato": fp_perc},
            {"Analisi": "Falsi Negativi (FN) [GRAVE]", "CV (Media)": "-", "CV (Dev. Std)": "-",
             "Test Blindato": fn_perc}
        ]

        immagini_excel = ["grafico_varianza.png", "grafico_pca.png", "grafico_distribuzione_distanze.png",
                          "grafico_matrice_confusione.png"]

        excel_export(export_data, output_filename="Report_Tumori_Cerebrali.xlsx", image_paths=immagini_excel)


if __name__ == "__main__":
    main()