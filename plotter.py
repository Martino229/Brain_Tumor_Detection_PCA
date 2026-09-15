import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Importiamo la tua funzione personalizzata
from confusion_matrix import confusion_matrix


def plot_pca(scores, labels, save_name="grafico_pca.png"):
    plt.figure(figsize=(8.0, 6.0))
    idx_healthy = np.where(labels == 0)[0]
    idx_disease = np.where(labels == 1)[0]

    plt.scatter(scores[idx_healthy, 0], scores[idx_healthy, 1], c='blue', label='Healthy', alpha=0.7)
    plt.scatter(scores[idx_disease, 0], scores[idx_disease, 1], c='red', label='Disease', alpha=0.7)

    plt.title('Proiezione dei Pazienti nello Spazio PCA (PC1 vs PC2)')
    plt.xlabel('Componente Principale 1 (PC1)')
    plt.ylabel('Componente Principale 2 (PC2)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] Grafico PCA salvato in: {os.path.abspath(save_name)}")
    plt.show()


def eigenimages(loadings, img_shape, n_images=5, save_name="grafico_eigenimages.png"):
    fig, axes = plt.subplots(1, n_images, figsize=(15.0, 4.0))
    for i in range(n_images):
        if i >= loadings.shape[0]:
            break
        eigen_img = loadings[i, :].reshape(img_shape)
        axes[i].imshow(eigen_img, cmap='bone')
        axes[i].set_title(f'Eigenimage {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] Grafico Eigenimages salvato in: {os.path.abspath(save_name)}")
    plt.show()


def plot_variance(pve_cumulative, save_name="grafico_varianza.png"):
    plt.figure(figsize=(8.0, 5.0))
    components = np.arange(1, len(pve_cumulative) + 1)

    plt.plot(components, pve_cumulative, marker='o', linestyle='-', color='b', linewidth=2)
    plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Soglia 90%')
    plt.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='Soglia 95%')

    plt.title('Curva della Varianza Cumulativa Spiegata')
    plt.xlabel('Numero di Componenti Principali (k)')
    plt.ylabel('Varianza Cumulativa (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] Grafico Varianza salvato in: {os.path.abspath(save_name)}")
    plt.show()


def plot_mean_image(mean_vector, img_shape, save_name="grafico_mean_image.png"):
    plt.figure(figsize=(5.0, 5.0))
    plt.imshow(mean_vector.reshape(img_shape), cmap='bone')
    plt.title('Risonanza Magnetica Media (Baseline Anatomica)')
    plt.axis('off')

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] Grafico Mean Image salvato in: {os.path.abspath(save_name)}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_name="grafico_matrice_confusione.png"):
    cm = confusion_matrix(y_true, y_pred)

    if cm is None:
        print("[WARNING] Nessun campione valido per generare la matrice.")
        return

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sano (0)", "Malato (1)"])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    plt.title('Matrice di Confusione Diagnostica')

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] Matrice di Confusione salvata in: {os.path.abspath(save_name)}")
    plt.show()


def plot_distance_distribution(distances, threshold, save_name="grafico_distribuzione_distanze.png"):
    plt.figure(figsize=(8.0, 5.0))
    plt.hist(distances, bins=30, alpha=0.7, color='teal', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Soglia Anomalia ({threshold})')

    plt.title('Distribuzione delle Distanze Minime nello Spazio PCA')
    plt.xlabel('Distanza Euclidea dal Vicino Più Prossimo')
    plt.ylabel('Frequenza (Numero di Pazienti)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] Grafico Distanze salvato in: {os.path.abspath(save_name)}")
    plt.show()