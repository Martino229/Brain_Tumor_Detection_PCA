import numpy as np


class PCA_SVD:
    def __init__(self):
        self.mean_vector = None
        self.components = None  # Matrice V^T (Loadings/Autocervelli)
        self.singular_values = None  # Matrice Sigma
        self.explained_variance = None
        self.cumulative_variance = None

    def fit(self, X):
        # Sicurezza: conversione in float64 per evitare underflow
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # 1. Centratura dei dati (Mean Centering)
        self.mean_vector = np.mean(X, axis=0)
        X_centered = X - self.mean_vector

        # 2. Decomposizione SVD
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components = Vt
        self.singular_values = Sigma

        # 3. Calcolo degli autovalori e varianza
        eigenvalues = (Sigma ** 2) / (n_samples - 1)
        total_variance = np.sum(eigenvalues)

        self.explained_variance = (eigenvalues / total_variance) * 100
        self.cumulative_variance = np.cumsum(self.explained_variance)

    def transform(self, X, k_components):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_vector

        # Seleziona solo le prime 'k' direzioni principali
        Vk_transposed = self.components[:k_components, :].T

        # Proietta i dati nello spazio ridotto (Scores)
        scores = np.dot(X_centered, Vk_transposed)
        return scores

    # --- NUOVE FUNZIONI AGGIUNTE ---

    def inverse_transform(self, scores, k_components):

        #Ricostruisce l'immagine dallo spazio compresso a quello originale.
        # Seleziona le prime k componenti (Autocervelli)
        Vk = self.components[:k_components, :]

        # Moltiplica i punteggi per le componenti e riaggiunge la media
        X_reconstructed = np.dot(scores, Vk) + self.mean_vector
        return X_reconstructed

    def get_k_for_variance(self, target_variance_percent=90.0):
        #Trova in automatico il numero ottimale di k
        # Cerca il primo indice dove la varianza cumulativa supera la soglia
        k = np.argmax(self.cumulative_variance >= target_variance_percent) + 1
        return k




