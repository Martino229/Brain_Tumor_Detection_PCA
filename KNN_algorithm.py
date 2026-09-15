import numpy as np

class KNN:
    def __init__(self, n=3, unknown_threshold=None):
        self.n =n
        self.unknown_threshold = unknown_threshold
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def hybrid_confidence(self,n_nearest_lables, majority_vote, n_nearest_distances):
        consensus= np.sum(n_nearest_lables == majority_vote) / self.n
        avg_distance = np.mean(n_nearest_distances)
        distance_confidence= 1.0/(1.0+avg_distance)
        return (0.5*distance_confidence)+(0.5*consensus)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        min_distances = []
        confidences = []

        for x in X_test:
            # Calcolo distanze vettorizzate
            distances=np.sqrt(np.sum((self.X_train-x)**2, axis=1))
            min_dist=np.min(distances)
            min_distances.append(min_dist)

            # Controllo Anomalia
            if self.unknown_threshold is not None and min_dist > self.unknown_threshold:
                predictions.append(-1)
                confidences.append(0.0)
                continue

            # Estrazione vicini
            n_indices=np.argsort(distances)[:self.n]
            n_nearest_lables=self.y_train[n_indices]
            n_nearest_distances= distances[n_indices]

            # Maggioranza
            majority_vote = np.bincount(n_nearest_lables.astype(int)).argmax()
            predictions.append(majority_vote)

            # Confidenza
            conf=self.hybrid_confidence(n_nearest_lables, majority_vote, n_nearest_distances)
            confidences.append(conf)

        return np.array(predictions), np.array(min_distances), np.array(confidences)

    def evaluate_accuracy(self, X_test, y_test):
        predictions, _, _= self.predict(X_test)
        valid_indices= predictions !=-1
        if not np.any(valid_indices):
            return 0.0
        return np.mean(predictions[valid_indices] == y_test[valid_indices])

    def optimize_n(self,X_train, y_train, n_values=[1, 3, 5, 7, 9]):
        best_n=self.n
        best_acc =-1

        np.random.seed(42)
        indices= np.random.permutation(len(X_train))
        split_idx = int(len(X_train)*0.8)
        X_study, y_study = X_train[indices[:split_idx]], y_train[indices[:split_idx]]
        X_sim, y_sim = X_train[indices[split_idx:]], y_train[indices[split_idx:]]

        for test_n in n_values:
            self.n= test_n
            self.fit(X_study, y_study)
            acc = self.evaluate_accuracy(X_sim, y_sim)

            if acc>best_acc:
                best_acc=acc
                best_n=test_n
        self.n=best_n
        self.fit(X_train, y_train)
        print(f"\n[KNN Tuning] Miglior numero di vicini (n) trovato: {self.n} (Accuracy stimata: {best_acc * 100:.1f}%)")
        return self.n

