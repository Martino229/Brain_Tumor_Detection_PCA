import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def process_single_image(img_path, img_size=(128,128)):
    # Legge, ritaglia i bordi neri, ridimensiona e appiattisce la MRI.
    # Lettura in scala di grigi, restituisce un valore da 0.0 a 1.0
    img_array = imread(img_path, as_gray=True)

    # Taglio bordi neri dell'immagine in maniera tale da analizzare solo il cervello
    threshold = np.max(img_array) * 0.05
    mask = img_array > threshold

    # Se l'immagine non è completamente nera o corrotta, procedi col taglio
    if np.any(mask):
        # Trova le righe e le colonne che contengono l'organo
        righe_valide = np.any(mask, axis=1)
        colonne_valide = np.any(mask, axis=0)

        # Estrai il primo e l'ultimo indice utile
        r_min, r_max = np.where(righe_valide)[0][[0, -1]]
        c_min, c_max = np.where(colonne_valide)[0][[0, -1]]

        # Applica il ritaglio mantenendo solo il cervello
        img_array = img_array[r_min:r_max + 1, c_min:c_max + 1]

    # Ridimensionamento immagine chirurgico a matrice quadrata
    resized_img = resize(img_array, img_size, anti_aliasing=True)

    return resized_img.flatten().astype(np.float64)

def preprocessing(base_dir, categories=('Healthy', 'Disease'), img_size=(128,128)):
    data = []
    labels = []

    for class_num, category in enumerate(categories):
        category_path = os.path.join(base_dir, category)

        if not os.path.exists(category_path):
            print(f"[WARNING] Cartella non trovata: {category_path}")
            continue

        # CORREZIONE FATALE RISOLTA: Scansiona ESCLUSIVAMENTE la sottocartella in corso
        for root, dirs, files in os.walk(category_path):
            for img_name in files:
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(root, img_name)
                        vector = process_single_image(img_path, img_size)
                        data.append(vector)
                        labels.append(class_num)
                    except Exception as e:
                        print(f"[ERROR] Impossibile elaborare {img_name}: {e}")

    return np.array(data), np.array(labels)

