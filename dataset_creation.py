import os
import cv2
import numpy as np

def build_dataset(dir):
    categories = ["glioma", "meningioma", "no_tumor", "pituitary"]
    X = []
    y = []

    for idx, category in enumerate(categories):
        folder_path = os.path.join(dir, category)

        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img_flattened = img.flatten()
                X.append(img_flattened)
                y.append(idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int8)

    X = X / 255.0

    return X, y


