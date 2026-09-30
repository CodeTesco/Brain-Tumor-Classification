import cv2
import os
from pathlib import Path

def pad_and_resize(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h*scale), int(w*scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img

def process_dataset(input_dir, output_dir, target_size=(224, 224)):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)

                processed_img = pad_and_resize(input_path, target_size)
                if processed_img is not None:
                    save_path = os.path.join(save_dir, file)
                    cv2.imwrite(save_path, processed_img)


dataset_path = Path.cwd() / "Projects" / "Brain Tumor Classification" / "dataset"
process_dataset(Path(dataset_path) / "Train", Path(dataset_path) / "Train_preprocessed")
process_dataset(Path(dataset_path) / "Test", Path(dataset_path) / "Test_preprocessed")