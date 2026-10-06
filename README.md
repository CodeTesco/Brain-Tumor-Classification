# Brain Tumor MRI Classification: Classical Machine Learning vs. Artificial Neural Networks

A comparative study evaluating classical machine learning models (Logistic Regression, Random Forest, XGBoost) and a fully connected Artificial Neural Network (ANN) for multi-class brain tumor classification on MRI scans.

---

## 📌 Project Overview

This project classifies magnetic resonance imaging (MRI) scans into four distinct classes:
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor**

The primary objective was to establish baseline performance metrics using classical ML algorithms on flattened spatial features, evaluate a fully connected ANN, and identify the strengths, failure modes, and spatial limitations of vector-based computer vision pipelines in medical diagnostics.

---

## 📊 Model Performance & Comparative Analysis

All models were evaluated on an unseen, balanced test dataset of 1,600 images (400 images per class) preprocessed to $224 \times 224$ grayscale pixels.

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Macro ROC-AUC | Glioma Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **85.81%** | **86.92%** | **85.81%** | **0.8538** | 0.9620 | 64.00% |
| **XGBoost** | 85.56% | 85.20% | 85.56% | 0.8514 | **0.9639** | 69.00% |
| **ANN (Dense Network)** | 84.50% | 84.25% | 84.50% | 0.8429 | 0.9521 | **70.25%** |
| **Logistic Regression** | 78.63% | 78.75% | 78.63% | 0.7838 | 0.9192 | 61.75% |

---

## 🔑 Key Engineering & Scientific Learnings

### 1. Data Preprocessing & Pipeline Integrity
* **Aspect Ratio Preservation:** Resizing non-square MRI scans directly into fixed square dimensions ($224 \times 224$) warps spatial tumor geometry. Implementing letterbox padding (padding borders with zero-value pixels prior to resizing) maintains the structural integrity of brain tissues and tumor boundaries.
* **Normalization & Gradient Flow:** Unscaled pixel values ($0$–$255$) fed into dense neural layers cause immediate gradient instability. Verifying data types (`uint8` vs `float32`) prevents critical failures such as **double normalization** (which squishes pixel ranges to $0.0$–$0.0039$, causing vanishing gradients) or zero-gradient saturation.
* **Label Order Consistency:** High validation performance (~83.5%) paired with near-random test accuracy (~25.0%) often points to a **label permutation error** rather than model overfitting. Ensuring index-to-class mappings (`0: glioma`, `1: meningioma`, `2: no_tumor`, `3: pituitary`) remain identical across `tf.data.Dataset` generators and evaluation scripts is essential.

### 2. The 1D Flattening Spatial Ceiling
* **Curse of Dimensionality:** Flattening a $224 \times 224$ image yields $50,176$ continuous features per sample. Traditional models (especially tree-based ensembles) encounter severe memory bottlenecks during tree-split calculations across large feature spaces.
* **Spatial Blindness:** Vectorizing 2D image matrices destroys pixel locality—the physical relationship between adjacent pixels is lost. As a result, classical models perform well on well-defined structural boundaries (`pituitary` and `no_tumor` $>90\%$ F1-score), but struggle with diffuse, irregular tumor margins (`glioma` recall drops to $64.0\%$).
* **Performance Clustering:** Non-linear models using 1D flattened vectors hit a natural performance ceiling at **84.5%–85.8% accuracy**, demonstrating the upper boundary of non-convolutional approaches on raw pixel arrays.

### 3. Model Dynamics & Trade-offs
* **Tree Ensembles vs. Neural Networks:** Random Forest achieved the highest overall accuracy (**85.81%**), but suffered from significant false negatives on aggressive tumors (missing $36\%$ of gliomas).
* **Glioma Sensitivity in ANNs:** The fully connected ANN achieved the highest **Glioma Recall (70.25%)** among all models. Non-linear activation combinations (`ReLU` + `BatchNormalization`) allowed the neural network to learn complex intensity thresholds better than orthogonal decision tree splits.
