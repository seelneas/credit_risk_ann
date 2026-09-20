# ANN Project — Credit Risk Classification

Build an Artificial Neural Network for credit-risk classification using the German Credit dataset. It covers multiclass classification, feature engineering, class imbalance, Batch Normalization, Dropout, early stopping, and hyperparameter experiments.

---

## 📂 Dataset Information & Instructions

### 1. Where to Download the Dataset
The project uses the benchmark **Statlog (German Credit Data)** dataset. You can obtain the official raw files from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

### 2. Which Part of the Dataset to Use
* **File to use:** Download the dataset package and locate the core data file named **`german.data`**.
* **Directory Structure:** Place this file inside a folder named `statlog+german+credit+data/` in your project root directory so that it matches the notebook's expected path:
  ```text
  statlog+german+credit+data/
  └── german.data
  ```
* **Dataset Characteristics:** 
  * Contains **1,000 instances** and **21 attributes** (8 numerical and 13 categorical attributes).
  * The target column (labeled as `risk` in the notebook) originally uses codes where `1` represents **Good** credit risk (700 customers) and `2` represents **Bad** credit risk (300 customers). In the preprocessing pipeline, these are mapped to binary labels (`0` for Good, `1` for Bad).

---

## 📖 Documentation & Code Structure Guide

To fully understand the implementation details, follow the sequential section breakdown within the Jupyter Notebook/script:

1. **Task Setup and Reproducibility:** Imports necessary data science libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) and fixes random seeds (`SEED = 42`) for TensorFlow and NumPy to ensure reproducible training runs.
2. **Data Loading & Inspection:** Loads the space-separated `german.data` file, assigns descriptive column names as per official UCI documentation, and separates features into numerical and categorical types.
3. **Data Cleaning:** Validates missing values and duplicate entries to guarantee data integrity.
4. **Target Mapping:** Recodes the target column (`risk`) into binary format (`0` and `1`).
5. **Feature Engineering:** Introduces domain-specific features such as `credit_per_month` (ratio of total credit amount to duration).
6. **Categorical Encoding:** Transforms string-based categorical variables into a numeric feature matrix using one-hot encoding (`pd.get_dummies` with `drop_first=True`).
7. **Data Splitting:** Partitions the dataset into **Train (70%)**, **Validation (15%)**, and **Test (15%)** subsets using stratified sampling to preserve the target class distribution.
8. **Feature Scaling:** Standardizes features using `StandardScaler` fitted exclusively on the training split to prevent data leakage.
9. **Class Imbalance Mitigation:** Computes balanced class weights to address the unequal distribution between good and bad credit profiles during training.
10. **ANN Architecture & Regularization:** Builds a sequential deep neural network featuring:
    * Dense hidden layers with `ReLU` activations.
    * **Batch Normalization** layers to stabilize learning gradients.
    * **Dropout** layers ($0.3$ and $0.25$) to prevent overfitting.
    * A final output layer with a `sigmoid` activation for binary probability prediction.
11. **Training & Callbacks:** Compiles the model using the `Adam` optimizer and `binary_crossentropy` loss. Implements **Early Stopping** with a patience parameter monitored against validation loss (`val_loss`) to restore optimal weights automatically.
12. **Evaluation & Performance Metrics:** Evaluates the model on unseen test data using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
13. **Hyperparameter Experiments:** Implements a secondary model variation with deeper layers, adjusted dropout configurations, and alternative learning rates to compare metric performance.
14. **Model Serialization:** Exports the final trained production model as `credit_risk_ann.keras`.

---

## 🛠️ Installation & Requirements

Ensure you have Python installed along with the required deep learning and data processing packages:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

---

## 🚀 Running the Project

1. Clone the repository and navigate to the project directory.
2. Ensure the `statlog+german+credit+data/german.data` file is correctly positioned.
3. Run the Jupyter Notebook cell-by-cell or execute the script to train the model, observe training/validation loss curves, and evaluate test metrics.