# 🧬 Membangun_model

<p align="center">
  <strong>Project machine learning untuk klasifikasi breast cancer dengan XGBoost, MLflow tracking, tuning, dan interpretasi model.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/ML-XGBoost-orange" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Tracking-MLflow-0194E2" alt="MLflow"/>
  <img src="https://img.shields.io/badge/Explainability-SHAP-purple" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Experiment-DagsHub-green" alt="DagsHub"/>
</p>

---

## 📖 Tentang Project

**Membangun_model** adalah project machine learning untuk membangun model klasifikasi breast cancer. Dataset yang digunakan sudah melalui tahap preprocessing, lalu model dilatih menggunakan **XGBoost** dengan bantuan **SMOTE** untuk menangani imbalance data.

Project ini juga mencatat eksperimen menggunakan **MLflow**, melakukan tuning hyperparameter, menyimpan artefak evaluasi, dan menambahkan visualisasi interpretasi model menggunakan **SHAP**.

## ✨ Fitur

| Fitur | Deskripsi |
|-------|-----------|
| 🧪 **Training Baseline** | Melatih model XGBoost dengan konfigurasi awal |
| ⚖️ **SMOTE** | Menyeimbangkan data training sebelum proses training |
| 📊 **Metric Evaluation** | Menghitung accuracy, precision, recall, dan F1-score |
| 🧾 **MLflow Tracking** | Mencatat metric, parameter, model, dan artifact eksperimen |
| 🎯 **Hyperparameter Tuning** | Menggunakan `RandomizedSearchCV` untuk mencari parameter terbaik |
| 🔍 **Model Explainability** | Membuat visualisasi SHAP untuk melihat pengaruh fitur |
| 📉 **Confusion Matrix** | Menyimpan visualisasi evaluasi klasifikasi |

## 🛠️ Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Language | Python |
| Data | Pandas |
| ML | Scikit-learn, XGBoost, Imbalanced-learn |
| Tracking | MLflow, DagsHub |
| Visualization | Matplotlib, Seaborn, SHAP, mpld3 |
| Tuning | RandomizedSearchCV |

## 📂 Struktur Project

```text
Membangun_model/
├── breast_cancer_dataset_preprocessing/
│   ├── X_train_scaled.csv
│   ├── X_test_scaled.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── feature_names.txt
│   └── scaler.joblib
├── modeling.py                # Training baseline + MLflow autolog
├── modeling_tuning.py         # Hyperparameter tuning + logging ke DagsHub
├── requirements.txt           # Dependensi Python
├── DagsHub.txt                # Link eksperimen DagsHub
├── screenshot_artifact/       # Bukti artifact MLflow
└── screnshot_dashboard/       # Bukti dashboard metric MLflow
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Training Baseline

```bash
python modeling.py
```

### 3. Jalankan Training dengan Tuning

```bash
python modeling_tuning.py
```

## 🧮 Alur Machine Learning

```text
Preprocessed Dataset
        ↓
Load X_train, X_test, y_train, y_test
        ↓
SMOTE Oversampling
        ↓
Train XGBoost Classifier
        ↓
Evaluate Model
        ↓
Log Metrics + Artifacts ke MLflow
        ↓
SHAP + Confusion Matrix
```

## 📊 Output Eksperimen

| Output | Deskripsi |
|--------|-----------|
| `metric_info.json` | Ringkasan metric model |
| `confusion_matrix.png` | Visualisasi confusion matrix |
| `estimator.html` / SHAP plot | Interpretasi fitur model |
| MLflow run | Catatan eksperimen dan artifact |

## 📄 Catatan

Pastikan semua file dataset di folder `breast_cancer_dataset_preprocessing/` tersedia sebelum menjalankan script. Untuk `modeling_tuning.py`, koneksi ke DagsHub diperlukan agar tracking eksperimen berjalan.