import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import dagshub 
import os

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

try:

    dagshub.init(repo_owner='a548ybm523', repo_name='my-first-repo', mlflow=True) 
    print("Berhasil terhubung ke DagsHub untuk MLflow tracking (modeling_tuning.py).")
except Exception as e:
    print(f"Gagal menginisialisasi DagsHub (modeling_tuning.py): {e}")
    print("Pastikan Anda sudah login ke DagsHub melalui CLI atau browser jika diminta, dan library dagshub terinstal.")

try:
    X_train_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_train_scaled.csv')
    X_test_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_test_scaled.csv')
    y_train = pd.read_csv('breast_cancer_dataset_preprocessing/y_train.csv')
    y_test = pd.read_csv('breast_cancer_dataset_preprocessing/y_test.csv')
except FileNotFoundError as e:
    print(f"Error: File dataset tidak ditemukan. Pastikan path sudah benar. Detail: {e}")
    exit()

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

mlflow.set_experiment("xgboost_hyperparameter_tuning_dagshub")

param_grid = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.29), 
    'subsample': uniform(0.6, 0.4),      
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)             
}

with mlflow.start_run(run_name="XGBoost_RandomizedSearch_Tuning_DagsHub"):

    xgb_model_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_model_base,
        param_distributions=param_grid,
        n_iter=50,  
        cv=5,       
        scoring='f1', 
        n_jobs=-1,  
        random_state=42,
        verbose=1   
    )

    print("Memulai Hyperparameter Tuning dengan RandomizedSearchCV...")
    random_search.fit(X_train_smote, y_train_smote)
    print("Tuning selesai.")

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score_cv = random_search.best_score_ 

    print("\nParameter terbaik yang ditemukan:")
    print(best_params)
    print(f"\nSkor F1 terbaik dari Cross-Validation: {best_score_cv:.4f}")

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_f1_score", best_score_cv)

    y_pred = best_model.predict(X_test_scaled) 

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)
    mlflow.log_metric("test_f1_score", f1)

    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="best_xgb_model_tuned")

    print("\nModel terbaik telah dilatih, dievaluasi, dan dicatat di MLflow.")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Best Tuned Model')
    confusion_matrix_path = "confusion_matrix_tuned.png"
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path)

    print("\nClassification Report Model Terbaik di Data Test:")
    report_text = classification_report(y_test, y_pred)
    print(report_text)

    classification_report_path = "classification_report_tuned.txt"
    with open(classification_report_path, "w") as f:
        f.write(report_text)
    mlflow.log_artifact(classification_report_path)

print("\nProses modeling_tuning.py selesai. Periksa MLflow UI di DagsHub untuk melihat hasilnya.")