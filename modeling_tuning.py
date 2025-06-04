import pandas as pd
import mlflow
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

try:
    dagshub.init(repo_owner='a548ybm523', repo_name='my-first-repo', mlflow=True)
    print("✅ Terhubung ke DagsHub untuk MLflow tracking (modeling_tuning.py).")
except Exception as e:
    print(f"❌ Gagal koneksi ke DagsHub: {e}")
    exit()

try:
    X_train_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_train_scaled.csv')
    X_test_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_test_scaled.csv')
    y_train = pd.read_csv('breast_cancer_dataset_preprocessing/y_train.csv')
    y_test = pd.read_csv('breast_cancer_dataset_preprocessing/y_test.csv')
except FileNotFoundError as e:
    print(f"❌ File tidak ditemukan: {e}")
    exit()

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

mlflow.set_experiment("model_tuning_manual_loging")

param_grid = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.29),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)
}

with mlflow.start_run(run_name="Model_tuning_manual_loging"):

    base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    print("🔍 Memulai hyperparameter tuning...")
    random_search.fit(X_train_smote, y_train_smote)
    print("✅ Tuning selesai.")

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score_cv = random_search.best_score_

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

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Best Tuned Model')
    cm_path = "confusion_matrix_tuned.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    report_text = classification_report(y_test, y_pred)
    print("\n📋 Classification Report:\n", report_text)
    report_path = "classification_report_tuned.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    mlflow.log_artifact(report_path)

    print("📊 Menghitung SHAP values...")
    explainer = shap.Explainer(best_model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    shap.summary_plot(shap_values, X_test_scaled, show=False)
    shap_path = "shap_summary_tuned.png"
    plt.savefig(shap_path, bbox_inches="tight")
    mlflow.log_artifact(shap_path)
    plt.close()

    print("📈 Menyimpan Feature Importance XGBoost...")
    xgb.plot_importance(best_model, importance_type='gain', title='XGBoost Feature Importance', show_values=False)
    plt.tight_layout()
    fi_path = "xgb_feature_importance.png"
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path)
    plt.close()

print("\n🎉 Selesai! Cek hasil eksperimen dan artefak di DagsHub MLflow UI.")
