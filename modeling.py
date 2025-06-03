import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

X_train_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_train_scaled.csv')
X_test_scaled = pd.read_csv('breast_cancer_dataset_preprocessing/X_test_scaled.csv')
y_train = pd.read_csv('breast_cancer_dataset_preprocessing/y_train.csv')
y_test = pd.read_csv('breast_cancer_dataset_preprocessing/y_test.csv')

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

mlflow.set_experiment("xgboost_experiment")

mlflow.xgboost.autolog() 

with mlflow.start_run():
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        use_label_encoder=False, 
        eval_metric='logloss'
    )

    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Model dilatih dan dicatat di MLflow (dengan autologging).")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)