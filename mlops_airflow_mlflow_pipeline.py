from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.exceptions import AirflowException

import pandas as pd
import os

# ML / MLflow imports (for Tasks 6–9)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

# ---------- Config ----------
DATA_PATH = "/home/ammar/mlops-assignment2/data/titanic.csv"
MISSING_THRESHOLD = 0.30  # 30%
PROCESSED_DIR = "/home/ammar/mlops-assignment2/data/processed"
MLFLOW_EXPERIMENT_NAME = "titanic_survival_experiment"  # you will see this in MLflow UI

default_args = {
    "owner": "ammar",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "depends_on_past": False,
}

# ---------- Task functions ----------

# Task 2 – Data Ingestion
def ingest_data(**context):
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    missing_counts = df.isna().sum()
    print("Missing values:\n", missing_counts)

    # Push dataset path via XCom (Task 2 requirement)
    context["ti"].xcom_push(key="dataset_path", value=DATA_PATH)


# Task 3 – Data Validation
def validate_data(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="data_ingestion")
    if not dataset_path or not os.path.exists(dataset_path):
        raise AirflowException("Dataset path from XCom is invalid")

    df = pd.read_csv(dataset_path)
    total = len(df)

    missing_age = df["Age"].isna().sum() / total
    missing_embarked = df["Embarked"].isna().sum() / total

    print(f"Missing Age %: {missing_age:.2%}")
    print(f"Missing Embarked %: {missing_embarked:.2%}")

    if missing_age > MISSING_THRESHOLD or missing_embarked > MISSING_THRESHOLD:
        raise AirflowException(
            "Missing percentage in Age or Embarked is above threshold"
        )


# Task 1 – Simple branching placeholder (will be re-used for Task 8 later)
def branch_after_validation(**context):
    """Simple branch so we already have branching in DAG (Task 1 theme)."""
    # For now we always go to 'processing_done'; later we can expand for Task 8.
    return "processing_done"


# Task 4 – Handle missing values and save cleaned CSV
def handle_missing_and_save(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="data_ingestion")
    df = pd.read_csv(dataset_path)

    # Handle missing Age (median) and Embarked (mode)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cleaned_path = os.path.join(PROCESSED_DIR, "titanic_cleaned.csv")
    df.to_csv(cleaned_path, index=False)

    print("Saved cleaned data to:", cleaned_path)
    ti.xcom_push(key="cleaned_path", value=cleaned_path)


# Task 4 – Feature engineering (FamilySize, IsAlone)
def feature_engineering(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="data_ingestion")
    df = pd.read_csv(dataset_path)

    # FamilySize = SibSp + Parch + 1
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    # IsAlone = 1 if FamilySize == 1 else 0
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feat_path = os.path.join(PROCESSED_DIR, "titanic_features.csv")
    df.to_csv(feat_path, index=False)

    print("Saved feature engineered data to:", feat_path)
    ti.xcom_push(key="feat_path", value=feat_path)


# Task 5 – Merge cleaned + features, encode categoricals, drop columns
def merge_and_encode(**context):
    ti = context["ti"]
    cleaned_path = ti.xcom_pull(key="cleaned_path", task_ids="handle_missing")
    feat_path = ti.xcom_pull(key="feat_path", task_ids="feature_engineering")

    if not cleaned_path or not feat_path:
        raise AirflowException("Missing paths from previous tasks")

    df_clean = pd.read_csv(cleaned_path)
    df_feat = pd.read_csv(feat_path)

    # Align on index and combine columns
    df = df_clean.copy()
    df["FamilySize"] = df_feat["FamilySize"]
    df["IsAlone"] = df_feat["IsAlone"]

    # Encode categorical variables (Sex, Embarked)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")

    # Drop irrelevant columns (Task 5 requirement)
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    final_path = os.path.join(PROCESSED_DIR, "titanic_final.csv")
    df.to_csv(final_path, index=False)
    print("Saved final encoded data to:", final_path)

    # Will be used in Task 6 (model training)
    ti.xcom_push(key="final_data_path", value=final_path)


# Task 6 – Model Training with MLflow
def train_model_with_mlflow(**context):
    ti = context["ti"]
    final_data_path = ti.xcom_pull(key="final_data_path", task_ids="data_encoding")
    if not final_data_path or not os.path.exists(final_data_path):
        raise AirflowException("Final data path not found for training")

    df = pd.read_csv(final_data_path)

    # Assume 'Survived' is the target
    if "Survived" not in df.columns:
        raise AirflowException("Survived column not found in final dataset")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameters (you can change between runs for Task 10)
    model_type = "LogisticRegression"
    params = {"C": 0.5, "max_iter": 200, "solver": "lbfgs"}

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="titanic_lr_run") as run:
        model = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"],
        )
        model.fit(X_train, y_train)

        # Log params and dataset size
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Push things needed for evaluation / registration
        ti.xcom_push(key="run_id", value=run.info.run_id)
        ti.xcom_push(key="X_test", value=X_test.to_json())
        ti.xcom_push(key="y_test", value=y_test.to_json())


# Task 7 – Model Evaluation with MLflow
def evaluate_model_with_mlflow(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="model_training")
    X_test_json = ti.xcom_pull(key="X_test", task_ids="model_training")
    y_test_json = ti.xcom_pull(key="y_test", task_ids="model_training")

    if not run_id:
        raise AirflowException("run_id not found for evaluation")

    X_test = pd.read_json(X_test_json)
    y_test = pd.read_json(y_test_json, typ="series")

    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.start_run(run_id=run_id)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.end_run()

    print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")

    # Push accuracy for Task 8 branching
    ti.xcom_push(key="accuracy", value=acc)


# Task 8 – Branching Logic (register or reject model based on accuracy)
def branch_on_accuracy(**context):
    ti = context["ti"]
    acc = ti.xcom_pull(key="accuracy", task_ids="model_evaluation")

    threshold = 0.80
    if acc is None:
        raise AirflowException("Accuracy not found for branching")

    if acc >= threshold:
        print(f"Accuracy {acc} >= {threshold}, going to register_model")
        return "register_model"
    else:
        print(f"Accuracy {acc} < {threshold}, going to reject_model")
        return "reject_model"


# Task 9 – Model Registration / Rejection in MLflow
def register_or_reject_model(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="model_training")
    acc = ti.xcom_pull(key="accuracy", task_ids="model_evaluation")

    task_id = context["task"].task_id

    if task_id == "register_model":
        # Register approved model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri, name="titanic_survival_model"
        )
        print("Registered model:", registered_model)
    else:
        # Log rejection reason if accuracy is low
        mlflow.start_run(run_id=run_id)
        mlflow.set_tag("rejection_reason", f"Accuracy too low: {acc}")
        mlflow.end_run()
        print("Model rejected due to low accuracy:", acc)


# ---------- DAG definition ----------

with DAG(
    dag_id="mlops_airflow_mlflow_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    description="Titanic ML pipeline with Airflow + MLflow",
) as dag:

    # Task 1 – DAG design / start node
    start = EmptyOperator(task_id="start")

    # Task 2 – Data Ingestion
    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data,
        provide_context=True,
    )

    # Task 3 – Data Validation (with retry)
    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data,
        provide_context=True,
        retries=2,
        retry_delay=timedelta(seconds=30),
    )

    # (Old branching_after_validation kept for Task 1 requirement only)
    branching = BranchPythonOperator(
        task_id="branch_after_validation",
        python_callable=branch_after_validation,
        provide_context=True,
    )

    # Task 4 – Parallel processing
    handle_missing = PythonOperator(
        task_id="handle_missing",
        python_callable=handle_missing_and_save,
        provide_context=True,
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
        provide_context=True,
    )

    # Task 5 – Data Encoding
    data_encoding = PythonOperator(
        task_id="data_encoding",
        python_callable=merge_and_encode,
        provide_context=True,
    )

    # Task 6 – Model Training
    model_training = PythonOperator(
        task_id="model_training",
        python_callable=train_model_with_mlflow,
        provide_context=True,
    )

    # Task 7 – Model Evaluation
    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=evaluate_model_with_mlflow,
        provide_context=True,
    )

    # Task 8 – Branching based on accuracy
    accuracy_branch = BranchPythonOperator(
        task_id="accuracy_branch",
        python_callable=branch_on_accuracy,
        provide_context=True,
    )

    # Task 9 – Register / Reject model
    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_or_reject_model,
        provide_context=True,
    )

    reject_model = PythonOperator(
        task_id="reject_model",
        python_callable=register_or_reject_model,
        provide_context=True,
    )

    processing_done = EmptyOperator(task_id="processing_done")
    end = EmptyOperator(task_id="end")

    # Dependencies up to Task 5
    start >> data_ingestion >> data_validation
    data_validation >> [handle_missing, feature_engineering_task]
    [handle_missing, feature_engineering_task] >> data_encoding

    # Tasks 6–9 chain
    data_encoding >> model_training >> model_evaluation >> accuracy_branch
    accuracy_branch >> [register_model, reject_model]
    [register_model, reject_model] >> processing_done >> end
