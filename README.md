# MLOPS_ASSIGNMENT2
README.md (for GitHub)
text
# Titanic Survival Pipeline with Airflow & MLflow

This repository contains an end-to-end machine learning pipeline for predicting Titanic passenger survival, using **Apache Airflow** for orchestration and **MLflow** for experiment tracking and model registry.[file:161][web:205]

## Features

- DAG with parallel preprocessing and branching, no cyclic dependencies.[file:161]
- Data ingestion, validation, missing-value handling, feature engineering, and encoding.
- Model training (Logistic Regression) with MLflow tracking.
- Model evaluation (accuracy, precision, recall, F1-score).
- Branching based on accuracy: register or reject model in MLflow Model Registry.
- Multiple runs with different hyperparameters for experiment comparison.[file:161][web:205]

## Project Structure

```text
.
├── mlops_airflow_mlflow_pipeline.py   # Main Airflow DAG
├── requirements.txt                   # Python dependencies
├── data/
│   └── titanic.csv                    # Titanic dataset (if included)
└── screenshots/
    ├── airflow/                       # Airflow DAG & logs screenshots
    └── mlflow/                        # MLflow runs & registry screenshots
Prerequisites
Python 3.12

Apache Airflow 2.10.x

MLflow

scikit-learn

pandas

See requirements.txt for the dependency list.

Setup Instructions
1. Clone the repository
bash
git clone <your-repo-url>.git
cd <your-repo-folder>
2. Create and activate virtual environment
bash
python3 -m venv venv
source venv/bin/activate  # On Windows (WSL): source venv/bin/activate
pip install --upgrade pip
3. Install dependencies
bash
pip install -r requirements.txt
4. Configure Airflow
Set AIRFLOW_HOME (optional but recommended):

bash
export AIRFLOW_HOME=/home/ammar/airflow   # adjust path if needed
Initialize the Airflow database:

bash
airflow db init
Copy the DAG file into the Airflow dags_folder (if not already there):

bash
mkdir -p $AIRFLOW_HOME/dags
cp mlops_airflow_mlflow_pipeline.py $AIRFLOW_HOME/dags/
Create an admin user:

bash
airflow users create \
  --username admin \
  --password admin \
  --firstname Ammar \
  --lastname Ahmed \
  --role Admin \
  --email admin@example.com
5. Start Airflow webserver and scheduler
In two separate terminals (with the venv activated):

bash
# Terminal 1 – Scheduler
cd <your-repo-folder>
source venv/bin/activate
airflow scheduler
bash
# Terminal 2 – Webserver
cd <your-repo-folder>
source venv/bin/activate
airflow webserver
Open the Airflow UI at:

text
http://localhost:8080
Log in with admin / admin (or your chosen credentials).

6. Prepare data
Place titanic.csv in the data/ directory:

bash
mkdir -p data
# copy titanic.csv into ./data
Update the DATA_PATH constant in mlops_airflow_mlflow_pipeline.py if required:

python
DATA_PATH = "/home/ammar/mlops-assignment2/data/titanic.csv"
(or adjust to your actual path).

7. Start MLflow UI
In another terminal:

bash
cd <your-repo-folder>
source venv/bin/activate
mlflow ui
Then open:

text
http://localhost:5000
You will see the MLflow experiment titanic_survival_experiment after running the DAG.[web:205]

Running the Pipeline
In Airflow UI, locate the DAG mlops_airflow_mlflow_pipeline.

Turn it On and click the Trigger DAG button.

Use Graph view to observe the workflow:

start → data_ingestion → data_validation → [handle_missing, feature_engineering] → data_encoding → model_training → model_evaluation → accuracy_branch → register_model/reject_model → processing_done → end.

Click each task to inspect logs, especially:

data_validation (retry behavior).

model_training (MLflow run id and dataset sizes).

model_evaluation (metrics).

register_model or reject_model (registration or rejection).

Hyperparameter Experiments (Task 10)
To run multiple experiments with different hyperparameters:

Open mlops_airflow_mlflow_pipeline.py.

In train_model_with_mlflow, adjust the params dictionary, for example:

python
params = {"C": 0.1, "max_iter": 200, "solver": "lbfgs"}
Save the file and trigger the DAG again from Airflow.

Repeat with different values (e.g., C = 1.0, C = 10.0).

In MLflow UI, open the titanic_survival_experiment and compare runs by parameters and metrics.[file:161][web:205]
