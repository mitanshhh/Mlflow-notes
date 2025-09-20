import mlflow

# Log model to MLflow Tracking
mlflow.sklearn.log_model(lr_model,"lr_tracking")

# Get the last run
run = mlflow.last_active_run()

# Get the run_id of the above run
run_id = run.info.run_id

# Load model from MLflow Tracking
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lr_tracking")

