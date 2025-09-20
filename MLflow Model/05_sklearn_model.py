import mlflow

# Log the pyfunc model 
mlflow.pyfunc.log_model(
	artifact_path="lr_pyfunc", 
    # Set model to use CustomPredict Class
	python_model=____, 
	artifacts={"lr_model": "lr_model"}
)

run = mlflow.last_active_run()
run_id = run.info.run_id

# Load the model in python_function format
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/lr_pyfunc")

