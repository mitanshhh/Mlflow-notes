import mlflow

# Log the metric r2_score as "r2_score"
mlflow.log_metric("r2_score",r2_score)

# Log parameter n_jobs as "n_jobs"
mlflow.log_param("n_jobs",n_jobs)

# Log the training code
mlflow.log_artifact("train.py")