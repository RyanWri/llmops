import mlflow
import torch

params = {
    "model_name": "distilbert-base-uncased",
    "epochs": 100,
    "lr": 0.1,
    "momentum": 0.8,
    "dataset_name": "imdb",
    "task_name": "sequence_classification",
}


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(params["task_name"])

# with context to automaticlly end run
with mlflow.start_run(
    run_name=f"{params['model_name']}-{params['dataset_name']}-1"
) as run:
    # log params
    mlflow.log_params(params)
    mlflow.log_metric("score", 100)
