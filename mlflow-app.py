import os
from mlflow import log_metric, log_param, log_artifact

if __name__ == "__main__":
    log_param("param1", 5)
    log_metric("foo", 1)
    log_metric("foo", 2)
    log_metric("foo", 3)

    # Log an artifact 
    with open("output.txt", "w") as f:
        f.write("Hello world!")
    log_artifact("output.txt")