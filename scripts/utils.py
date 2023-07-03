import os
from datetime import datetime


def create_run_directory(model_type) -> str:
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y")
    head_directory = f"experiments/supervised/{model_type}/experiments_{date_time.replace('/', '_')}"
    i = 1
    while os.path.exists(os.path.join(head_directory,f"run_{i}")):
        i += 1
    run_directory = os.path.join(head_directory,f"run_{i}")
    os.makedirs(run_directory)
    return run_directory
   