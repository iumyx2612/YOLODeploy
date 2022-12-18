from enum import Enum

from fastapi import FastAPI

from .models import yolov5n, yolov5s, yolov5m, DetectMultiBackend

app = FastAPI()


class ModelName(DetectMultiBackend, Enum):
    yolov5n = yolov5n
    yolov5s = yolov5s
    yolov5m = yolov5m


@app.get("/models/{model_name}")
def get_model(model_name: ModelName):
    return model_name.value