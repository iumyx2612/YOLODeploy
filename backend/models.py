import sys, os
sys.path.append("yolov5")

from yolov5.models.common import DetectMultiBackend


yolov5n = DetectMultiBackend(
    weights="checkpoints/yolov5n.pt"
)

yolov5s = DetectMultiBackend(
    weights="checkpoints/yolov5s.pt"
)

yolov5m = DetectMultiBackend(
    weights="checkpoints/yolov5m.pt"
)


def load_models():
    prefix = "checkpoints"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    all_models = ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt"]
    for i, model in enumerate(all_models):
            DetectMultiBackend(
                weights=f"{prefix}/{model}",
                fp16=False)
