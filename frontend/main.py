import time
import sys, os
sys.path.append("yolov5")
from PIL import Image
import numpy as np

import streamlit as st
import torch

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes

from ObjectDetUtils.draw import draw_boxes_with_label


def store_model():
    prefix = "checkpoints"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    all_models = ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt"]
    model_list = []
    for i, model in enumerate(all_models):
        model_list.append(
            DetectMultiBackend(
                weights=f"{prefix}/{model}",
                fp16=False
            )
        )
    return model_list, ["YOLOv5n", "YOLOv5s", "YOLOv5m"]


def _upload_image():
    file = st.file_uploader("Upload an image!")
    if file is not None:
        numpy_img = np.asarray(Image.open(file))
        return numpy_img
    else:
        return None


def detect(model_list, idx, image):
    placeholder = st.empty()
    placeholder.text("Detecting...")
    model = model_list[idx]
    image = letterbox(image, (640, 640))[0]
    image = image.transpose((2, 0, 1))[None]
    image = torch.from_numpy(image).float() / 255.
    result = model(image)
    result = non_max_suppression(result)[0]
    placeholder.text("Done!")
    placeholder.empty()
    return result, image


def draw_result(det_img, image, result):
    result[:, :4] = scale_boxes(det_img.shape[2:], result[:, :4], image.shape[:2])
    xyxys, labels = result[:, :4].type(torch.int), result[:, -1]
    img = draw_boxes_with_label(image, xyxys, labels)
    return img


@st.cache(suppress_st_warning=True)
def _start():
    placeholder = st.empty()
    placeholder.text("Loading all models...")
    model_list = store_model()
    placeholder.text("Done loading! All model are 'yolov5n', yolov5s', 'yolov5m'")
    time.sleep(2)
    placeholder.empty()
    return model_list


if __name__ == '__main__':
    st.title("YOLOv5 Detect")
    model_list, all_models = _start()
    image = _upload_image()
    if image is not None:
        st.image(image)
    model = st.selectbox("Choose model", ["YOLOv5n", "YOLOv5s", "YOLOv5m"])
    st.write(model)
    idx = all_models.index(model)
    result = None
    if image is not None and model is not None:
        result, det_img = detect(model_list, idx, image)
        if result is not None:
            img = draw_result(det_img, image, result)
            st.image(img)
    st.write(result)
