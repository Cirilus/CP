import cv2
from loguru import logger
from ultralytics import YOLO

model = YOLO("ml/best.pt")

def on_epoch_end(model, metrics, **kwargs):
    logger.info(f"Epoch {model.current_epoch}: {metrics}")


model.add_callback("on_epoch_end", on_epoch_end)
