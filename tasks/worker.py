import os
import tempfile
import time
import uuid
from io import BytesIO

import boto3
import cv2
from PIL import Image
from celery import Celery
from ffmpeg_streaming import S3
from loguru import logger
from sqlalchemy.orm import Session

from configs.Database import get_selery_connection
from configs.Environment import get_environment_variables
from ml.classificator import classify_image
from ml.model import model
from models.Screenshots import Screenshot
from configs.MinioConfig import minio_client
from models.Video import Video

env = get_environment_variables()

celery = Celery("tasks", broker=env.REDIS_HOST)

celery.conf.update(
    worker_log_level='INFO',
)

minio = S3(
    aws_access_key_id=env.MINIO_ACCESS,
    aws_secret_access_key=env.MINIO_SECRET,
    endpoint_url=f"http://{env.MINIO_HOST}",
    use_ssl=False,
    verify=False,
    config=boto3.session.Config(signature_version='s3v4'),
)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def numpy_to_bytes(img) -> BytesIO:
    pil_img = Image.fromarray(img)
    image_bytes = BytesIO()
    pil_img.save(image_bytes, format='PNG')

    image_bytes.seek(0)
    return image_bytes


def save_screenshot(img, db, video_id, name, time, type):
    image_bytes = numpy_to_bytes(img)

    sc_id = uuid.uuid4()
    sc = Screenshot(
        id=sc_id,
        time=time,
        path=f"{name}/{sc_id}",
        type=type,
        video_id=video_id,
    )
    logger.debug(f"save to bd, id={sc_id}")
    db.add(sc)
    db.commit()

    logger.debug(f"save to minio, path={name}/{sc_id}")
    minio_client.put_object(
        "static",
        f"{name}/{sc_id}",
        data=image_bytes,
        length=image_bytes.getbuffer().nbytes,
        content_type='image/png'
    )


def crop_video(input_video_path, output_video_path, start_time, end_time):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the start and end frames
    start_frame = int(fps * start_time)
    end_frame = int(fps * end_time)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Crop the video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break  # Stop when end of video or end time is reached
        if frame_count >= start_frame:
            out.write(frame)  # Write the frame to the output
        frame_count += 1

    # Clean up
    cap.release()
    out.release()


def detect_objects_in_video(video_path, model, db, video_id, name, n=3):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when end of video

        if frame_count % n == 0:  # Process every nth frame
            # Perform detection
            results = model.predict(frame, imgsz=720, conf=0.7)
            if results[0].boxes.xyxy.cpu().numpy().size > 0:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()[0]

                x1, y1, x2, y2 = int(boxes_xyxy[0]), int(boxes_xyxy[1]), int(boxes_xyxy[2]), int(boxes_xyxy[3])
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                if (width / 3 <= center_x <= 2 * width / 3 and
                        height / 3 <= center_y <= 2 * height / 3):

                    type = classify_image(frame)
                    time = frame*fps
                    save_screenshot(frame, db, video_id, name, time, type)

        frame_count += 1

    # Clean up
    cap.release()


@celery.task
def mp4_to_hls_minio(stream: bytes, name: str, video_id: uuid.UUID):
    db: Session = get_selery_connection()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mp4 = os.path.join(temp_dir, f"{name}")

        temp_cropped_mp4 = os.path.join(temp_dir, f"{name}_crop.mp4v")

        start_time = (1 * 60) + 50
        end_time = (2 * 60) + 10

        temp_hls = os.path.join(temp_dir, f"{name}_hls")

        logger.debug("created the tmp dir")
        mkdir(temp_hls)

        logger.debug("writing the mp4")
        with open(temp_mp4, 'wb') as f:
            f.write(stream)

        logger.debug(f"cropped video, path={temp_cropped_mp4}")
        crop_video(temp_mp4, temp_cropped_mp4, start_time, end_time)

        time.sleep(10)

        logger.debug("predicting")
        detect_objects_in_video(temp_mp4, model, db, video_id, name, 3)

        video = db.query(Video).filter_by(id=video_id).first()

        video.status="completed"
        db.commit()

        db.close()
