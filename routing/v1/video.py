import uuid
import ffmpeg
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
import cv2
from typing import List
from collections import Counter
from models.Video import Video
from services.Video import VideoService
from schemas.Schemas import VideoSchema, VideoSchemaList
from utils.wrappers import error_wrapper
from ml.model import model

router = APIRouter(prefix="/api/v1/ml", tags=["company"])


@router.post(
    "/load",
    summary="loading the video",
    response_model=VideoSchema
)
async def load(video: UploadFile = File(...), video_service: VideoService = Depends()):
    video_content = video.file.read()
    video.file.seek(0)

    file = Video(
        name=video.filename,
    )

    result = video_service.create(file, video_content)

    return result


@router.get(
    "/list",
    summary="returning the list of the videos",
    response_model=List[VideoSchemaList],
)
async def list(minio_service: VideoService = Depends()):
    results = error_wrapper(minio_service.get_list)

    answers = []
    for result in results:
        types_counter = Counter()
        for screenshot in result.screenshots:
            types_counter[screenshot.type] += 1

        if len(types_counter.most_common(1)) > 0:
            answer = VideoSchemaList(
                id=result.id,
                name=result.name,
                path=result.path,
                status=result.status,
                created=result.created,
                screenshots=result.screenshots,
                popular=types_counter.most_common(1)[0][0]
            )
            answers.append(answer)
            continue
        answers.append(VideoSchemaList(
            id=result.id,
            name=result.name,
            path=result.path,
            status=result.status,
            created=result.created,
            screenshots=result.screenshots,
        ))

    return answers


@router.get(
    "/get",
    summary="returning the info about video",
    response_model=VideoSchema
)
async def get(id: uuid.UUID, minio_service: VideoService = Depends()):
    result = error_wrapper(minio_service.get_by_id, id)

    return result


@router.get(
    "/check",
    summary="if the video uploading to minio",
)
async def check(id: uuid.UUID, minio_service: VideoService = Depends()):
    file = error_wrapper(minio_service.check_if_exist, id)

    return {"url": f"static/{file.path}/{file.path}.m3u8"}


def streamer(url_rtsp: str):
    cap = cv2.VideoCapture(url_rtsp)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        model.predict(source=frame, stream=True)

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@router.get("/rl")
async def video_feed(url_rtsp: str):
    return StreamingResponse(streamer(url_rtsp), media_type="multipart/x-mixed-replace;boundary=frame")


@router.get("/rl_hls")
async def video_feed(url_rtsp: str):
    ffmpeg_cmd = (
        ffmpeg.input(url_rtsp, rtsp_transport="tcp")
        .output("pipe:", format="hls", hls_time=2, hls_list_size=5)
        .run_async(pipe_stdout=True)
    )

    async def generate():
        try:
            while True:

                chunk = ffmpeg_cmd.stdout.read(1024)
                if not chunk:
                    break
                yield chunk
        finally:

            ffmpeg_cmd.stdout.close()
            await ffmpeg_cmd.wait()

    return StreamingResponse(generate(), media_type="application/vnd.apple.mpegurl")
