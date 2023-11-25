from typing import Type, List
import uuid
from fastapi import Depends, BackgroundTasks
from loguru import logger
from minio import S3Error

from models.Video import Video
from configs.MinioConfig import minio_client, bucket
from repositories.Video import VideoRepository
from tasks.worker import mp4_to_hls_minio
from utils.errors import ErrEntityNotFound


class VideoService:
    def __init__(self,
                 video_repo: VideoRepository = Depends()):
        self.video_repo = video_repo

    def create(self, video: Type[Video], file_content: bytes) -> Type[Video]:
        logger.debug("Video - Service - create")

        video.id = uuid.uuid4()
        video.path = f"{str(video.id)}_{video.name}"
        video.status = "processing"

        mp4_to_hls_minio.delay(file_content, video.path, video.id)

        result = self.video_repo.create(video)
        return result

    def get_list(self) -> List[Type[Video]]:
        logger.debug("Video - Service - get_users")
        result = self.video_repo.get_list()
        return result

    def get_by_id(self, id: uuid.UUID) -> Type[Video]:
        logger.debug("Video - Service - get_user_by_id")
        result = self.video_repo.get_by_id(id)
        return result

    def check_if_exist(self, id: uuid.UUID) -> Type[Video]:
        logger.debug("MinioStorage - Service - check_if_exist")
        result = self.video_repo.get_by_id(id)

        try:
            minio_client.stat_object(
                bucket,
                f"{result.path}/{result.path}.m3u8",
            )
            return result
        except S3Error as e:
            if e.code == 'NoSuchKey':
                raise ErrEntityNotFound("There is no this file in minio")

    def delete(self, id: uuid.UUID) -> None:
        logger.debug("Video - Service - delete_user")

        result = self.video_repo.get_by_id(id)

        minio_client.remove_object(
            bucket,
            result.path + result.name
        )

        self.video_repo.delete(result)
        return None
