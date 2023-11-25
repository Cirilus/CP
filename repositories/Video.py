import uuid
from typing import List, Type

from fastapi import Depends
from loguru import logger

from configs.Database import get_db_connection
from models.Video import Video
from sqlalchemy.orm import Session, lazyload
from utils.errors import ErrEntityNotFound


class VideoRepository:
    def __init__(self, db: Session = Depends(get_db_connection)):
        self.db = db

    def get_list(self) -> List[Type[Video]]:
        logger.debug("Video - Repository - get_list")
        query = self.db.query(Video)
        video_list = query.all()
        return video_list

    def get_by_id(self, id: uuid.UUID) -> Type[Video]:
        logger.debug("Video - Repository - get_by_id")
        video = self.db.get(
            Video,
            id
        )
        if video is None:
            raise ErrEntityNotFound("error entity not found")
        return video

    def create(self, video: Type[Video]) -> Type[Video]:
        logger.debug("Video - Repository - create")
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

    def delete(self, video: Type[Video]) -> None:
        logger.debug("Video - Repository - delete")
        self.db.delete(video)
        self.db.commit()
        self.db.flush()
