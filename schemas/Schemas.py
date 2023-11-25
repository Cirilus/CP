from typing import List
from schemas.Screenshot import Screenshot
from schemas.Video import Video


class VideoSchema(Video):
    screenshots: List[Screenshot]
