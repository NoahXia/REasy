from __future__ import annotations

import struct

from file_handlers.base_handler import BaseFileHandler

from .errors import MotionWriteError
from .mot_list.model import EmbeddedPayload, MotList, MotionSlot, MotionSlotType
from .motlist_file import MotListFile
from .wots_codec import WOTS_MOTION_FORMAT_CODEC


class WotsMotHandler(BaseFileHandler):
    """Read-only application handler for a standalone WOTS MOT v973."""

    def __init__(self):
        super().__init__()
        self.motlist_file: MotListFile | None = None
        self.raw_data: bytes = b""

    @classmethod
    def can_handle(cls, data: bytes) -> bool:
        return (
            len(data) >= 8
            and struct.unpack_from("<I", data)[0] == 973
            and data[4:8] == b"mot "
        )

    def supports_editing(self) -> bool:
        return False

    def read(self, data: bytes) -> None:
        motion = WOTS_MOTION_FORMAT_CODEC.parse_mot(
            data, label=self.filepath or "MOT v973"
        )
        facade = MotListFile(WOTS_MOTION_FORMAT_CODEC)
        facade.model = MotList(
            name=motion.name,
            slots=[MotionSlot(0, MotionSlotType.MOT, EmbeddedPayload(motion))],
        )
        self.motlist_file = facade
        self.raw_data = data
        self.modified = False

    @property
    def model(self) -> MotList:
        if self.motlist_file is None:
            raise MotionWriteError("no MOT v973 is loaded")
        return self.motlist_file.model

    def rebuild(self) -> bytes:
        raise MotionWriteError("WOTS MOT v973 is read-only")

    def create_viewer(self):
        from .preview.widget import MotListPreviewWidget

        viewer = MotListPreviewWidget(self)
        viewer.modified_changed.connect(self.modified_changed.emit)
        return viewer
