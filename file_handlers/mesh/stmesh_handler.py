from __future__ import annotations

from pathlib import Path

from file_handlers.base_handler import BaseFileHandler
from file_handlers.mdf.mdf_file import MdfFile
from utils.resource_file_utils import (
    ResourceResolutionContext,
    resource_context_for_app,
    resource_version_from_path,
)

from .stmesh_file import STMESH_MAGIC, StMeshFile


class StMeshHandler(BaseFileHandler):
    """Read-only WOTS SpeedTree mesh handler."""

    def __init__(self):
        super().__init__()
        self.mesh: StMeshFile | None = None
        self.game_version = ""
        self.diagnostics: list[str] = []
        self._material_skinning_cache: dict[tuple[str, str], dict[str, int]] = {}
        self._mmtr_skinning_cache: dict[tuple[str, str], int] = {}

    @classmethod
    def can_handle(cls, data: bytes) -> bool:
        return len(data) >= 4 and data[:4] == STMESH_MAGIC

    @classmethod
    def from_bytes(
        cls,
        filepath: str,
        data: bytes,
        *,
        app=None,
        resource_context: ResourceResolutionContext | None = None,
        game_version: str = "",
    ) -> "StMeshHandler":
        handler = cls()
        handler.filepath = filepath
        handler.app = app
        handler.resource_context = resource_context or resource_context_for_app(
            app, game=game_version
        )
        handler.game_version = str(
            game_version or getattr(handler.resource_context, "game", "") or ""
        )
        handler.read(data)
        return handler

    def supports_editing(self) -> bool:
        return False

    def rebuild(self) -> bytes:
        raise ValueError("WOTS STMESH support is read-only")

    def _file_version(self) -> int:
        version = resource_version_from_path(self.filepath, "stmesh")
        if version is None:
            raise ValueError(
                f"STMESH filename needs a numeric version suffix: {self.filepath}"
            )
        return version

    def _companion_mdf_path(self) -> Path | None:
        if not self.filepath:
            return None
        path = Path(self.filepath)
        marker = path.name.lower().find(".stmesh")
        if marker < 0:
            return None
        stem = path.name[:marker]
        for candidate in (
            path.with_name(f"{stem}_A.mdf2.51"),
            path.with_name(f"{stem}_A.mdf2"),
            path.with_name(f"{stem}.mdf2.51"),
            path.with_name(f"{stem}.mdf2"),
        ):
            if candidate.is_file():
                return candidate
        return None

    def _material_names(self) -> list[str]:
        companion = self._companion_mdf_path()
        if companion is None:
            self.diagnostics.append(
                "Material dependency missing: expected sibling *_A.mdf2.51; "
                "geometry remains available without textures."
            )
            return []
        try:
            mdf = MdfFile()
            if not mdf.read(companion.read_bytes(), str(companion)):
                raise ValueError("MDF parser rejected the file")
            return [material.header.mat_name for material in mdf.materials]
        except Exception as exc:
            self.diagnostics.append(
                f"Could not read material dependency {companion}: {exc}; "
                "geometry remains available without textures."
            )
            return []

    def read(self, data: bytes):
        self.diagnostics.clear()
        mesh = StMeshFile()
        mesh.read(
            data,
            file_version=self._file_version(),
            material_names=self._material_names(),
        )
        self.mesh = mesh
        self.modified = False

    def create_viewer(self):
        from .mesh_viewer import MeshViewer

        viewer = MeshViewer(self)
        viewer.modified_changed.connect(self.modified_changed.emit)
        return viewer


__all__ = ["StMeshHandler"]
