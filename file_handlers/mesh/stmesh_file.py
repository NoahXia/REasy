from __future__ import annotations

import math
import struct
from array import array
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


STMESH_MAGIC = b"STM8"
WOTS_STMESH_VERSION = 260209350


class StMeshParseError(ValueError):
    """A malformed or unsupported RE Engine SpeedTree mesh."""


@dataclass(frozen=True, slots=True)
class StMeshBlock:
    lod_index: int
    offset: int
    end_offset: int
    vertex_count: int
    triangle_count: int
    material_index: int
    batch_type: int
    vertex_flags: int


@dataclass(frozen=True, slots=True)
class StMeshLod:
    index: int
    blocks: tuple[StMeshBlock, ...]


def _require(data: bytes, offset: int, size: int, what: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise StMeshParseError(
            f"{what} at 0x{offset:X} extends past file size 0x{len(data):X}"
        )


def _unpack(data: bytes, fmt: str, offset: int, what: str):
    size = struct.calcsize(fmt)
    _require(data, offset, size, what)
    return struct.unpack_from(fmt, data, offset)


def _octahedral_normals(raw: np.ndarray) -> np.ndarray:
    """Decode the two-byte octahedral normal stored by SpeedTree 8."""
    xy = raw[:, :2].astype(np.float32) / np.float32(255.0) * 2.0 - 1.0
    result = np.empty((len(raw), 3), dtype=np.float32)
    result[:, :2] = xy
    result[:, 2] = 1.0 - np.abs(xy[:, 0]) - np.abs(xy[:, 1])
    folded = result[:, 2] < 0.0
    if np.any(folded):
        old = result[folded, :2].copy()
        result[folded, 0] = (1.0 - np.abs(old[:, 1])) * np.sign(old[:, 0])
        result[folded, 1] = (1.0 - np.abs(old[:, 0])) * np.sign(old[:, 1])
    lengths = np.linalg.norm(result, axis=1)
    valid = lengths > 1e-8
    result[valid] /= lengths[valid, None]
    result[~valid] = (0.0, 1.0, 0.0)
    return result


class StMeshFile:
    """Read-only adapter for WOTS' RE Engine SpeedTree 8 mesh container.

    Public geometry attributes mirror ``MeshFile`` so the existing mesh
    viewer, material resolver and glTF exporter can consume this object.
    """

    def __init__(self):
        self.file_version = 0
        self.speedtree_version = 0
        self.metadata_size = 0
        self.geometry_offset = 0
        self.geometry_size = 0
        self.bounds_min = (0.0, 0.0, 0.0)
        self.bounds_max = (0.0, 0.0, 0.0)
        self.lods: list[StMeshLod] = []
        self.material_names: list[str] = []
        self.mesh_buffer = None
        self.meshes: list[object] = []
        self.joint_count = 0
        self.bone_remap_indices: list[int] = []
        self.names: list[str] = []
        self.bone_indices: list[int] = []
        self.local_matrices: list[list[float]] = []
        self.world_matrices: list[list[float]] = []
        self.inverse_bind_matrices: list[list[float]] = []
        self.streaming_buffer_count = 0
        self.streaming_data_loaded = False

    def read(
        self,
        data: bytes,
        *,
        file_version: int,
        material_names: list[str] | None = None,
    ) -> bool:
        if file_version != WOTS_STMESH_VERSION:
            raise StMeshParseError(
                f"unsupported STMESH file version {file_version}; "
                f"expected WOTS 1.0.1.0 version {WOTS_STMESH_VERSION}"
            )
        _require(data, 0, 16, "STMESH header")
        if data[:4] != STMESH_MAGIC:
            raise StMeshParseError("expected STM8 magic at 0x0")

        self.file_version = file_version
        self.speedtree_version = _unpack(data, "<I", 8, "SpeedTree version")[0]
        if self.speedtree_version != 16:
            raise StMeshParseError(
                f"unsupported SpeedTree layout {self.speedtree_version} at 0x8"
            )
        self.metadata_size = _unpack(data, "<I", 0xC, "metadata size")[0]
        geometry_locator = self.metadata_size + 0x1C0
        self.geometry_offset, self.geometry_size = _unpack(
            data, "<II", geometry_locator, "geometry locator"
        )
        geometry_end = self.geometry_offset + self.geometry_size
        if self.geometry_offset < geometry_locator + 8:
            raise StMeshParseError(
                f"geometry offset 0x{self.geometry_offset:X} overlaps metadata"
            )
        if geometry_end != len(data):
            raise StMeshParseError(
                f"geometry range ends at 0x{geometry_end:X}, "
                f"file ends at 0x{len(data):X}"
            )

        base = self.geometry_offset
        self.bounds_min = _unpack(data, "<3f", base, "geometry minimum bounds")
        lod_count = _unpack(data, "<I", base + 0xC, "LOD count")[0]
        self.bounds_max = _unpack(data, "<3f", base + 0x10, "geometry maximum bounds")
        if not 1 <= lod_count <= 16:
            raise StMeshParseError(
                f"unsupported LOD count {lod_count} at 0x{base + 0xC:X}"
            )
        if not all(
            math.isfinite(value) for value in (*self.bounds_min, *self.bounds_max)
        ):
            raise StMeshParseError("STMESH bounds contain non-finite values")
        if any(low > high for low, high in zip(self.bounds_min, self.bounds_max)):
            raise StMeshParseError("STMESH minimum bounds exceed maximum bounds")

        relative_lod_tables = _unpack(
            data, f"<{lod_count}I", base + 0x20, "LOD table pointers"
        )
        lod_block_offsets: list[list[int]] = []
        seen_offsets: set[int] = set()
        for lod_index, relative_table in enumerate(relative_lod_tables):
            table = base + relative_table
            if not relative_table or table < base or table + 4 > geometry_end:
                raise StMeshParseError(
                    f"LOD {lod_index} table pointer 0x{relative_table:X} is out of range"
                )
            block_count = _unpack(data, "<I", table, f"LOD {lod_index} block count")[0]
            if not 1 <= block_count <= 4096:
                raise StMeshParseError(
                    f"LOD {lod_index} has unsupported block count {block_count} at 0x{table:X}"
                )
            relative_blocks = list(
                _unpack(
                    data,
                    f"<{block_count}I",
                    table + 4,
                    f"LOD {lod_index} block offsets",
                )
            )
            for relative_block in relative_blocks:
                absolute = base + relative_block
                if (
                    relative_block == 0
                    or absolute < base
                    or absolute + 32 > geometry_end
                ):
                    raise StMeshParseError(
                        f"LOD {lod_index} block pointer 0x{relative_block:X} is out of range"
                    )
                if relative_block in seen_offsets:
                    raise StMeshParseError(
                        f"duplicate STMESH block pointer 0x{relative_block:X}"
                    )
                seen_offsets.add(relative_block)
            lod_block_offsets.append(relative_blocks)

        ordered = sorted(seen_offsets)
        block_ends = {
            relative: ordered[index + 1]
            if index + 1 < len(ordered)
            else self.geometry_size
            for index, relative in enumerate(ordered)
        }
        parsed_by_offset: dict[int, StMeshBlock] = {}
        for lod_index, relative_blocks in enumerate(lod_block_offsets):
            for relative in relative_blocks:
                offset = base + relative
                end = base + block_ends[relative]
                vertex_count, triangle_count, material_index, batch_type = _unpack(
                    data, "<4B", offset + 0xC, "block counts"
                )
                vertex_flags = _unpack(data, "<I", offset + 0x1C, "vertex flags")[0]
                if vertex_count == 0 or triangle_count == 0:
                    raise StMeshParseError(f"empty geometry block at 0x{offset:X}")
                index_end = offset + 0x20 + triangle_count * 3
                position_start = (index_end + 3) & ~3
                if position_start + vertex_count * 16 > end:
                    raise StMeshParseError(
                        f"vertex streams for block 0x{offset:X} exceed block end 0x{end:X}"
                    )
                indices = np.frombuffer(
                    data, dtype=np.uint8, count=triangle_count * 3, offset=offset + 0x20
                )
                maximum_index = int(indices.max(initial=0))
                if maximum_index >= vertex_count:
                    raise StMeshParseError(
                        f"block 0x{offset:X} index {maximum_index} exceeds "
                        f"vertex count {vertex_count}"
                    )
                parsed_by_offset[relative] = StMeshBlock(
                    lod_index=lod_index,
                    offset=offset,
                    end_offset=end,
                    vertex_count=vertex_count,
                    triangle_count=triangle_count,
                    material_index=material_index,
                    batch_type=batch_type,
                    vertex_flags=vertex_flags,
                )

        self.lods = [
            StMeshLod(index, tuple(parsed_by_offset[offset] for offset in offsets))
            for index, offsets in enumerate(lod_block_offsets)
        ]
        max_material = max(
            block.material_index for lod in self.lods for block in lod.blocks
        )
        supplied_names = list(material_names or [])
        self.material_names = [
            supplied_names[index]
            if index < len(supplied_names) and supplied_names[index]
            else f"Material_{index}"
            for index in range(max_material + 1)
        ]
        self._build_mesh_adapter(data)
        return True

    def _decode_block(self, data: bytes, block: StMeshBlock):
        index_count = block.triangle_count * 3
        index_start = block.offset + 0x20
        position_start = (index_start + index_count + 3) & ~3
        packed_positions = np.frombuffer(
            data, dtype="<u2", count=block.vertex_count * 4, offset=position_start
        ).reshape(block.vertex_count, 4)
        minimum = np.asarray(self.bounds_min, dtype=np.float32)
        maximum = np.asarray(self.bounds_max, dtype=np.float32)
        positions = minimum + (
            packed_positions[:, :3].astype(np.float32) / np.float32(65535.0)
        ) * (maximum - minimum)
        packed_normals = np.frombuffer(
            data,
            dtype=np.uint8,
            count=block.vertex_count * 4,
            offset=position_start + block.vertex_count * 8,
        ).reshape(block.vertex_count, 4)
        normals = _octahedral_normals(packed_normals)
        packed_uv = (
            np.frombuffer(
                data,
                dtype="<f2",
                count=block.vertex_count * 2,
                offset=position_start + block.vertex_count * 12,
            )
            .astype(np.float64)
            .reshape(block.vertex_count, 2)
        )
        uv = packed_uv if np.isfinite(packed_uv).all() else None
        indices = np.frombuffer(
            data, dtype=np.uint8, count=index_count, offset=index_start
        ).astype(np.uint32)
        return positions, normals, uv, indices

    def _build_mesh_adapter(self, data: bytes) -> None:
        lod0 = self.lods[0]
        positions: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        uvs: list[np.ndarray] = []
        faces = array("I")
        submeshes = []
        vertex_base = 0
        all_uvs_valid = True
        for block in lod0.blocks:
            block_positions, block_normals, block_uvs, block_indices = (
                self._decode_block(data, block)
            )
            positions.append(block_positions)
            normals.append(block_normals)
            if block_uvs is None:
                all_uvs_valid = False
            else:
                uvs.append(block_uvs)
            face_start = len(faces)
            faces.extend(block_indices.tolist())
            submeshes.append(
                SimpleNamespace(
                    buffer_index=0,
                    verts_index_offset=vertex_base,
                    vert_count=block.vertex_count,
                    faces_index_offset=face_start,
                    indices_count=len(block_indices),
                    material_index=block.material_index,
                )
            )
            vertex_base += block.vertex_count

        payload = SimpleNamespace(
            vertex_bytes=b"",
            face_bytes=b"",
            buffer_headers=[],
            positions=array("f", np.concatenate(positions).astype("<f4").reshape(-1)),
            normals=array("f", np.concatenate(normals).astype("<f4").reshape(-1)),
            tangents=array("f"),
            normal_ws=array("B"),
            tangent_ws=array("B"),
            uv0=array("d"),
            uv1=array("d"),
            uv2=array("d"),
            colors=array("B"),
            faces=faces,
            integer_faces=None,
            skin_weights=None,
            extra_skin_weights=None,
        )
        if all_uvs_valid and uvs:
            payload.uv0 = array("d", np.concatenate(uvs).astype("<f8").reshape(-1))
        self.mesh_buffer = SimpleNamespace(
            buffer_payloads={0: payload},
            positions=payload.positions,
            normals=payload.normals,
            uv0=payload.uv0,
            colors=payload.colors,
            faces=payload.faces,
            integer_faces=None,
            streaming_buffer_count=0,
        )
        group = SimpleNamespace(
            group_id=0,
            vertex_count=vertex_base,
            face_count=len(faces),
            submeshes=submeshes,
        )
        self.meshes = [
            SimpleNamespace(
                lods=[SimpleNamespace(mesh_groups=[group])],
                material_count=len(self.material_names),
            )
        ]


__all__ = [
    "STMESH_MAGIC",
    "WOTS_STMESH_VERSION",
    "StMeshBlock",
    "StMeshFile",
    "StMeshLod",
    "StMeshParseError",
]
