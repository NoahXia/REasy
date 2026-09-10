from __future__ import annotations

import math
import struct

from utils.hash_util import murmur3_hash_utf16le

from .binary import ReadContext
from .errors import MotionParseError, MotionWriteError
from .mot.model import AnimationNode, Joint, KeyTrack, Motion, Skeleton, TrackFamily
from .mot_clip.wots_v89 import WotsCompactMotClipV89Parser
from .mot_list.model import EmbeddedPayload, MotList, MotionSlot, MotionSlotType
from .mot_tree.wots_parser import WotsMotTreeV21Parser
from .profiles import WOTS_PROFILE
from .sequence.model import SequenceCategory, SequenceData, SequenceTrack


class WotsMotParser:
    """Read the modern MOT v973 subset used by WOTS 1.0.1.0.

    Every pointer in this format is relative to the start of its MOT object,
    including MOT objects embedded in a MOTLIST.
    """

    VERSION = 973
    HEADER_SIZE = 0x7C
    BONE_SIZE = 0x50
    CLIP_SIZE = 0x0C
    TRACK_SIZE = 0x14

    def __init__(self, context: ReadContext):
        self.context = context
        self.clip_parser = WotsCompactMotClipV89Parser()

    def _mot_context(self, base: int, physical_end: int) -> ReadContext:
        c = self.context.subcontext(
            base,
            physical_end,
            label=f"MOT v973@0x{base:X}",
            object_base=base,
        )
        c.require(base, self.HEADER_SIZE, "MOT header")
        if (
            c.u32(base, "MOT version") != self.VERSION
            or c.bytes(base + 4, 4) != b"mot "
        ):
            raise MotionParseError(f"{c.label}: expected MOT v973 magic 'mot '")
        return c

    def parse_skeleton(self, base: int, physical_end: int) -> Skeleton | None:
        c = self._mot_context(base, physical_end)
        joint_count = c.u16(base + 0x70, "MOT joint count")
        table_pointer_pointer = c.u64(base + 0x10, "MOT skeleton indirection")
        if not table_pointer_pointer:
            return None
        indirection = base + table_pointer_pointer
        # Many embedded v973 motions store their aligned object size in this
        # field when the skeleton is inherited from another MOT in the list.
        if indirection + 0x10 > physical_end:
            return None
        c.require(indirection, 0x10, "MOT skeleton indirection")
        table_stored = c.u64(indirection, "MOT skeleton table")
        stored_count = c.u64(indirection + 8, "MOT skeleton count")
        if stored_count != joint_count:
            raise MotionParseError(
                f"{c.label}: skeleton count {stored_count} does not match header count {joint_count}"
            )
        table = base + table_stored
        c.require(table, joint_count * self.BONE_SIZE, "MOT skeleton records")
        joints: list[Joint] = []
        parent_pointers: list[int] = []
        for index in range(joint_count):
            record = table + index * self.BONE_SIZE
            name_stored = c.u64(record, f"joint[{index}] name")
            name, _ = c.utf16_z(base + name_stored, f"joint[{index}] name")
            parent_pointers.append(c.u64(record + 8, f"joint[{index}] parent"))
            translation = tuple(c.f32(record + 0x20 + axis * 4) for axis in range(3))
            rotation = tuple(c.f32(record + 0x30 + axis * 4) for axis in range(4))
            stored_index = c.u32(record + 0x40, f"joint[{index}] index")
            if stored_index != index:
                raise MotionParseError(
                    f"{c.label}: joint record 0x{record - base:X} has index {stored_index}, expected {index}"
                )
            stored_hash = c.u32(record + 0x44, f"joint[{index}] hash")
            if stored_hash != murmur3_hash_utf16le(name):
                raise MotionParseError(
                    f"{c.label}: joint {name!r} hash mismatch at 0x{record + 0x44 - base:X}"
                )
            joints.append(
                Joint(
                    name=name,
                    binding_hash=stored_hash,
                    translation=translation,
                    rotation=rotation,
                )
            )

        for index, stored in enumerate(parent_pointers):
            if not stored:
                continue
            delta = stored - table_stored
            if delta < 0 or delta % self.BONE_SIZE:
                raise MotionParseError(
                    f"{c.label}: joint[{index}] parent pointer 0x{stored:X} is not a skeleton record"
                )
            parent_index = delta // self.BONE_SIZE
            if parent_index >= len(joints) or parent_index == index:
                raise MotionParseError(
                    f"{c.label}: joint[{index}] parent index {parent_index} is out of range"
                )
            joints[index].parent = joints[parent_index]
            joints[parent_index].children.append(joints[index])
        return Skeleton(joints)

    def parse(
        self,
        base: int,
        physical_end: int,
        *,
        shared_skeleton: Skeleton | None = None,
    ) -> Motion:
        c = self._mot_context(base, physical_end)
        name_stored = c.u64(base + 0x58, "MOT name")
        name, _ = c.utf16_z(base + name_stored, "MOT name")
        end_frame = c.f32(base + 0x60, "MOT end frame")
        loop_time = c.f32(base + 0x64, "MOT loop time")
        raw_start = c.f32(base + 0x68, "MOT raw start frame")
        raw_end = c.f32(base + 0x6C, "MOT raw end frame")
        joint_count = c.u16(base + 0x70, "MOT joint count")
        clip_count = c.u16(base + 0x72, "MOT bone clip count")
        sequence_count = c.u8(base + 0x74, "MOT sequence count")
        # v973 moved the rate after the legacy u16 at 0x76.
        fps = c.u16(base + 0x78, "MOT frame rate")
        if fps not in (30, 60):
            raise MotionParseError(f"{c.label}: unsupported frame rate {fps} at 0x78")

        own_skeleton = self.parse_skeleton(base, physical_end)
        skeleton = own_skeleton or self._clone_skeleton(shared_skeleton)
        if skeleton is not None and joint_count and len(skeleton.joints) != joint_count:
            raise MotionParseError(
                f"{c.label}: source skeleton has {len(skeleton.joints)} joints, "
                f"expected {joint_count}"
            )
        if clip_count and skeleton is None:
            # Some WOTS MOTLIST entries intentionally omit the source skeleton.
            # Keep hash-bound root joints so the animation can still be inspected
            # and rebound to a manually loaded MESH without inventing topology.
            skeleton = Skeleton([])

        clip_stored = c.u64(base + 0x18, "MOT bone clip table")
        if bool(clip_stored) != bool(clip_count):
            raise MotionParseError(
                f"{c.label}: bone clip pointer/count presence mismatch"
            )
        nodes: list[AnimationNode] = []
        if clip_count:
            assert skeleton is not None
            table = base + clip_stored
            c.require(table, clip_count * self.CLIP_SIZE, "MOT bone clip headers")
            by_hash = {
                murmur3_hash_utf16le(joint.name): joint for joint in skeleton.joints
            }
            for index in range(clip_count):
                record = table + index * self.CLIP_SIZE
                bone_index = c.u16(record, f"bone clip[{index}] index")
                flags = c.u16(record + 2, f"bone clip[{index}] flags")
                bone_hash = c.u32(record + 4, f"bone clip[{index}] hash")
                tracks_stored = c.u32(record + 8, f"bone clip[{index}] tracks")
                transform_flags = flags & 0x7
                if not transform_flags:
                    raise MotionParseError(
                        f"{c.label}: bone clip flags 0x{flags:X} contain no transform tracks "
                        f"at 0x{record + 2 - base:X}"
                    )
                joint = by_hash.get(bone_hash)
                if joint is None:
                    # Skeleton-less embedded MOTs can target a different rig
                    # from the only source skeleton carried by their list.
                    # Preserve those channels as explicitly hash-bound roots
                    # so a manually loaded MESH can still bind them safely.
                    joint = Joint(
                        name=f"bone_{bone_hash:08X}",
                        binding_hash=bone_hash,
                    )
                    skeleton.joints.append(joint)
                    by_hash[bone_hash] = joint
                if bone_index >= joint_count:
                    raise MotionParseError(
                        f"{c.label}: bone clip[{index}] index {bone_index} is outside {joint_count} joints"
                    )
                tracks: dict[str, KeyTrack] = {}
                header = base + tracks_stored
                for bit, family, field in (
                    (1, TrackFamily.VECTOR3, "translation"),
                    (2, TrackFamily.QUATERNION, "rotation"),
                    (4, TrackFamily.VECTOR3, "scale"),
                ):
                    if transform_flags & bit:
                        tracks[field] = self._parse_track(
                            c, base, header, family, field
                        )
                        header += self.TRACK_SIZE
                nodes.append(
                    AnimationNode(
                        joint=joint,
                        translation=tracks.get("translation"),
                        rotation=tracks.get("rotation"),
                        scale=tracks.get("scale"),
                    )
                )

        character_path = None
        character_stored = c.u64(base + 0x38, "MOT character path")
        if character_stored:
            character_path, _ = c.utf16_z(
                base + character_stored,
                "MOT character/JMAP path",
            )

        sequences: list[SequenceData] = []
        sequence_stored = c.u64(base + 0x30, "MOT sequence table")
        if bool(sequence_stored) != bool(sequence_count):
            raise MotionParseError(
                f"{c.label}: sequence pointer/count presence mismatch"
            )
        if sequence_count:
            sequence_table = base + sequence_stored
            c.require(sequence_table, sequence_count * 8, "MOT sequence pointer table")
            object_data = bytes(c.data[base:physical_end])
            for index in range(sequence_count):
                stored = c.u64(
                    sequence_table + index * 8,
                    f"MOT sequence[{index}] pointer",
                )
                if not stored:
                    raise MotionParseError(
                        f"{c.label}: sequence[{index}] has a null pointer"
                    )
                sequences.append(
                    self._parse_sequence(
                        c,
                        base,
                        physical_end,
                        base + stored,
                        index,
                        object_data,
                    )
                )

        return Motion(
            name=name,
            end_frame=end_frame,
            looping=loop_time == 0.0,
            raw_start_frame=raw_start,
            raw_end_frame=raw_end,
            skeleton=skeleton,
            animation_nodes=nodes,
            sequences=sequences,
            character_path=character_path,
            source_joint_count=joint_count,
        )

    def _parse_sequence(
        self,
        c: ReadContext,
        base: int,
        physical_end: int,
        offset: int,
        index: int,
        object_data: bytes,
    ) -> SequenceData:
        c.require(offset, 0x40, f"SequenceData[{index}]")
        clip_stored = c.u64(offset + 8, f"SequenceData[{index}] CLIP pointer")
        tracks_stored = c.u64(offset + 0x10, f"SequenceData[{index}] tracks pointer")
        attributes = c.u32(offset + 0x18, f"SequenceData[{index}] attributes")
        track_count = c.u32(offset + 0x1C, f"SequenceData[{index}] track count")
        use_flags = c.u32(offset + 0x20, f"SequenceData[{index}] use flags")
        raw_category = c.u32(offset + 0x24, f"SequenceData[{index}] category")
        if attributes:
            raise MotionParseError(
                f"{c.label}: SequenceData[{index}] attributes 0x{attributes:X} "
                f"at 0x{offset + 0x18 - base:X} are unsupported"
            )
        if use_flags != 1:
            raise MotionParseError(
                f"{c.label}: SequenceData[{index}] use flags 0x{use_flags:X} "
                f"at 0x{offset + 0x20 - base:X} are unsupported"
            )
        try:
            category = SequenceCategory(raw_category)
        except ValueError as exc:
            raise MotionParseError(
                f"{c.label}: SequenceData[{index}] category {raw_category} "
                f"at 0x{offset + 0x24 - base:X} is unsupported"
            ) from exc
        clip_offset = base + clip_stored
        tracks_offset = base + tracks_stored
        if clip_offset != offset + 0x40:
            raise MotionParseError(
                f"{c.label}: SequenceData[{index}] compact CLIP does not begin "
                f"at wrapper +0x40 (0x{clip_offset - base:X})"
            )
        c.require(tracks_offset, track_count * 0x1C, f"SequenceData[{index}] tracks")
        clip = self.clip_parser.parse(
            object_data,
            clip_offset - base,
            tracks_offset - base,
            label=f"{c.label} SequenceData[{index}]",
        )
        tracks = [
            SequenceTrack(
                c.u32(tracks_offset + row * 0x1C, "TracksData authored ID"),
                c.u32(tracks_offset + row * 0x1C + 4, "TracksData filter page 0"),
                c.u32(tracks_offset + row * 0x1C + 8, "TracksData filter page 1"),
                c.u32(tracks_offset + row * 0x1C + 0xC, "TracksData filter page 2"),
            )
            for row in range(track_count)
        ]
        child_count = len(clip.root.children)
        if track_count < child_count:
            raise MotionParseError(
                f"{c.label}: SequenceData[{index}] has {track_count} track metadata "
                f"rows for {child_count} timeline tracks"
            )
        return SequenceData(category, clip, tracks[:child_count])

    @staticmethod
    def _clone_skeleton(source: Skeleton | None) -> Skeleton | None:
        if source is None:
            return None
        clones = [
            Joint(
                name=joint.name,
                binding_hash=joint.binding_hash,
                translation=joint.translation,
                rotation=joint.rotation,
                joint_map_extra_type=joint.joint_map_extra_type,
            )
            for joint in source.joints
        ]
        indices = {id(joint): index for index, joint in enumerate(source.joints)}
        for index, joint in enumerate(source.joints):
            if joint.parent is not None and id(joint.parent) in indices:
                parent = clones[indices[id(joint.parent)]]
                clones[index].parent = parent
                parent.children.append(clones[index])
        return Skeleton(clones)

    def _parse_track(
        self,
        c: ReadContext,
        base: int,
        header: int,
        family: TrackFamily,
        field: str,
    ) -> KeyTrack:
        c.require(header, self.TRACK_SIZE, f"{field} track header")
        flags = c.u32(header, f"{field} track flags")
        count = c.u32(header + 4, f"{field} key count")
        frame_stored = c.u32(header + 8, f"{field} frame table")
        data_stored = c.u32(header + 0xC, f"{field} value table")
        unpack_stored = c.u32(header + 0x10, f"{field} unpack table")
        if not count:
            raise MotionParseError(
                f"{c.label}: zero-key {field} track at 0x{header - base:X}"
            )
        selector = flags >> 20
        if selector == 5:
            frame_kind, frame_width = "u32", 4
        elif selector == 2:
            frame_kind, frame_width = "u8", 1
        elif selector in (0, 4):
            frame_kind, frame_width = "u16", 2
        else:
            raise MotionParseError(
                f"{c.label}: unsupported frame-index type 0x{selector:X} "
                f"at track 0x{header - base:X}"
            )
        if frame_stored:
            frame_offset = base + frame_stored
            c.require(frame_offset, count * frame_width, f"{field} key frames")
            frames = [
                c.unpack(frame_kind, frame_offset + i * frame_width)
                for i in range(count)
            ]
        else:
            frames = [0] * count

        unpack = (0.0,) * 8
        if unpack_stored:
            unpack_offset = base + unpack_stored
            c.require(unpack_offset, 0x20, f"{field} unpack parameters")
            unpack = tuple(c.f32(unpack_offset + i * 4) for i in range(8))
        data_offset = base + data_stored
        values = self._decode_values(
            c, data_offset, count, family, flags & 0xFF000, unpack, field, header - base
        )
        if field == "scale":
            values = [
                tuple(component / 100.0 for component in value) for value in values
            ]
        return KeyTrack(family=family, frames=frames, values=values)

    @staticmethod
    def _be(c: ReadContext, offset: int, width: int, what: str) -> int:
        return int.from_bytes(c.bytes(offset, width, what), "big")

    def _decode_values(
        self, c, offset, count, family, compression, unpack, field, track_offset
    ):
        maximum = unpack[:4]
        minimum = unpack[4:]
        values = []

        def packed3(value: int, bits: int):
            limit = (1 << bits) - 1
            return tuple(
                ((value >> (bits * axis)) & limit) / limit for axis in range(3)
            )

        def quat(x, y, z):
            return (x, y, z, math.sqrt(max(0.0, 1.0 - x * x - y * y - z * z)))

        cursor = offset
        for _ in range(count):
            if family is TrackFamily.VECTOR3:
                if compression == 0x00000:
                    value = tuple(c.f32(cursor + i * 4) for i in range(3))
                    cursor += 12
                elif compression == 0x20000:
                    raw = packed3(c.u16(cursor), 5)
                    cursor += 2
                    value = (
                        maximum[0] * raw[0] + maximum[3],
                        maximum[1] * raw[1] + minimum[0],
                        maximum[2] * raw[2] + minimum[1],
                    )
                elif compression in (0x21000, 0x22000, 0x23000, 0x24000):
                    raw = c.u16(cursor) / 0xFFFF
                    cursor += 2
                    if compression == 0x21000:
                        value = (maximum[0] * raw + maximum[1], maximum[2], maximum[3])
                    elif compression == 0x22000:
                        value = (maximum[1], maximum[0] * raw + maximum[2], maximum[3])
                    elif compression == 0x23000:
                        value = (maximum[1], maximum[2], maximum[0] * raw + maximum[3])
                    else:
                        value = (maximum[0] * raw + maximum[3],) * 3
                elif compression in (0x25000, 0x26000, 0x27000):
                    a, b = c.u8(cursor) / 255.0, c.u8(cursor + 1) / 255.0
                    cursor += 2
                    if compression == 0x25000:
                        value = (
                            maximum[0] * a + maximum[2],
                            maximum[1] * b + maximum[3],
                            minimum[0],
                        )
                    elif compression == 0x26000:
                        value = (
                            maximum[0] * a + maximum[2],
                            maximum[3],
                            maximum[1] * b + minimum[0],
                        )
                    else:
                        value = (
                            maximum[2],
                            maximum[0] * a + maximum[3],
                            maximum[1] * b + minimum[0],
                        )
                elif compression == 0x30000:
                    value = (
                        maximum[0] * c.u8(cursor) / 255.0 + maximum[3],
                        maximum[1] * c.u8(cursor + 1) / 255.0 + minimum[0],
                        maximum[2] * c.u8(cursor + 2) / 255.0 + minimum[1],
                    )
                    cursor += 3
                elif compression in (0x31000, 0x32000, 0x33000):
                    raw = self._be(c, cursor, 3, field) / 0xFFFFFF
                    cursor += 3
                    if compression == 0x31000:
                        value = (maximum[0] * raw + maximum[1], maximum[2], maximum[3])
                    elif compression == 0x32000:
                        value = (maximum[1], maximum[0] * raw + maximum[2], maximum[3])
                    else:
                        value = (maximum[1], maximum[2], maximum[0] * raw + maximum[3])
                elif compression in (0x35000, 0x36000, 0x37000):
                    raw = self._be(c, cursor, 3, field)
                    cursor += 3
                    a, b = (raw & 0xFFF) / 0xFFF, ((raw >> 12) & 0xFFF) / 0xFFF
                    if compression == 0x35000:
                        value = (
                            maximum[2],
                            maximum[0] * a + maximum[3],
                            maximum[1] * b + minimum[0],
                        )
                    elif compression == 0x36000:
                        value = (
                            maximum[0] * a + maximum[2],
                            maximum[3],
                            maximum[1] * b + minimum[0],
                        )
                    else:
                        value = (
                            maximum[0] * a + maximum[2],
                            maximum[1] * b + maximum[3],
                            minimum[0],
                        )
                elif compression == 0x40000:
                    raw = packed3(c.u32(cursor), 10)
                    cursor += 4
                    value = (
                        maximum[0] * raw[0] + maximum[3],
                        maximum[1] * raw[1] + minimum[0],
                        maximum[2] * raw[2] + minimum[1],
                    )
                elif compression in (0x41000, 0x42000, 0x43000, 0x44000):
                    raw = c.f32(cursor)
                    cursor += 4
                    if compression == 0x41000:
                        value = (raw, maximum[1], maximum[2])
                    elif compression == 0x42000:
                        value = (maximum[0], raw, maximum[2])
                    elif compression == 0x43000:
                        value = (maximum[0], maximum[1], raw)
                    else:
                        value = (raw, raw, raw)
                elif compression in (0x45000, 0x46000, 0x47000):
                    a, b = c.u16(cursor) / 0xFFFF, c.u16(cursor + 2) / 0xFFFF
                    cursor += 4
                    if compression == 0x45000:
                        value = (
                            maximum[0] * a + maximum[2],
                            maximum[1] * b + maximum[3],
                            minimum[0],
                        )
                    elif compression == 0x46000:
                        value = (
                            maximum[0] * a + maximum[2],
                            maximum[3],
                            maximum[1] * b + minimum[0],
                        )
                    else:
                        value = (
                            maximum[2],
                            maximum[0] * a + maximum[3],
                            maximum[1] * b + minimum[0],
                        )
                elif compression in (
                    0x50000,
                    0x55000,
                    0x56000,
                    0x57000,
                    0x60000,
                    0x65000,
                    0x66000,
                    0x67000,
                    0x70000,
                ):
                    width = {
                        0x50000: 5,
                        0x55000: 5,
                        0x56000: 5,
                        0x57000: 5,
                        0x60000: 6,
                        0x65000: 6,
                        0x66000: 6,
                        0x67000: 6,
                        0x70000: 7,
                    }[compression]
                    raw = self._be(c, cursor, width, field)
                    cursor += width
                    if compression == 0x50000:
                        q = 0x1FFF
                        a, b, d = tuple(((raw >> (13 * i)) & q) / q for i in range(3))
                        value = (
                            maximum[0] * a + maximum[3],
                            maximum[1] * b + minimum[0],
                            maximum[2] * d + minimum[1],
                        )
                    elif compression in (0x55000, 0x56000, 0x57000):
                        q = 0xFFFFF
                        a, b = (raw & q) / q, ((raw >> 20) & q) / q
                        if compression == 0x55000:
                            value = (
                                maximum[0] * a + maximum[2],
                                maximum[1] * b + maximum[3],
                                minimum[0],
                            )
                        elif compression == 0x56000:
                            value = (
                                maximum[2],
                                maximum[0] * a + maximum[3],
                                maximum[1] * b + minimum[0],
                            )
                        else:
                            value = (
                                maximum[1] * b + maximum[2],
                                maximum[3],
                                maximum[0] * a + minimum[0],
                            )
                    elif compression == 0x60000:
                        q = 0xFFFF
                        a, b, d = tuple(((raw >> (16 * i)) & q) / q for i in range(3))
                        value = (
                            maximum[0] * a + maximum[3],
                            maximum[1] * b + minimum[0],
                            maximum[2] * d + minimum[1],
                        )
                    elif compression in (0x65000, 0x66000, 0x67000):
                        q = 0xFFFFFF
                        a, b = (raw & q) / q, ((raw >> 24) & q) / q
                        if compression == 0x65000:
                            value = (
                                maximum[0] * a + maximum[2],
                                maximum[1] * b + maximum[3],
                                minimum[0],
                            )
                        elif compression == 0x66000:
                            value = (
                                maximum[2],
                                maximum[0] * a + maximum[3],
                                maximum[1] * b + minimum[0],
                            )
                        else:
                            value = (
                                maximum[1] * b + maximum[2],
                                maximum[3],
                                maximum[0] * a + minimum[0],
                            )
                    else:
                        q = 0x3FFFF
                        a, b, d = tuple(((raw >> (18 * i)) & q) / q for i in range(3))
                        value = (
                            maximum[0] * a + maximum[3],
                            maximum[1] * b + minimum[0],
                            maximum[2] * d + minimum[1],
                        )
                elif compression == 0x80000:
                    raw = packed3(c.u64(cursor), 21)
                    cursor += 8
                    value = (
                        maximum[0] * raw[0] + maximum[3],
                        maximum[1] * raw[1] + minimum[0],
                        maximum[2] * raw[2] + minimum[1],
                    )
                elif compression in (0x85000, 0x86000, 0x87000):
                    a, b = c.f32(cursor), c.f32(cursor + 4)
                    cursor += 8
                    if compression == 0x85000:
                        value = (a, b, maximum[2])
                    elif compression == 0x86000:
                        value = (maximum[0], a, b)
                    else:
                        value = (b, maximum[1], a)
                else:
                    raise MotionParseError(
                        f"{c.label}: unsupported {field} compression 0x{compression:05X} at track 0x{track_offset:X}"
                    )
            else:
                if compression == 0x00000:
                    value = tuple(c.f32(cursor + i * 4) for i in range(4))
                    cursor += 16
                elif compression in (
                    0x20000,
                    0x30000,
                    0x40000,
                    0x50000,
                    0x60000,
                    0x70000,
                    0x80000,
                ):
                    widths = {
                        0x20000: 2,
                        0x30000: 3,
                        0x40000: 4,
                        0x50000: 5,
                        0x60000: 6,
                        0x70000: 7,
                        0x80000: 8,
                    }
                    width = widths[compression]
                    if compression in (0x20000, 0x40000, 0x80000):
                        raw = packed3(
                            c.unpack({2: "u16", 4: "u32", 8: "u64"}[width], cursor),
                            {2: 5, 4: 10, 8: 21}[width],
                        )
                    elif compression == 0x30000:
                        raw = tuple(c.u8(cursor + i) / 255.0 for i in range(3))
                    else:
                        bits = {5: 13, 6: 16, 7: 18}[width]
                        integer = self._be(c, cursor, width, field)
                        mask = (1 << bits) - 1
                        raw = tuple(
                            ((integer >> (bits * i)) & mask) / mask for i in range(3)
                        )
                    cursor += width
                    value = quat(*(maximum[i] * raw[i] + minimum[i] for i in range(3)))
                elif compression in (0x21000, 0x22000, 0x23000):
                    axis = (compression - 0x21000) // 0x1000
                    raw = c.u16(cursor) / 0xFFFF
                    cursor += 2
                    xyz = [0.0, 0.0, 0.0]
                    xyz[axis] = maximum[0] * raw + maximum[1]
                    value = quat(*xyz)
                elif compression in (0x31000, 0x32000, 0x33000):
                    axis = (compression - 0x31000) // 0x1000
                    raw = self._be(c, cursor, 3, field) / 0xFFFFFF
                    cursor += 3
                    xyz = [0.0, 0.0, 0.0]
                    xyz[axis] = maximum[0] * raw + maximum[1]
                    value = quat(*xyz)
                elif compression in (0x41000, 0x42000, 0x43000):
                    axis = (compression - 0x41000) // 0x1000
                    xyz = [0.0, 0.0, 0.0]
                    xyz[axis] = c.f32(cursor)
                    cursor += 4
                    value = quat(*xyz)
                elif compression in (0xB0000, 0xC0000):
                    xyz = tuple(c.f32(cursor + i * 4) for i in range(3))
                    cursor += 12
                    value = quat(*xyz)
                else:
                    raise MotionParseError(
                        f"{c.label}: unsupported rotation compression 0x{compression:05X} at track 0x{track_offset:X}"
                    )
            values.append(value)
        return values


class WotsMotListParser:
    VERSION = 1036
    HEADER_SIZE = 0x3C
    ID_ROW_SIZE = 72

    def parse(self, data, *, label: str) -> MotList:
        c = ReadContext.from_bytes(data, label=label)
        c.require(0, self.HEADER_SIZE, "MOTLIST header")
        if c.u32(0) != self.VERSION or c.bytes(4, 4) != b"mlst":
            raise MotionParseError(f"{label}: expected MOTLIST v1036 magic 'mlst'")
        pointers_offset = c.u64(0x10, "MOTLIST pointer table")
        ids_offset = c.u64(0x18, "MOTLIST motion-ID table")
        name_offset = c.u64(0x20, "MOTLIST name")
        count = c.u32(0x38, "MOTLIST entry count")
        if count > len(c.data) // 8:
            raise MotionParseError(f"{label}: impossible MOTLIST entry count {count}")
        name, _ = c.utf16_z(name_offset, "MOTLIST name")
        c.require(pointers_offset, count * 8, "MOTLIST pointer table")
        c.require(ids_offset, count * self.ID_ROW_SIZE, "MOTLIST motion-ID table")
        pointers = [
            c.u64(pointers_offset + i * 8, f"MOTLIST pointer[{i}]")
            for i in range(count)
        ]
        ids = [
            c.u16(ids_offset + i * self.ID_ROW_SIZE + 8, f"MOTLIST motion ID[{i}]")
            for i in range(count)
        ]
        nonzero = sorted(set(pointer for pointer in pointers if pointer))
        boundaries = nonzero[1:] + [ids_offset]
        ends = dict(zip(nonzero, boundaries))
        parser = WotsMotParser(c)
        tree_parser = WotsMotTreeV21Parser()
        skeletons_by_count: dict[int, Skeleton] = {}
        for pointer in nonzero:
            if c.bytes(pointer + 4, 4, "MOTLIST payload magic") != b"mot ":
                continue
            skeleton = parser.parse_skeleton(pointer, ends[pointer])
            if skeleton is not None:
                skeletons_by_count.setdefault(len(skeleton.joints), skeleton)
        cache: dict[int, EmbeddedPayload] = {}
        slots = []
        diagnostics: list[str] = []
        for index, (pointer, motion_id) in enumerate(zip(pointers, ids)):
            payload = None
            slot_type = MotionSlotType.MOT
            if pointer:
                if pointer >= ids_offset:
                    raise MotionParseError(
                        f"{label}: MOT pointer[{index}] 0x{pointer:X} overlaps the ID table"
                    )
                magic = c.bytes(pointer + 4, 4, f"MOTLIST payload[{index}] magic")
                version = c.u32(pointer, f"MOTLIST payload[{index}] version")
                external_path = None
                if version == 1 and magic[:2] != b"\0\0":
                    external_path, _ = c.utf16_z(
                        pointer + 4, f"MOTLIST external MOT path[{index}]"
                    )
                    external_path = "natives/stm/" + external_path.replace(
                        "\\", "/"
                    ).lstrip("/")
                elif magic == b"mtre":
                    slot_type = MotionSlotType.MOT_TREE
                    if version != tree_parser.VERSION:
                        raise MotionParseError(
                            f"{label}: unsupported MotTree v{version} at 0x{pointer:X}"
                        )
                    if pointer not in cache:
                        cache[pointer] = EmbeddedPayload(
                            tree_parser.parse(c, pointer, ends[pointer])
                        )
                    payload = cache[pointer]
                elif magic != b"mot ":
                    raise MotionParseError(
                        f"{label}: unsupported payload {magic!r} v{version} at 0x{pointer:X}"
                    )
                elif pointer not in cache:
                    joint_count = c.u16(pointer + 0x70, f"MOT[{index}] joint count")
                    cache[pointer] = EmbeddedPayload(
                        parser.parse(
                            pointer,
                            ends[pointer],
                            shared_skeleton=skeletons_by_count.get(joint_count),
                        )
                    )
                if magic == b"mot ":
                    payload = cache[pointer]
            else:
                external_path = None
            slots.append(
                MotionSlot(
                    motion_id,
                    slot_type,
                    payload,
                    external_path=external_path,
                )
            )
        return MotList(name=name, slots=slots, diagnostics=diagnostics)


class WotsMotionFormatCodec:
    profile = WOTS_PROFILE
    supports_writing = False

    def matches(self, data) -> bool:
        return (
            len(data) >= 8
            and struct.unpack_from("<I", data)[0] == 1036
            and bytes(data[4:8]) == b"mlst"
        )

    def parse(self, data, *, label: str) -> MotList:
        return WotsMotListParser().parse(data, label=label)

    def parse_mot(self, data, *, label: str = "MOT") -> Motion:
        c = ReadContext.from_bytes(data, label=label)
        return WotsMotParser(c).parse(0, len(c.data))

    def parse_skeletons(self, data, *, label: str = "MOTLIST") -> tuple[Skeleton, ...]:
        """Scan a v1036 list for embedded skeleton owners without decoding tracks."""
        c = ReadContext.from_bytes(data, label=label)
        c.require(0, WotsMotListParser.HEADER_SIZE, "MOTLIST header")
        if c.u32(0) != self.profile.motlist.version or c.bytes(4, 4) != b"mlst":
            raise MotionParseError(f"{label}: expected MOTLIST v1036 magic 'mlst'")
        pointers_offset = c.u64(0x10, "MOTLIST pointer table")
        ids_offset = c.u64(0x18, "MOTLIST motion-ID table")
        count = c.u32(0x38, "MOTLIST entry count")
        if count > len(c.data) // 8:
            raise MotionParseError(f"{label}: impossible MOTLIST entry count {count}")
        c.require(pointers_offset, count * 8, "MOTLIST pointer table")
        pointers = sorted(
            {
                c.u64(pointers_offset + index * 8, f"MOTLIST pointer[{index}]")
                for index in range(count)
            }
            - {0}
        )
        boundaries = pointers[1:] + [ids_offset]
        parser = WotsMotParser(c)
        result = []
        for pointer, physical_end in zip(pointers, boundaries):
            if c.bytes(pointer + 4, 4, "MOTLIST payload magic") != b"mot ":
                continue
            skeleton = parser.parse_skeleton(pointer, physical_end)
            if skeleton is not None:
                result.append(skeleton)
        return tuple(result)

    def write(self, model: MotList) -> bytes:
        raise MotionWriteError("WOTS MOTLIST v1036 is read-only")


WOTS_MOTION_FORMAT_CODEC = WotsMotionFormatCodec()
