from __future__ import annotations

"""Read the compact CLIP v89 payload embedded by WOTS MOT v973 files.

This is deliberately an adapter over REasy's standalone CLIP reader.  WOTS
uses the same property/key records, but has a smaller owner-specific header
and compact 0x28-byte node rows whose pointers are relative to the MOT object.
"""

from file_handlers.clip.enums import PropertyType, property_type_or_unknown
from file_handlers.clip.parser import ClipParser, ParsedClip
from file_handlers.clip.reader import ClipParserError, Reader
from file_handlers.clip.structures import (
    ActionKey,
    BoolKey,
    ClipHeader,
    Key,
    Node,
    NoHermiteKey,
)
from file_handlers.clip.value_adapters import key_payload_text

from ..errors import MotionParseError
from .model import (
    ClipInterpolation,
    ClipKey,
    ClipNode,
    ClipProperty,
    CompactMotClip,
    SpeedPoint,
)


class WotsCompactMotClipV89Parser:
    VERSION = 89
    HEADER_SIZE = 0x98
    NODE_SIZE = 0x28

    _POINTER_FIELDS = (
        "node_tbl_ptr",
        "property_tbl_ptr",
        "key_tbl_ptr",
        "bool_keys_offset",
        "action_keys_offset",
        "no_hermite_keys_offset",
        "speed_point_tbl_ptr",
        "interpolation_hermite_tbl_ptr",
        "interpolation_hermite3d_tbl_ptr",
        "user_data_asset_info_ptr",
        "c8_ptr",
        "c16_ptr",
        "oword_ptr",
        "data_ptr",
    )

    def parse(
        self,
        object_data: bytes,
        clip_offset: int,
        following_data_offset: int,
        *,
        label: str,
    ) -> CompactMotClip:
        reader = Reader(object_data)
        helper = ClipParser()
        try:
            header = self._read_header(reader, clip_offset)
            self._validate_sections(
                reader,
                header,
                clip_offset,
                following_data_offset,
            )
            nodes = self._read_nodes(reader, helper, header)
            properties = helper._read_properties(reader, header)
            main_keys = helper._read_key_table(
                reader, header.key_tbl_ptr, header.key_num, header.version
            )
            bool_keys = helper._read_bool_keys(reader, header)
            action_keys = helper._read_action_keys(reader, header)
            no_hermite_keys = helper._read_no_hermite_keys(reader, header)
            speed_points = helper._read_speed_points(reader, header)
            hermite_nodes = helper._read_hermite_nodes(reader, header)
            bezier3d_nodes = helper._read_bezier3d_nodes(reader, header)
            user_data_assets = helper._read_user_data_assets(reader, header)

            parsed = ParsedClip(
                header=header,
                root_node_offsets=[],
                root_nodes=[nodes[0]] if nodes else [],
                nodes_reorder_offsets=[],
                nodes_reorder_nodes=[],
                track_child_offsets=[],
                tracks=[],
                clip_infos=[],
                nodes=nodes,
                properties=properties,
                main_keys=main_keys,
                last_keys=[],
                bool_keys=bool_keys,
                action_keys=action_keys,
                no_hermite_keys=no_hermite_keys,
                speed_points=speed_points,
                hermite_nodes=hermite_nodes,
                bezier3d_nodes=bezier3d_nodes,
                user_data_assets=user_data_assets,
                owords=[],
            )
            self._attach_nodes(nodes, properties)
            helper._attach_property_ranges(parsed)
            helper._attach_interpolation_references(parsed)
            helper._decode_key_string_payloads(parsed, reader)
            return self._convert(parsed)
        except (ClipParserError, IndexError, ValueError) as exc:
            raise MotionParseError(
                f"{label}: compact CLIP v89 at 0x{clip_offset:X}: {exc}"
            ) from exc

    def _read_header(self, reader: Reader, offset: int) -> ClipHeader:
        reader._check(offset, self.HEADER_SIZE)
        if reader.u32(offset) != 0x50494C43:
            raise ClipParserError("expected CLIP magic")
        version = reader.u32(offset + 4)
        if version != self.VERSION:
            raise ClipParserError(f"unsupported compact CLIP version {version}")
        header = ClipHeader(
            magic=0x50494C43,
            version=version,
            total_frame=reader.f32(offset + 8),
            node_num=reader.u32(offset + 0xC),
            property_num=reader.u32(offset + 0x10),
            key_num=reader.u32(offset + 0x14),
            bool_key_num=reader.u32(offset + 0x18),
            action_key_num=reader.u32(offset + 0x1C),
            no_hermite_key_num=reader.u32(offset + 0x20),
        )
        if reader.u32(offset + 0x24):
            raise ClipParserError("reserved header word is nonzero")
        for index, name in enumerate(self._POINTER_FIELDS):
            setattr(header, name, reader.u64(offset + 0x28 + index * 8))
        return header

    def _validate_sections(
        self,
        reader: Reader,
        header: ClipHeader,
        clip_offset: int,
        following_data_offset: int,
    ) -> None:
        if not header.node_num:
            raise ClipParserError("compact CLIP has no root node")
        if header.node_tbl_ptr != clip_offset + self.HEADER_SIZE:
            raise ClipParserError("node table does not follow the compact header")
        pointers = [
            getattr(header, name)
            for name in self._POINTER_FIELDS
            if getattr(header, name)
        ]
        if pointers != sorted(pointers):
            raise ClipParserError("section pointers are not monotonic")
        if any(pointer < clip_offset or pointer > following_data_offset for pointer in pointers):
            raise ClipParserError("section pointer is outside its SequenceData payload")
        if header.property_tbl_ptr != header.node_tbl_ptr + header.node_num * self.NODE_SIZE:
            raise ClipParserError("property table does not follow compact nodes")
        reader._check(header.node_tbl_ptr, header.node_num * self.NODE_SIZE)
        reader._check(header.property_tbl_ptr, header.property_num * 0x38)

    def _read_nodes(
        self,
        reader: Reader,
        helper: ClipParser,
        header: ClipHeader,
    ) -> list[Node]:
        result: list[Node] = []
        for index in range(header.node_num):
            offset = header.node_tbl_ptr + index * self.NODE_SIZE
            node = Node(
                node_num=reader.u16(offset),
                property_num=reader.u16(offset + 2),
                name_hash=reader.u32(offset + 8),
                unicode_name_hash=reader.u32(offset + 0xC),
                name=helper._string(
                    reader,
                    header,
                    reader.u64(offset + 0x10),
                    True,
                ),
                child_offset=reader.u64(offset + 0x18),
                property_offset=reader.u64(offset + 0x20),
            )
            if reader.u32(offset + 4):
                raise ClipParserError(
                    f"node[{index}] reserved word at 0x{offset + 4:X} is nonzero"
                )
            result.append(node)
        return result

    @staticmethod
    def _attach_nodes(nodes, properties) -> None:
        owners = [0] * len(nodes)
        for index, node in enumerate(nodes):
            child_end = node.child_offset + node.node_num
            property_end = node.property_offset + node.property_num
            if child_end > len(nodes):
                raise ClipParserError(f"node[{index}] child range is out of bounds")
            if property_end > len(properties):
                raise ClipParserError(f"node[{index}] property range is out of bounds")
            node.child_nodes = nodes[node.child_offset:child_end]
            node.properties = properties[node.property_offset:property_end]
            for child_index in range(node.child_offset, child_end):
                owners[child_index] += 1
        if owners and (owners[0] or any(count != 1 for count in owners[1:])):
            raise ClipParserError("compact nodes do not form one rooted tree")

    def _convert(self, parsed: ParsedClip) -> CompactMotClip:
        if not parsed.nodes:
            raise ClipParserError("compact CLIP has no nodes")

        def convert_node(node: Node, *, root: bool = False) -> ClipNode:
            properties = [self._convert_property(prop) for prop in node.properties]
            children = [convert_node(child) for child in node.child_nodes]
            bounds = [
                (prop.start_frame, prop.end_frame)
                for prop in properties
            ]
            start = 0.0 if root or not bounds else min(value[0] for value in bounds)
            end = (
                parsed.header.total_frame
                if root or not bounds
                else max(value[1] for value in bounds)
            )
            return ClipNode(
                name=node.name,
                start_frame=start,
                end_frame=end,
                properties=properties,
                children=children,
            )

        return CompactMotClip(
            total_frame=parsed.header.total_frame,
            root=convert_node(parsed.nodes[0], root=True),
        )

    @staticmethod
    def _convert_property(prop) -> ClipProperty:
        converted = ClipProperty(
            name=prop.name,
            property_type=property_type_or_unknown(prop.property_type),
            start_frame=prop.begin_frame,
            end_frame=prop.end_frame,
            array_index=prop.array_index,
            enum_closed=bool(prop.is_enum_closed),
            set_after_end_frame=bool(prop.set_after_end_frame),
            restoration=bool(prop.is_restoration),
            set_delegate_enable=bool(prop.is_set_delegate_enable),
            prev_diff_frame_set=bool(prop.is_prev_diff_frame_set),
            next_diff_frame_set=bool(prop.is_next_diff_frame_set),
            prev_key_value_set=bool(prop.is_prev_key_value_set),
        )
        converted.children = [
            WotsCompactMotClipV89Parser._convert_property(child)
            for child in prop.child_properties
        ]
        converted.keys = [
            WotsCompactMotClipV89Parser._convert_key(prop, key)
            for key in (*prop.keys, *prop.extra_keys)
        ]
        if prop.last_key_ref is not None:
            converted.last_key = WotsCompactMotClipV89Parser._convert_key(
                prop, prop.last_key_ref
            )
        converted.speed_points = [
            SpeedPoint(
                frame=point.frame,
                rate=point.rate,
                interpolation=WotsCompactMotClipV89Parser._interpolation(
                    point.interpolation_type
                ),
            )
            for point in prop.speed_points_ref
        ]
        return converted

    @staticmethod
    def _convert_key(prop, key) -> ClipKey:
        if isinstance(key, BoolKey):
            interpolation = key.interpolation_type_to_next
            value = bool(key.bool_value)
            frame_span = key.range_v2_frame_span
        elif isinstance(key, ActionKey):
            interpolation = key.interpolation_type
            value = True
            frame_span = 0
        elif isinstance(key, NoHermiteKey):
            interpolation = key.interpolation_type_to_next
            value = WotsCompactMotClipV89Parser._typed_value(prop, key)
            frame_span = key.range_v2_frame_span
        elif isinstance(key, Key):
            interpolation = key.interpolation_type
            value = WotsCompactMotClipV89Parser._typed_value(prop, key)
            frame_span = key.frame_span
        else:
            raise ClipParserError(f"unsupported compact key type {type(key).__name__}")
        return ClipKey(
            frame=key.frame,
            rate=getattr(key, "rate", 0.0),
            interpolation=WotsCompactMotClipV89Parser._interpolation(interpolation),
            offset_frame=bool(getattr(key, "offset_frame_flag", 0)),
            value=value,
            curve=None,
        )

    @staticmethod
    def _interpolation(raw: int) -> ClipInterpolation:
        try:
            return ClipInterpolation(raw)
        except ValueError as exc:
            raise ClipParserError(f"unknown compact key interpolation {raw}") from exc

    @staticmethod
    def _typed_value(prop, key):
        text = key_payload_text(prop, key)
        kind = property_type_or_unknown(prop.property_type)
        if kind == PropertyType.BOOL:
            return text == "True"
        if kind in {
            PropertyType.S8,
            PropertyType.U8,
            PropertyType.S16,
            PropertyType.U16,
            PropertyType.S32,
            PropertyType.U32,
            PropertyType.S64,
            PropertyType.U64,
        }:
            return int(text)
        if kind in {PropertyType.F32, PropertyType.F64}:
            return float(text)
        return text
