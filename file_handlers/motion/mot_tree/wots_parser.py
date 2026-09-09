from __future__ import annotations

import struct

from utils.hash_util import murmur3_hash

from ..binary import ReadContext
from ..errors import MotionParseError
from .model import (
    MotTree,
    MotionIdRemap,
    TreeLink,
    TreeLinkType,
    TreeNode,
    TreeNodeType,
    TreeParameter,
    TreeParameterType,
    TreeTag,
)


class WotsMotTreeV21Parser:
    """Read the MotTree v21 layout used by WOTS 1.0.1.0.

    v21 removes parameter-name strings from the v4 node records. Parameters
    are identified by Murmur3 property hashes instead. Known motion reference
    properties are restored by name so the existing preview resolver can
    follow BankID/MotionID pairs; other properties remain visible under stable
    ``property_<hash>`` names.
    """

    VERSION = 21
    HEADER_SIZE = 0x58
    NODE_SIZE = 0x30
    PARAMETER_SIZE = 0x10
    LINK_SIZE = 0x28

    _PARAMETER_NAMES = {
        0x5CD4677B: "BankID",
        0x9F595148: "MotionID",
    }

    def parse(self, context: ReadContext, base: int, physical_end: int) -> MotTree:
        c = context.subcontext(
            base,
            physical_end,
            label=f"MotTree v21@0x{base:X}",
            object_base=base,
        )
        c.require(base, self.HEADER_SIZE, "MotTree v21 header")
        if c.u32(base) != self.VERSION or c.bytes(base + 4, 4) != b"mtre":
            raise MotionParseError(f"{c.label}: expected MotTree v21")

        node_stored = c.u64(base + 0x20, "MotTree node table")
        link_stored = c.u64(base + 0x28, "MotTree link table")
        name_stored = c.u64(base + 0x30, "MotTree name")
        reference_stored = c.u64(base + 0x40, "MotTree motion-reference table")
        node_count = c.u16(base + 0x48, "MotTree node count")
        link_count = c.u16(base + 0x4A, "MotTree link count")
        root_index = c.u16(base + 0x4C, "MotTree root index")
        reference_count = c.u16(base + 0x52, "MotTree motion-reference count")

        name, _ = c.utf16_z(base + name_stored, "MotTree name")
        node_table = base + node_stored
        c.require(node_table, node_count * self.NODE_SIZE, "MotTree node table")

        raw_nodes = []
        for index in range(node_count):
            record = node_table + index * self.NODE_SIZE
            raw_nodes.append(
                {
                    "name": c.u64(record, f"node[{index}] name"),
                    "tags": c.u64(record + 8, f"node[{index}] tag table"),
                    "tag_hashes": c.u64(record + 0x10, f"node[{index}] tag hashes"),
                    "params": c.u64(record + 0x18, f"node[{index}] parameters"),
                    "class_hash": c.u32(record + 0x20, f"node[{index}] class hash"),
                    "class_hash2": c.u32(record + 0x24, f"node[{index}] class hash 2"),
                    "name_hash": c.u32(record + 0x28, f"node[{index}] name hash"),
                    "tag_count": c.u8(record + 0x2C, f"node[{index}] tag count"),
                    "param_count": c.u8(record + 0x2D, f"node[{index}] parameter count"),
                    "node_type": c.u8(record + 0x2E, f"node[{index}] type"),
                    "flags": c.u8(record + 0x2F, f"node[{index}] flags"),
                }
            )

        # v21 stores class-name strings sequentially immediately after the
        # fixed node table instead of keeping one pointer per node.
        class_cursor = node_table + node_count * self.NODE_SIZE
        class_names = []
        for index in range(node_count):
            class_name, class_cursor = c.ascii_z(
                class_cursor, f"node[{index}] class name"
            )
            class_names.append(class_name)

        nodes = []
        for index, raw in enumerate(raw_nodes):
            node_name = None
            if raw["name"]:
                value, _ = c.utf16_z(base + raw["name"], f"node[{index}] name")
                if raw["name_hash"]:
                    actual_hash = murmur3_hash(value.encode("utf-16le"))
                    if actual_hash != raw["name_hash"]:
                        raise MotionParseError(
                            f"{c.label}: node[{index}] name hash mismatch"
                        )
                    node_name = value
            elif raw["name_hash"]:
                raise MotionParseError(
                    f"{c.label}: node[{index}] has a name hash without a name"
                )

            tags = self._parse_tags(c, base, index, raw)
            parameters = self._parse_parameters(c, base, index, raw)
            try:
                node_type = TreeNodeType(raw["node_type"])
            except ValueError as exc:
                raise MotionParseError(
                    f"{c.label}: node[{index}] has unknown type {raw['node_type']}"
                ) from exc
            nodes.append(
                TreeNode(
                    class_name=class_names[index],
                    name=node_name,
                    authored_id=index + 1,
                    node_type=node_type,
                    tags=tags,
                    parameters=parameters,
                )
            )

        links = self._parse_links(c, base, nodes, link_stored, link_count)
        references = []
        if reference_count:
            if not reference_stored:
                raise MotionParseError(
                    f"{c.label}: motion-reference count has no table"
                )
            table = base + reference_stored
            c.require(table, reference_count * 8, "MotTree motion references")
            for index in range(reference_count):
                record = table + index * 8
                references.append(
                    MotionIdRemap(
                        c.u32(record, f"motion reference[{index}] bank ID"),
                        c.u32(record + 4, f"motion reference[{index}] motion ID"),
                    )
                )
        elif reference_stored:
            raise MotionParseError(
                f"{c.label}: empty motion-reference table has a pointer"
            )

        if node_count and root_index >= node_count:
            raise MotionParseError(f"{c.label}: root node index is out of range")
        return MotTree(
            name,
            nodes,
            links,
            nodes[root_index] if nodes else None,
            references,
        )

    def _parse_tags(self, c, base, node_index, raw):
        count = raw["tag_count"]
        if not count:
            return []
        if not raw["tags"]:
            raise MotionParseError(
                f"{c.label}: node[{node_index}] tag count has no table"
            )
        table = base + raw["tags"]
        c.require(table, count * 8, f"node[{node_index}] tag table")
        if raw["tag_hashes"]:
            c.require(
                base + raw["tag_hashes"],
                count * 4,
                f"node[{node_index}] tag hash table",
            )
        result = []
        for index in range(count):
            stored = c.u64(table + index * 8, f"node[{node_index}] tag[{index}]")
            value, _ = c.utf16_z(
                base + stored, f"node[{node_index}] tag[{index}]"
            )
            result.append(TreeTag(value))
        return result

    def _parse_parameters(self, c, base, node_index, raw):
        count = raw["param_count"]
        if not count:
            return []
        if not raw["params"]:
            raise MotionParseError(
                f"{c.label}: node[{node_index}] parameter count has no table"
            )
        table = base + raw["params"]
        c.require(
            table,
            count * self.PARAMETER_SIZE,
            f"node[{node_index}] parameter table",
        )
        result = []
        for index in range(count):
            record = table + index * self.PARAMETER_SIZE
            raw_type = c.u32(record, f"node[{node_index}] parameter[{index}] type")
            property_hash = c.u32(
                record + 4, f"node[{node_index}] parameter[{index}] hash"
            )
            raw_value = c.u64(
                record + 8, f"node[{node_index}] parameter[{index}] value"
            )
            try:
                parameter_type = TreeParameterType(raw_type)
            except ValueError as exc:
                raise MotionParseError(
                    f"{c.label}: node[{node_index}] parameter[{index}] has "
                    f"unsupported type {raw_type} at 0x{record - base:X}"
                ) from exc
            value = self._parameter_value(
                c,
                base,
                parameter_type,
                raw_value,
                node_index,
                index,
            )
            result.append(
                TreeParameter(
                    self._PARAMETER_NAMES.get(
                        property_hash, f"property_{property_hash:08X}"
                    ),
                    parameter_type,
                    value,
                )
            )
        return result

    @staticmethod
    def _parameter_value(c, base, parameter_type, raw_value, node_index, index):
        what = f"node[{node_index}] parameter[{index}]"
        raw_bytes = struct.pack("<Q", raw_value)
        if parameter_type == TreeParameterType.BOOL:
            if raw_value not in (0, 1):
                raise MotionParseError(f"{c.label}: {what} has invalid Bool value")
            return bool(raw_value)
        if parameter_type == TreeParameterType.I32:
            if raw_value >> 32:
                raise MotionParseError(f"{c.label}: {what} I32 padding is nonzero")
            return struct.unpack("<i", raw_bytes[:4])[0]
        if parameter_type == TreeParameterType.U32:
            if raw_value >> 32:
                raise MotionParseError(f"{c.label}: {what} U32 padding is nonzero")
            return raw_value
        if parameter_type == TreeParameterType.F32:
            if raw_value >> 32:
                raise MotionParseError(f"{c.label}: {what} F32 padding is nonzero")
            return struct.unpack("<f", raw_bytes[:4])[0]
        if parameter_type == TreeParameterType.STR16:
            if not raw_value:
                return ""
            return c.utf16_z(base + raw_value, f"{what} string")[0]
        # Structured v21 parameter payloads are retained as their relative
        # data offsets. They are not needed for motion-reference resolution.
        return raw_value

    def _parse_links(self, c, base, nodes, link_stored, link_count):
        if not link_count:
            if link_stored:
                raise MotionParseError(f"{c.label}: empty link table has a pointer")
            return []
        if not link_stored:
            raise MotionParseError(f"{c.label}: link count has no table")
        table = base + link_stored
        c.require(table, link_count * self.LINK_SIZE, "MotTree link table")
        result = []
        for index in range(link_count):
            record = table + index * self.LINK_SIZE
            input_index = c.u32(record, f"link[{index}] input node")
            output_index = c.u32(record + 8, f"link[{index}] output node")
            if input_index >= len(nodes) or output_index >= len(nodes):
                raise MotionParseError(f"{c.label}: link[{index}] node is out of range")
            try:
                link_type = TreeLinkType(c.u32(record + 0x10, f"link[{index}] type"))
            except ValueError as exc:
                raise MotionParseError(
                    f"{c.label}: link[{index}] has an unknown type"
                ) from exc
            link = TreeLink(
                nodes[input_index],
                c.u32(record + 4, f"link[{index}] input pin"),
                nodes[output_index],
                c.u32(record + 0xC, f"link[{index}] output pin"),
                link_type,
            )
            output_guid = c.u64(record + 0x20, f"link[{index}] output GUID")
            if output_guid:
                link.output_guid = c.bytes(
                    base + output_guid, 16, f"link[{index}] output GUID"
                )
            result.append(link)
        return result
