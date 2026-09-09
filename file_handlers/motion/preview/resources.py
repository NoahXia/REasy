from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from utils.resource_file_utils import (
    ResourceDataLoader,
    ResourceResolutionContext,
    normalize_resource_path,
    resource_context_for_app,
    resource_path_key,
)

from ..format_codec import MotionFormatCodec
from ..motlist_file import MotListFile
from ..mot_list.model import MotionSlotType
from .catalog_reader import MotionListCatalogDocument, MotionListCatalogReader
from .resolution import (
    MotionListDocument,
    MotionResolutionDiagnostic,
    PreviewMotionEntry,
    PreviewMotionOrigin,
)


def _resource_path(path: str) -> str:
    normalized = normalize_resource_path(path)
    lowered = normalized.casefold()
    marker = lowered.find("natives/")
    return normalized[marker:] if marker >= 0 else normalized


class MotionListResourceStore:
    """Load related MOTLISTs from disk, project roots, or configured PAKs."""

    def __init__(
        self,
        codec: MotionFormatCodec,
        *,
        app=None,
        anchor_path: str = "",
        selection_parent=None,
        resource_context: ResourceResolutionContext | None = None,
        resource_data_loader: ResourceDataLoader | None = None,
        catalog_reader: MotionListCatalogReader | None = None,
    ):
        self.codec = codec
        self.anchor_path = anchor_path
        self.selection_parent = selection_parent
        self.resource_context = resource_context or resource_context_for_app(app)
        self._resource_data_loader = resource_data_loader
        self._catalog_reader = catalog_reader
        self.errors: list[str] = []
        self._documents: dict[str, MotionListDocument] = {}
        self._catalogs: dict[str, MotionListCatalogDocument] = {}
        self._resource_data: dict[str, tuple[str, bytes]] = {}

    def parse(self, path: str, data: bytes) -> MotionListDocument:
        facade = MotListFile(self.codec)
        facade.read(data, label=path)
        document = MotionListDocument(path, facade.model)
        self._documents[resource_path_key(path)] = document
        return document

    def load(self, path: str, *, allow_selection_dialog: bool = False) -> MotionListDocument | None:
        cached = self._documents.get(resource_path_key(path))
        if cached is not None:
            return cached
        hit = self.resource_data(path, allow_selection_dialog=allow_selection_dialog)
        if hit is None:
            return None
        resolved_path, data = hit
        try:
            document = self.parse(resolved_path, data)
        except Exception as exc:
            self.errors.append(f"could not parse related MOTLIST {resolved_path!r}: {exc}")
            return None
        self._documents[resource_path_key(path)] = document
        return document

    def load_motion(self, path: str):
        parser = getattr(self.codec, "parse_mot", None)
        if parser is None:
            return None
        hit = self.resource_data(path)
        if hit is None:
            return None
        resolved_path, data = hit
        try:
            return parser(data, label=resolved_path)
        except (OSError, ValueError) as exc:
            self.errors.append(f"could not parse external MOT {resolved_path!r}: {exc}")
            return None

    def motion_entries(
        self,
        path: str,
    ) -> tuple[tuple[PreviewMotionEntry, ...], tuple[MotionResolutionDiagnostic, ...]] | None:
        """Resolve preview entries without decoding complete embedded MOTs."""

        if self._catalog_reader is None:
            return None
        diagnostics: list[MotionResolutionDiagnostic] = []
        document = self._load_catalog(path)
        if document is None:
            return None
        entries = self._effective_entries(document, (), diagnostics)
        return tuple(entries.values()), tuple(diagnostics)

    def _load_catalog(self, path: str) -> MotionListCatalogDocument | None:
        key = resource_path_key(path)
        cached = self._catalogs.get(key)
        if cached is not None:
            return cached
        hit = self.resource_data(path)
        if hit is None:
            return None
        resolved_path, data = hit
        try:
            document = self._catalog_reader.parse(resolved_path, data)
        except (OSError, ValueError) as exc:
            self.errors.append(
                f"could not read related MOTLIST catalog {resolved_path!r}: {exc}"
            )
            return None
        self._catalogs[key] = document
        self._catalogs[resource_path_key(resolved_path)] = document
        return document

    def _effective_entries(
        self,
        document: MotionListCatalogDocument,
        active_paths: tuple[str, ...],
        diagnostics: list[MotionResolutionDiagnostic],
    ) -> dict[int, PreviewMotionEntry]:
        identity = resource_path_key(document.path)
        if identity in active_paths:
            diagnostics.append(MotionResolutionDiagnostic(
                "base_list_cycle",
                f"base MOTLIST cycle reaches {document.path!r}",
            ))
            return {}
        base_entries: dict[int, PreviewMotionEntry] = {}
        if document.base_motion_list_path:
            base = self._load_catalog(document.base_motion_list_path)
            if base is None:
                diagnostics.append(MotionResolutionDiagnostic(
                    "missing_base_list",
                    f"base MOTLIST {document.base_motion_list_path!r} could not be loaded for {document.path!r}",
                ))
            else:
                base_entries = self._effective_entries(
                    base,
                    (*active_paths, identity),
                    diagnostics,
                )

        result = {}
        for slot in document.slots:
            if slot.slot_type != MotionSlotType.MOT:
                continue
            if slot.motion is not None:
                result[slot.motion_id] = PreviewMotionEntry(
                    slot.motion_id,
                    slot.motion,
                    PreviewMotionOrigin.EMBEDDED,
                    document.path,
                    document.name,
                    slot.slot_index,
                    (document.path,),
                )
                continue
            inherited = base_entries.get(slot.motion_id)
            if inherited is None:
                diagnostics.append(MotionResolutionDiagnostic(
                    "missing_inherited_motion",
                    f"slot ID {slot.motion_id} in {document.path!r} has no matching base MOT",
                ))
                continue
            result[slot.motion_id] = replace(
                inherited,
                motion_id=slot.motion_id,
                origin=PreviewMotionOrigin.INHERITED,
                slot_index=slot.slot_index,
                inheritance_chain=(document.path, *inherited.inheritance_chain),
            )
        return result

    def resource_data(
        self,
        path: str,
        *,
        allow_selection_dialog: bool = False,
    ) -> tuple[str, bytes] | None:
        key = resource_path_key(path)
        cached = self._resource_data.get(key)
        if cached is not None:
            return cached
        hit = (
            self._resource_data_loader(path)
            if self._resource_data_loader is not None
            else None
        )
        if hit is None:
            hit = self._filesystem_hit(path)
        if hit is None:
            hit = (
                self.resource_context.resolve(
                    path,
                    self.selection_parent,
                    allow_selection_dialog=allow_selection_dialog,
                )
                if self.resource_context is not None
                else None
            )
        if hit is not None:
            self._resource_data[key] = hit
            self._resource_data[resource_path_key(hit[0])] = hit
        return hit

    def find_bank_motion_lists(self, bank_ids: set[int]) -> dict[int, str]:
        items = self.nearby_motion_bank_items()
        return {bank_id: items[bank_id] for bank_id in bank_ids if bank_id in items}

    def nearby_motion_list_paths(self) -> tuple[str, ...]:
        """Return every MOTLIST registered by the nearest owning MOTBANK."""
        return tuple(dict.fromkeys(self.nearby_motion_bank_items().values()))

    def nearby_motion_bank_items(self) -> dict[int, str]:
        """Find bank IDs and MOTLIST paths in MOTBANK files near this list.

        WOTS keeps tree-only MOTLISTs beside an owning MOTBANK. Search both a
        loose-file ancestor chain and the owning PAK's known paths so this also
        works when the editor tab uses a virtual ``natives/stm/...`` filename.
        """
        from file_handlers.motbank.motbank_file import MotbankFile

        result: dict[int, str] = {}

        def consume(candidate_path: str, data: bytes) -> None:
            try:
                bank = MotbankFile()
                bank.read(data)
            except (OSError, ValueError) as exc:
                self.errors.append(
                    f"could not parse nearby MOTBANK {candidate_path!r}: {exc}"
                )
                return
            for item in bank.items:
                if not item.path:
                    continue
                path = normalize_resource_path(item.path)
                if not path.casefold().startswith(
                    ("natives/stm/", "natives/x64/")
                ):
                    path = f"natives/stm/{path}"
                result.setdefault(item.bank_id, path)

        local_anchors: list[Path] = []
        anchor = Path(self.anchor_path)
        if anchor.is_absolute() and anchor.is_file():
            local_anchors.append(anchor)
        elif self.resource_context is not None:
            hit = self.resource_context.resolve(
                self.anchor_path,
                self.selection_parent,
                allow_selection_dialog=False,
            )
            if hit is not None:
                resolved = Path(hit[0])
                if resolved.is_absolute() and resolved.is_file():
                    local_anchors.append(resolved)

        visited_directories: set[str] = set()
        for local_anchor in local_anchors:
            current = local_anchor.parent
            while current != current.parent:
                key = str(current).casefold()
                if key in visited_directories:
                    break
                visited_directories.add(key)
                try:
                    candidates = sorted(
                        item
                        for item in current.iterdir()
                        if item.is_file() and ".motbank." in item.name.casefold()
                    )
                except OSError:
                    candidates = []
                for candidate in candidates:
                    try:
                        data = candidate.read_bytes()
                    except OSError as exc:
                        self.errors.append(
                            f"could not read nearby MOTBANK {str(candidate)!r}: {exc}"
                        )
                        continue
                    consume(str(candidate), data)
                if candidates and result:
                    break
                if current.name.casefold() == "natives":
                    break
                current = current.parent

        context = self.resource_context
        reader = context.pak_cached_reader if context is not None else None
        if not result and reader is not None:
            cached_known_paths = getattr(reader, "cached_known_paths", None)
            try:
                paths = tuple(
                    cached_known_paths()
                    if callable(cached_known_paths)
                    else reader.cached_paths(include_unknown=False)
                )
            except (AttributeError, OSError, ValueError) as exc:
                self.errors.append(f"could not enumerate MOTBANK resources: {exc}")
                paths = ()
            anchor_key = resource_path_key(self.anchor_path)
            anchor_directory = anchor_key.rsplit("/", 1)[0]
            candidates = []
            for path in paths:
                normalized = normalize_resource_path(path)
                lowered = normalized.casefold()
                if ".motbank" not in lowered:
                    continue
                parent = lowered.rsplit("/", 1)[0]
                if "/" in anchor_directory and not anchor_directory.startswith(
                    parent + "/"
                ):
                    continue
                candidates.append(normalized)
            candidates.sort(
                key=lambda value: value.count("/"),
                reverse=True,
            )
            for candidate in candidates:
                hit = self.resource_data(candidate)
                if hit is not None:
                    consume(hit[0], hit[1])
        return result

    def load_skeletons(self, path: str):
        """Read only embedded source skeletons from a related MOTLIST."""
        scanner = getattr(self.codec, "parse_skeletons", None)
        if scanner is None:
            return ()
        hit = self.resource_data(path)
        if hit is None:
            return ()
        resolved_path, data = hit
        try:
            return tuple(scanner(data, label=resolved_path))
        except (OSError, ValueError) as exc:
            self.errors.append(
                f"could not scan related MOTLIST skeletons {resolved_path!r}: {exc}"
            )
            return ()

    def _filesystem_hit(self, path: str) -> tuple[str, bytes] | None:
        requested = Path(path)
        candidates = [requested] if requested.is_absolute() else []
        anchor_string = str(self.anchor_path).replace("\\", "/")
        marker = anchor_string.lower().find("/natives/")
        if marker >= 0:
            root = Path(anchor_string[:marker])
            candidates.append(root / Path(_resource_path(path)))
        elif Path(self.anchor_path).is_file():
            candidates.append(Path(self.anchor_path).parent / requested.name)

        for candidate in candidates:
            match = self._matching_file(candidate)
            if match is not None:
                return str(match), match.read_bytes()
        return None

    @staticmethod
    def _matching_file(path: Path) -> Path | None:
        if path.is_file():
            return path
        parent = path.parent
        if not parent.is_dir():
            return None
        matches = sorted(
            item for item in parent.iterdir()
            if item.is_file() and (item.name == path.name or item.name.startswith(path.name + "."))
        )
        return matches[0] if matches else None
