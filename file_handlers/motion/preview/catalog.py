from __future__ import annotations

from dataclasses import replace

from utils.resource_file_utils import ResourceResolutionContext

from ..format_codec import MotionFormatCodec
from ..mot.model import Joint, Motion, Skeleton
from .resolution import (
    MotionListDocument,
    MotionResolutionDiagnostic,
    MotionPreviewResolution,
    MotionPreviewResolver,
    TreeMotionReferenceStrategy,
)
from .resources import MotionListResourceStore


class MotionPreviewCatalog:
    """Resolve all playable motions and external list dependencies for a preview."""

    def __init__(
        self,
        root: MotionListDocument,
        tree_references: TreeMotionReferenceStrategy,
        format_codec: MotionFormatCodec,
        *,
        app=None,
        selection_parent=None,
        resource_context: ResourceResolutionContext | None = None,
    ):
        self.root = root
        self._tree_references = tree_references
        self.resources = MotionListResourceStore(
            format_codec,
            app=app,
            anchor_path=root.path,
            selection_parent=selection_parent,
            resource_context=resource_context,
        )
        self._resolver = MotionPreviewResolver(
            self.resources.load,
            tree_references,
            self.resources.load_motion,
        )
        self.resolution = MotionPreviewResolution((), (), (), ())
        self.messages: tuple[str, ...] = ()

    def refresh(self) -> MotionPreviewResolution:
        resolution = self._resolver.resolve(self.root)
        resolution = self._resolve_nearby_motion_banks(resolution)
        resolution = self._resolve_shared_skeletons(resolution)
        self.resolution = resolution
        self.messages = tuple(
            dict.fromkeys(
                list(getattr(self.root.model, "diagnostics", ()))
                + [item.message for item in resolution.diagnostics]
                + self.resources.errors
            )
        )
        return resolution

    def _resolve_nearby_motion_banks(
        self,
        resolution: MotionPreviewResolution,
    ) -> MotionPreviewResolution:
        if not resolution.tree_references or not resolution.unresolved_bank_ids:
            return resolution
        paths = self.resources.find_bank_motion_lists(
            set(resolution.unresolved_bank_ids)
        )
        if not paths:
            return resolution

        entries = list(resolution.entries)
        diagnostics = list(resolution.diagnostics)
        requested = {
            (reference.bank_id, reference.motion_id)
            for reference in resolution.tree_references
        }
        loaded_banks: set[int] = set()
        for bank_id, path in paths.items():
            document = self.resources.load(path)
            if document is None:
                continue
            loaded_banks.add(bank_id)
            bank_resolution = MotionPreviewResolver(
                self.resources.load,
                self._tree_references,
                self.resources.load_motion,
            ).resolve(document)
            diagnostics.extend(bank_resolution.diagnostics)
            by_id = {entry.motion_id: entry for entry in bank_resolution.entries}
            for requested_bank, motion_id in sorted(requested):
                if requested_bank != bank_id:
                    continue
                entry = by_id.get(motion_id)
                if entry is None:
                    diagnostics.append(
                        MotionResolutionDiagnostic(
                            "missing_tree_motion",
                            f"bank {bank_id} MOTLIST {path!r} has no MotionID {motion_id}",
                        )
                    )
                    continue
                entries.append(replace(entry, bank_id=bank_id))

        unresolved = {
            bank_id
            for bank_id in resolution.unresolved_bank_ids
            if bank_id not in loaded_banks
        }
        return MotionPreviewResolution(
            tuple(entries),
            resolution.tree_references,
            tuple(sorted(unresolved)),
            tuple(diagnostics),
        )

    def _resolve_shared_skeletons(
        self,
        resolution: MotionPreviewResolution,
    ) -> MotionPreviewResolution:
        unresolved = []
        candidates: list[Skeleton] = []
        seen_motions: set[int] = set()
        for entry in resolution.entries:
            motion = entry.loaded_motion
            if motion is None or id(motion) in seen_motions:
                continue
            seen_motions.add(id(motion))
            if _motion_needs_shared_skeleton(motion):
                unresolved.append(motion)
            elif _is_complete_source_skeleton(motion.skeleton, motion.source_joint_count):
                candidates.append(motion.skeleton)
        if not unresolved:
            return resolution
        preferred_candidate_ids = {id(skeleton) for skeleton in candidates}

        scanner = getattr(self.resources.codec, "parse_skeletons", None)
        if scanner is None:
            return resolution
        for path in self.resources.nearby_motion_list_paths():
            candidates.extend(self.resources.load_skeletons(path))

        unique_candidates = []
        signatures = set()
        for skeleton in candidates:
            signature = (
                len(skeleton.joints),
                tuple(joint.binding_hash for joint in skeleton.joints),
            )
            if signature not in signatures:
                signatures.add(signature)
                unique_candidates.append(skeleton)

        diagnostics = list(resolution.diagnostics)
        missing_counts = set()
        for motion in unresolved:
            match = _select_shared_skeleton(
                motion,
                unique_candidates,
                preferred_candidate_ids,
            )
            if match is not None:
                _rebind_motion_skeleton(motion, match)
            elif motion.source_joint_count is not None:
                missing_counts.add(motion.source_joint_count)

        for count in sorted(missing_counts):
            diagnostics.append(
                MotionResolutionDiagnostic(
                    "missing_shared_skeleton",
                    f"no compatible {count}-joint skeleton was found in the owning MOTBANK",
                )
            )
        return MotionPreviewResolution(
            resolution.entries,
            resolution.tree_references,
            resolution.unresolved_bank_ids,
            tuple(diagnostics),
        )


def _select_shared_skeleton(
    motion: Motion,
    candidates: list[Skeleton],
    preferred_candidate_ids: set[int] | None = None,
) -> Skeleton | None:
    """Choose a skeleton that can name every animated track safely.

    A skeleton-less WOTS MOT can declare a larger source joint count than the
    skeleton embedded in its MOTLIST.  The extra source joints are not always
    present in that list and may not be animated.  Requiring the candidate's
    record count to reach the declared count therefore rejects valid rigs.
    Track-hash coverage is the binding invariant; when counts do not match,
    prefer a skeleton already decoded from the current MOTLIST over a nearby
    bank entry.
    """
    required = {
        node.joint.binding_hash
        for node in motion.animation_nodes
        if node.joint.binding_hash is not None
    }
    compatible = [
        skeleton
        for skeleton in candidates
        if required.issubset(_skeleton_hashes(skeleton))
    ]
    if not compatible:
        return None
    preferred = preferred_candidate_ids or set()
    source_count = motion.source_joint_count
    return min(
        compatible,
        key=lambda skeleton: (
            source_count is None or len(skeleton.joints) != source_count,
            id(skeleton) not in preferred,
            abs(len(skeleton.joints) - source_count) if source_count is not None else 0,
        ),
    )


def _is_complete_source_skeleton(
    skeleton: Skeleton | None,
    source_joint_count: int | None,
) -> bool:
    if skeleton is None or source_joint_count is None:
        return False
    if len(skeleton.joints) != source_joint_count:
        return False
    hashes = [joint.binding_hash for joint in skeleton.joints]
    return None not in hashes and len(hashes) == len(set(hashes))


def _motion_needs_shared_skeleton(motion: Motion) -> bool:
    if motion.skeleton is not None:
        available = _skeleton_hashes(motion.skeleton)
        required = {
            node.joint.binding_hash
            for node in motion.animation_nodes
            if node.joint.binding_hash is not None
        }
        if required.issubset(available) and all(
            not node.joint.name.startswith("bone_")
            for node in motion.animation_nodes
        ):
            return False
    return bool(
        motion.animation_nodes
        and motion.source_joint_count
        and not _is_complete_source_skeleton(
            motion.skeleton,
            motion.source_joint_count,
        )
    )


def _skeleton_hashes(skeleton: Skeleton) -> set[int]:
    return {
        joint.binding_hash
        for joint in skeleton.joints
        if joint.binding_hash is not None
    }


def _rebind_motion_skeleton(motion: Motion, source: Skeleton) -> None:
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
        if joint.parent is None:
            continue
        parent_index = indices.get(id(joint.parent))
        if parent_index is None:
            continue
        parent = clones[parent_index]
        clones[index].parent = parent
        parent.children.append(clones[index])
    by_hash = {
        joint.binding_hash: joint
        for joint in clones
        if joint.binding_hash is not None
    }
    for node in motion.animation_nodes:
        binding_hash = node.joint.binding_hash
        if binding_hash is not None:
            node.joint = by_hash[binding_hash]
    motion.skeleton = Skeleton(clones)
