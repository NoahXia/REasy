from __future__ import annotations

from typing import Protocol

import numpy as np

from file_handlers.mesh.material_session import scoped_material_key
from ui.scene.mesh_scene import build_mesh_scene
from ui.scene.scene_model import SceneDrawMesh

from ..mot.model import Motion
from .attack_collision import (
    AttackCollisionOverlay,
    AttackCollisionOwnerPose,
    AttackCollisionSource,
    active_attack_collisions,
    attack_collision_geometry_diagnostic,
)
from .model import MotionPreviewSnapshot
from .scene import SkeletonScene
from .skinning import (
    SkinningError,
    build_shared_rig_deformer,
    build_skinned_mesh_deformer,
)
from .target import RigPreviewTarget


class PreviewViewport(Protocol):
    def set_scene(
        self,
        meshes: list[SceneDrawMesh],
        *,
        reset_camera: bool = True,
    ) -> None: ...

    def set_mesh_skinning(self, key: str, binding: object) -> None: ...

    def can_use_gpu_skinning(self, binding: object) -> bool: ...

    def update_mesh_skinning(
        self,
        key: str,
        matrices: np.ndarray,
    ) -> None: ...

    def update_mesh_skinning_source(
        self,
        key: str,
        positions: np.ndarray,
        normals: np.ndarray | None,
    ) -> None: ...

    def clear_mesh_skinning(self, keys: set[str] | None = None) -> None: ...

    def update_mesh_geometry(
        self,
        key: str,
        vertices: np.ndarray,
        normals: np.ndarray | None,
        *,
        recompute_bounds: bool = True,
    ) -> None: ...

    def update_mesh_transforms(
        self,
        matrices: dict[str, np.ndarray],
        *,
        recompute_bounds: bool = True,
    ) -> None: ...

    def set_wireframe_overlays(
        self,
        meshes: list[SceneDrawMesh],
    ) -> None: ...

    def update_wireframe_overlay_geometries(
        self,
        geometries: dict[str, np.ndarray],
    ) -> None: ...

    def set_bone_name_labels(
        self,
        names: tuple[str, ...],
        positions: tuple[tuple[float, float, float], ...],
    ) -> None: ...

    def clear_bone_name_labels(self) -> None: ...


class MotionRenderState:
    """Detect semantic pose/deformation changes between rendered snapshots."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.world_matrices = None
        self.root_deltas = None
        self.deformation_weights = None

    def changes(self, snapshot: MotionPreviewSnapshot) -> tuple[bool, bool]:
        return (
            snapshot.pose.world_matrices is not self.world_matrices
            or snapshot.root_deltas != self.root_deltas,
            snapshot.deformation_weights != self.deformation_weights,
        )

    def accept(self, snapshot: MotionPreviewSnapshot) -> None:
        self.world_matrices = snapshot.pose.world_matrices
        self.root_deltas = snapshot.root_deltas
        self.deformation_weights = snapshot.deformation_weights


class MotionPreviewRenderer:
    """Own the stable preview scene and apply sampled poses incrementally."""

    TARGET_KEY = "motion-preview:target"

    def __init__(self, viewport: PreviewViewport):
        self.viewport = viewport
        self._target: RigPreviewTarget | None = None
        self._target_meshes: dict[str, SceneDrawMesh] = {}
        self._deformers: dict[str, object] = {}
        self._gpu_skinned_keys: set[str] = set()
        self._skeleton: SkeletonScene | None = None
        self._attack_collision_source: AttackCollisionSource | None = None
        self._weapon_attack_sources: dict[int, AttackCollisionSource] = {}
        self._weapon_collision_poses: dict[int, AttackCollisionOwnerPose] = {}
        self._attack_overlay: AttackCollisionOverlay | None = None
        self._attack_hitboxes_enabled = True
        self._hitbox_signature: tuple[int, ...] | None = None
        self._render_state = MotionRenderState()

    def set_attack_collision_source(
        self,
        source: AttackCollisionSource | None,
    ) -> None:
        if source is self._attack_collision_source:
            return
        self._attack_collision_source = source
        self._sync_attack_overlay()
        self._hitbox_signature = None

    def set_weapon_attack_collision_sources(
        self,
        sources: dict[int, AttackCollisionSource] | None,
    ) -> None:
        self._weapon_attack_sources = dict(sources or {})
        self._weapon_collision_poses.clear()
        self._sync_attack_overlay()
        self._hitbox_signature = None

    def _sync_attack_overlay(self) -> None:
        self._attack_overlay = (
            AttackCollisionOverlay(
                self._attack_collision_source,
                self._weapon_attack_sources,
            )
            if self._attack_collision_source is not None
            or self._weapon_attack_sources
            else None
        )

    def set_attack_hitboxes_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._attack_hitboxes_enabled:
            return
        self._attack_hitboxes_enabled = enabled
        self._hitbox_signature = None

    def present(
        self,
        snapshot: MotionPreviewSnapshot,
        target: RigPreviewTarget | None,
        *,
        reset_camera: bool,
        motion: Motion | None = None,
    ) -> None:
        hitbox_signature = self._current_hitbox_signature(motion, snapshot.frame)
        if (
            reset_camera
            or self._skeleton is None
            or self._target is not target
            or not self._skeleton.accepts(snapshot)
        ):
            self._build(
                snapshot,
                target,
                motion=motion,
                reset_camera=reset_camera,
            )
            return
        self._update(snapshot)
        if self._hitbox_signature != hitbox_signature:
            self._replace_attack_overlay(motion, snapshot)

    def _current_hitbox_signature(
        self,
        motion: Motion | None,
        frame: float,
    ) -> tuple[int, ...]:
        if not self._attack_hitboxes_enabled or self._attack_overlay is None:
            return ()
        return self._attack_overlay.signature(
            motion,
            frame,
            self._weapon_collision_poses,
        )

    def attack_collision_status(
        self,
        motion: Motion | None,
        frame: float,
    ) -> str:
        events = active_attack_collisions(motion, frame)
        if not events or not self._attack_hitboxes_enabled:
            return ""
        if self._attack_overlay is None:
            return "Attack hitboxes unavailable: no matching actor Attack RCOL was found."
        resolved = []
        missing_source = []
        for event in events:
            source = self._attack_overlay.source_for_event(event)
            if source is None:
                missing_source.append(event)
            elif source.request_set(event.request_set_id) is not None:
                resolved.append(event)
        unresolved = sorted({
            event.request_set_id
            for event in events
            if self._attack_overlay.source_for_event(event) is not None
            and self._attack_overlay.source_for_event(event).request_set(
                event.request_set_id
            ) is None
        })
        parts = []
        drawable = []
        unsupported = []
        for event in resolved:
            collision_type = int(event.collision_type or 0)
            diagnostic = attack_collision_geometry_diagnostic(
                event,
                weapon_source_available=(
                    collision_type in self._weapon_attack_sources
                ),
                weapon_owner_available=(
                    collision_type in self._weapon_collision_poses
                ),
            )
            (drawable if diagnostic is None else unsupported).append(
                (event, diagnostic)
            )
        if drawable:
            requests = ", ".join(str(value) for value in sorted({
                event.request_set_id for event, _diagnostic in drawable
            }))
            slots = ", ".join(dict.fromkeys(
                event.collision_type_name for event, _diagnostic in drawable
            ))
            parts.append(f"Attack hitboxes: RequestSet {requests} · {slots}")
        for event, diagnostic in unsupported:
            parts.append(
                f"Attack hitbox not drawn: RequestSet {event.request_set_id} · "
                + str(diagnostic)
            )
        for event in missing_source:
            diagnostic = attack_collision_geometry_diagnostic(event)
            parts.append(
                f"Attack hitbox not drawn: RequestSet {event.request_set_id} · "
                + str(diagnostic or "matching Attack RCOL is unavailable")
            )
        if unresolved:
            parts.append(
                "Attack RCOL has no authored RequestSetID "
                + ", ".join(str(value) for value in unresolved)
            )
        return "  ".join(parts)

    def clear(self, *, reset_camera: bool = True) -> None:
        self._target = None
        self._target_meshes.clear()
        self._deformers.clear()
        self._gpu_skinned_keys.clear()
        self._skeleton = None
        self._hitbox_signature = None
        self._weapon_collision_poses.clear()
        self._render_state.clear()
        self.viewport.clear_bone_name_labels()
        self.viewport.set_wireframe_overlays([])
        self.viewport.set_scene([], reset_camera=reset_camera)

    def _build(
        self,
        snapshot: MotionPreviewSnapshot,
        target: RigPreviewTarget | None,
        *,
        motion: Motion | None,
        reset_camera: bool,
    ) -> None:
        self._target = target
        self._target_meshes.clear()
        self._deformers.clear()
        self._gpu_skinned_keys.clear()
        self._render_state.clear()
        self._skeleton = SkeletonScene(snapshot)
        meshes: list[SceneDrawMesh] = []

        if target is not None:
            parts = target.render_parts
            for index, part in enumerate(parts):
                key = (
                    self.TARGET_KEY
                    if len(parts) == 1
                    else f"{self.TARGET_KEY}:{index}"
                )
                material_key = (
                    (
                        lambda name, scope=part.material_scope: scoped_material_key(
                            scope,
                            name,
                        )
                    )
                    if part.material_scope
                    else None
                )
                target_scene = build_mesh_scene(
                    part.mesh,
                    key=key,
                    material_key=material_key,
                )
                if len(target_scene) != 1:
                    raise SkinningError(
                        f"preset mesh {part.label!r} has no renderable LOD 0 geometry"
                    )
                target_mesh = target_scene[0]
                if index == 0 and part.rig is target.rig:
                    deformer = build_skinned_mesh_deformer(
                        part.mesh,
                        target.rig,
                        target_mesh.vertices,
                        target_mesh.normals,
                        target_mesh.indices,
                        handler=part.handler,
                    )
                else:
                    deformer = build_shared_rig_deformer(
                        part.mesh,
                        part.rig,
                        target.rig,
                        target_mesh.vertices,
                        target_mesh.normals,
                        target_mesh.indices,
                        pose_to_constrained_matrix=np.identity(
                            4,
                            dtype=np.float32,
                        ),
                        root_attachment_joint=part.attachment_joint,
                        handler=part.handler,
                    )
                gpu_capability = getattr(
                    self.viewport,
                    "can_use_gpu_skinning",
                    None,
                )
                use_gpu_skinning = (
                    not deformer.requires_post_skin_normals
                    and (
                        gpu_capability is None
                        or bool(gpu_capability(deformer.binding))
                    )
                )
                if not use_gpu_skinning:
                    vertices, normals = deformer.deform(snapshot)
                    target_mesh.vertices = vertices
                    if normals is not None:
                        target_mesh.normals = normals
                else:
                    self._gpu_skinned_keys.add(key)
                self._target_meshes[key] = target_mesh
                self._deformers[key] = deformer
                meshes.append(target_mesh)

        self._weapon_collision_poses = self._build_weapon_collision_poses(
            snapshot
        )
        if not self._target_meshes:
            meshes.extend(self._skeleton.meshes)
        hitbox_meshes = []
        if self._attack_hitboxes_enabled and self._attack_overlay is not None:
            hitbox_meshes = self._attack_overlay.meshes(
                motion,
                snapshot,
                self._weapon_collision_poses,
            )
        self._hitbox_signature = self._current_hitbox_signature(
            motion,
            snapshot.frame,
        )
        self.viewport.set_scene(meshes, reset_camera=reset_camera)
        self.viewport.set_wireframe_overlays(hitbox_meshes)
        self.viewport.set_bone_name_labels(
            snapshot.joint_names,
            snapshot.joint_positions,
        )
        for key in self._gpu_skinned_keys:
            deformer = self._deformers[key]
            self.viewport.set_mesh_skinning(key, deformer.binding)
            if snapshot.deformation_weights:
                positions, normals = deformer.source_geometry(
                    snapshot.deformation_weights
                )
                self.viewport.update_mesh_skinning_source(key, positions, normals)
            self.viewport.update_mesh_skinning(
                key,
                deformer.skin_matrices(snapshot),
            )
        self._render_state.accept(snapshot)

    def _update(self, snapshot: MotionPreviewSnapshot) -> None:
        assert self._skeleton is not None
        self.viewport.set_bone_name_labels(
            snapshot.joint_names,
            snapshot.joint_positions,
        )
        pose_changed, deformation_changed = self._render_state.changes(snapshot)
        for key, deformer in self._deformers.items():
            if key in self._gpu_skinned_keys:
                if deformation_changed:
                    positions, normals = deformer.source_geometry(
                        snapshot.deformation_weights
                    )
                    self.viewport.update_mesh_skinning_source(
                        key,
                        positions,
                        normals,
                    )
                if pose_changed:
                    self.viewport.update_mesh_skinning(
                        key,
                        deformer.skin_matrices(snapshot),
                    )
            elif pose_changed or deformation_changed:
                vertices, normals = deformer.deform(snapshot)
                self.viewport.update_mesh_geometry(
                    key,
                    vertices,
                    normals,
                    recompute_bounds=False,
                )

        if self._target_meshes:
            pass
        elif pose_changed:
            transforms = self._skeleton.transforms(snapshot)
            self.viewport.update_mesh_transforms(transforms, recompute_bounds=False)
        if (
            pose_changed
            and self._attack_hitboxes_enabled
            and self._attack_overlay is not None
        ):
            self._weapon_collision_poses = self._build_weapon_collision_poses(
                snapshot
            )
            self.viewport.update_wireframe_overlay_geometries(
                self._attack_overlay.geometry_updates(
                    snapshot,
                    self._weapon_collision_poses,
                )
            )
        self._render_state.accept(snapshot)

    def _replace_attack_overlay(
        self,
        motion: Motion | None,
        snapshot: MotionPreviewSnapshot,
    ) -> None:
        meshes = []
        if self._attack_hitboxes_enabled and self._attack_overlay is not None:
            meshes = self._attack_overlay.meshes(
                motion,
                snapshot,
                self._weapon_collision_poses,
            )
        self.viewport.set_wireframe_overlays(meshes)
        self._hitbox_signature = self._current_hitbox_signature(
            motion,
            snapshot.frame,
        )

    def _build_weapon_collision_poses(
        self,
        snapshot: MotionPreviewSnapshot,
    ) -> dict[int, AttackCollisionOwnerPose]:
        target = self._target
        if target is None:
            return {}
        parts = target.render_parts
        result = {}
        for index, part in enumerate(parts):
            collision_type = part.weapon_collision_type
            if collision_type is None:
                continue
            key = (
                self.TARGET_KEY
                if len(parts) == 1
                else f"{self.TARGET_KEY}:{index}"
            )
            deformer = self._deformers.get(key)
            world_matrices = getattr(deformer, "world_matrices", None)
            if world_matrices is None:
                continue
            result[int(collision_type)] = AttackCollisionOwnerPose.from_rig(
                part.rig,
                world_matrices(snapshot),
            )
        return result
