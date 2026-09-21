from __future__ import annotations

"""Verified WOTS 1.0.1.0 RSZ layout corrections."""

from collections import ChainMap
from copy import deepcopy
from pathlib import Path


WOTS_ATTACK_PARAM_CRCS = {
    "app.cAttackParamDataEnemy": "cdee5bcd",
    "app.cAttackParamDataPlayer": "b25b515f",
}

WOTS_GPU_CLOTH_CRC = "425c28f8"
WOTS_AFTER_IMAGE_CRC = "bda90b3d"

WOTS_LAYOUT_COMPATIBLE_CRCS = {
    # Verified against the WOTS 1.0.1.0 GrappleTablePack resources.  The
    # serialized field layouts match the dump, but the shipped type CRCs do
    # not.  Keep these as an overlay so other games/dumps remain untouched.
    "app.GrappleTableParam.cGrapplePatternInfoData": "b8c51d0e",
    "app.GrappleTableParam.cGrappleTableData": "2967860d",
    "app.user_data.JustGuardConditionMap": "e378c0e0",
    # Verified against the shipped WOTS 1.0.1.0 enemy character PFBs.  These
    # runtime component layouts parse cleanly with the public dump, but their
    # serialized CRCs come from the retail build rather than the dump build.
    "app.ColliderSwitcher": "6877a33c",
    "app.PressController": "8e0091f0",
    "app.EnemyController": "6588d6bc",
    "app.EnemyCharacter": "8e9285fe",
    "app.user_data.EnemyParamPack": "ffc31463",
    "app.AppCoord": "f81038c1",
    "app.CharacterTimelineEventActorEnemy.SubUnitVisibleSetting": "e1f37307",
    "app.CharacterTimelineEventActorEnemy": "36bdf65c",
    "app.MeshSettingController": "1d68a34b",
    "app.EffectMaterialManager": "71412e55",
    "app.user_data.RecursiveEffectTargetList": "fc95f3a6",
    "app.ChainSettingCollection": "08bc6bb8",
    "app.CharacterWetBlendController": "b819f813",
    "app.PhotoControlObject": "79f4ce2f",
    "app.MeshSetting": "1208eaf6",
    "app.ChainSetting": "7f328fd8",
    "app.EnemyGameObjTag": "c57b9d04",
}


def _float_field(name: str) -> dict:
    return {
        "align": 4,
        "array": False,
        "name": name,
        "native": False,
        "original_type": "System.Single",
        "size": 4,
        "type": "F32",
    }


def _s32_field(name: str) -> dict:
    return {
        "align": 4,
        "array": False,
        "name": name,
        "native": False,
        "original_type": "System.Int32",
        "size": 4,
        "type": "S32",
    }


def _native_field(
    name: str,
    field_type: str,
    size: int,
    align: int,
    original_type: str = "",
) -> dict:
    return {
        "align": align,
        "array": False,
        "name": name,
        "native": True,
        "original_type": original_type,
        "size": size,
        "type": field_type,
    }


def _wots_gpu_cloth_fields(base_fields: list[dict]) -> list[dict]:
    """Return the retail WOTS 1.0.1.0 serialized GpuCloth layout.

    The public WOTS dump describes an older 86-field layout.  The retail
    PFBs use the 0x425C28F8 revision: three obsolete flags and the old
    AutoTransform block are gone, while CurrentCalculateMode and
    SmoothingQueueSize were added.  These insertions provide the alignment
    required by DevelopDrawPartBits and every field that follows it.
    """
    fields = deepcopy(list(base_fields[:13]))
    spec = (
        ("ResetPoseResource", "Resource", 4, 4, "via.dynamics.ClothResetPoseResourceHolder"),
        ("DeltaTime", "F32", 4, 4, "System.Single"),
        ("FrameRate", "S32", 4, 4, "System.Int32"),
        ("AutoReset", "Bool", 1, 1, "System.Boolean"),
        ("AutoResetThreshold", "F32", 4, 4, "System.Single"),
        ("AutoResetRotationThreshold", "F32", 4, 4, "System.Single"),
        ("DragEnabled", "Bool", 1, 1, "System.Boolean"),
        ("CalculateMode", "S32", 4, 4, "via.dynamics.ClothCalculateMode"),
        ("CurrentCalculateMode", "S32", 4, 4, "via.dynamics.ClothCalculateMode"),
        ("SimLodLevel", "U32", 4, 4, "System.UInt32"),
        ("SimulationCalculateType", "S32", 4, 4, "via.dynamics.ClothSimulationCalculateType"),
        ("UpdateTiming", "S32", 4, 4, "via.dynamics.ClothUpdateTiming"),
        ("DevelopDraw", "Bool", 1, 1, "System.Boolean"),
        ("DevelopDrawPartBits", "U64", 8, 8, "System.UInt64"),
        ("IsJointVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsControlPointVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsNormalVisible", "Bool", 1, 1, "System.Boolean"),
        ("PositionType", "S32", 4, 4, "via.dynamics.PositionType"),
        ("ControlPointIndexVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsSimulationMeshVisible", "Bool", 1, 1, "System.Boolean"),
        ("SimulationMeshVisibleType", "S32", 4, 4, "via.dynamics.SimulationMeshVisibleType"),
        ("IsIntermediateStepVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsDistanceLinkVisible", "Bool", 1, 1, "System.Boolean"),
        ("DistanceLinkVisibleType", "S32", 4, 4, "via.dynamics.DistanceLinkVisibleType"),
        ("IsLongRangeAttachmentVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsCollisionEdgeVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsCollisionShapeVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsPreviousCollisionShapeVisible", "Bool", 1, 1, "System.Boolean"),
        ("IsCollisionVisible", "Bool", 1, 1, "System.Boolean"),
        ("MaxDistanceVisible", "Bool", 1, 1, "System.Boolean"),
        ("BackstopVisible", "Bool", 1, 1, "System.Boolean"),
        ("ControlPointRadiusVisible", "Bool", 1, 1, "System.Boolean"),
        ("BaseJointInfoVisible", "Bool", 1, 1, "System.Boolean"),
        ("DebugDrawOpacity", "F32", 4, 4, "System.Single"),
        ("DebugDrawFontSize", "F32", 4, 4, "System.Single"),
        ("OnlyControlPointIndex", "S32", 4, 4, "System.Int32"),
        ("IsDomainVisible", "Bool", 1, 1, "System.Boolean"),
        ("DevelopDrawThreshold", "F32", 4, 4, "System.Single"),
        ("SimulationContinuousType", "S32", 4, 4, "via.dynamics.SimulationContinuousType"),
        ("NormalRecalculationEnable", "Bool", 1, 1, "System.Boolean"),
        ("UseVertexShader", "Bool", 1, 1, "System.Boolean"),
        ("VariableFPS", "Bool", 1, 1, "System.Boolean"),
        ("SmoothingQueueSize", "U32", 4, 4, "System.UInt32"),
        ("SelfCollision", "Bool", 1, 1, "System.Boolean"),
        ("SelfCollisionPointVsTriangle", "Bool", 1, 1, "System.Boolean"),
        ("SelfCollisionEdgeVsEdge", "Bool", 1, 1, "System.Boolean"),
        ("CollisionEdgeVsOBBEnabled", "Bool", 1, 1, "System.Boolean"),
        ("UseDQJointDeform", "Bool", 1, 1, "System.Boolean"),
        ("AlwaysHighestSimLodDeform", "Bool", 1, 1, "System.Boolean"),
        ("FreezeMass", "Bool", 1, 1, "System.Boolean"),
        ("VolumeGrowth", "F32", 4, 4, "System.Single"),
        ("CollisionTolerance", "F32", 4, 4, "System.Single"),
        ("VelocityDamping", "F32", 4, 4, "System.Single"),
        ("RootJointName", "String", 4, 4, "System.String"),
        ("BlendAmountTranslation", "F32", 4, 4, "System.Single"),
        ("BlendAmountRotation", "F32", 4, 4, "System.Single"),
        ("WeightOffset", "F32", 4, 4, "System.Single"),
        ("BlendWeight", "F32", 4, 4, "System.Single"),
        ("Resource", "Resource", 4, 4, "via.dynamics.GpuClothResourceHolder"),
        ("FilterInfoResource", "Resource", 4, 4, "via.dynamics.FilterInfoResourceHolder"),
        ("ConstraintSolverIterationCount", "U32", 4, 4, "System.UInt32"),
        ("CollisionSolverIterationCount", "U32", 4, 4, "System.UInt32"),
        ("SelfCollisionSolverIterationCount", "U32", 4, 4, "System.UInt32"),
        ("LoopCount", "U32", 4, 4, "System.UInt32"),
        ("IterationCount", "U32", 4, 4, "System.UInt32"),
        ("InflateBoundingSize", "F32", 4, 4, "System.Single"),
        ("Gravity", "Vec3", 16, 16, "via.vec3"),
        ("LocalWind", "Bool", 1, 1, "System.Boolean"),
        ("CollisionTarget", "S32", 4, 4, "via.dynamics.GpuCloth.CollisionTarget"),
        ("Wind", "Object", 4, 4, "via.dynamics.cloth.Wind"),
        ("CameraSpaceDebugPositionOffset", "Bool", 1, 1, "System.Boolean"),
        ("DebugPositionOffset", "Vec3", 16, 16, "via.vec3"),
    )
    fields.extend(_native_field(*item) for item in spec)
    return fields


class WotsTypeRegistry:
    """Non-mutating overlay for known WOTS RSZ dump mismatches."""

    def __init__(self, base):
        self.base = base
        self.json_path = getattr(base, "json_path", "")
        self._patched_by_id: dict[int, dict] = {}
        self._patched_by_name: dict[str, tuple[dict, int]] = {}
        registry_overrides: dict[str, dict] = {}
        gpu_cloth_info, gpu_cloth_type_id = base.find_type_by_name("via.dynamics.GpuCloth")
        if gpu_cloth_info is not None and gpu_cloth_type_id is not None:
            patched = deepcopy(gpu_cloth_info)
            patched["fields"] = _wots_gpu_cloth_fields(patched.get("fields", ()))
            patched["crc"] = WOTS_GPU_CLOTH_CRC
            gpu_cloth_type_id = int(gpu_cloth_type_id)
            self._patched_by_id[gpu_cloth_type_id] = patched
            self._patched_by_name["via.dynamics.GpuCloth"] = (patched, gpu_cloth_type_id)
            registry_overrides[format(gpu_cloth_type_id, "x")] = patched

        after_image_info, after_image_type_id = base.find_type_by_name(
            "app.AfterImageController"
        )
        if after_image_info is not None and after_image_type_id is not None:
            patched = deepcopy(after_image_info)
            fields = list(patched.get("fields", ()))
            self._insert_after(
                fields,
                "_CacheCount",
                _s32_field("_InitialCacheCount"),
            )
            patched["fields"] = fields
            patched["crc"] = WOTS_AFTER_IMAGE_CRC
            after_image_type_id = int(after_image_type_id)
            self._patched_by_id[after_image_type_id] = patched
            self._patched_by_name["app.AfterImageController"] = (
                patched,
                after_image_type_id,
            )
            registry_overrides[format(after_image_type_id, "x")] = patched

        for name, crc in WOTS_ATTACK_PARAM_CRCS.items():
            info, type_id = base.find_type_by_name(name)
            if info is None or type_id is None:
                continue
            patched = deepcopy(info)
            fields = list(patched.get("fields", ()))
            self._insert_after(fields, "_FireDamage", _float_field("_MiasmaDamage"))
            self._insert_after(
                fields,
                "_MiasmaDamage",
                _float_field("_MiasmaOutsideDamage"),
            )
            self._insert_after(
                fields,
                "_MiasmaOutsideDamage",
                _float_field("_OilDamage"),
            )
            self._insert_after(
                fields,
                "_OniEnergyAttack",
                _float_field("_OniChangeEnergyAttack"),
            )
            if name.endswith("Player"):
                self._insert_after(
                    fields,
                    "_AddSoulBoostGauge",
                    _float_field("_AddSkill5Gauge"),
                )
            patched["fields"] = fields
            patched["crc"] = crc
            type_id = int(type_id)
            self._patched_by_id[type_id] = patched
            self._patched_by_name[name] = (patched, type_id)
            registry_overrides[format(type_id, "x")] = patched

        for name, crc in WOTS_LAYOUT_COMPATIBLE_CRCS.items():
            info, type_id = base.find_type_by_name(name)
            if info is None or type_id is None:
                continue
            patched = deepcopy(info)
            patched["crc"] = crc
            type_id = int(type_id)
            self._patched_by_id[type_id] = patched
            self._patched_by_name[name] = (patched, type_id)
            registry_overrides[format(type_id, "x")] = patched

        self.registry = ChainMap(registry_overrides, getattr(base, "registry", {}))

    @staticmethod
    def _insert_after(fields: list[dict], anchor: str, field: dict) -> None:
        if any(item.get("name") == field["name"] for item in fields):
            return
        for index, item in enumerate(fields):
            if item.get("name") == anchor:
                fields.insert(index + 1, field)
                return
        raise ValueError(f"WOTS RSZ registry is missing {anchor}")

    def get_type_info(self, type_id: int):
        return self._patched_by_id.get(int(type_id)) or self.base.get_type_info(type_id)

    def find_type_by_name(self, type_name: str):
        return self._patched_by_name.get(type_name) or self.base.find_type_by_name(type_name)

    def pre_cache_types(self, type_ids) -> None:
        pre_cache = getattr(self.base, "pre_cache_types", None)
        if callable(pre_cache):
            pre_cache(type_ids)

    def __getattr__(self, name):
        return getattr(self.base, name)


def needs_wots_registry_overlay(registry, game: str = "") -> bool:
    normalized_game = "".join(character for character in str(game) if character.isalnum()).casefold()
    registry_name = Path(str(getattr(registry, "json_path", "") or "")).name.casefold()
    return normalized_game in {"onimushawots", "oniwots", "wots"} or registry_name == "rszoniwots.json"


def apply_wots_registry_overlay(registry, game: str = ""):
    if registry is None or isinstance(registry, WotsTypeRegistry):
        return registry
    return WotsTypeRegistry(registry) if needs_wots_registry_overlay(registry, game) else registry
