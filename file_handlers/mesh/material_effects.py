from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol


BASE_METAL_MAP = "BaseMetalMap"
NORMAL_ROUGHNESS_MAP = "NormalRoughnessMap"
ATLAS_WRINKLE_MASK_MAP = "AtlasWrinkleMaskMap"
ALPHA_TRANSLUCENT_OCCLUSION_SSS_MAP = "AlphaTranslucentOcclusionSSSMap"
BLEND_ATOS = "BlendATOS"
WOTS_NRRO_TEXTURE = "WOTS_NormalRoughnessOcclusion"
WOTS_NORMAL_TEXTURE = "WOTS_Normal"
WOTS_RCTO_TEXTURE = "WOTS_RoughnessCavityTranslucentOcclusion"
WOTS_ALPHA_TEXTURE = "WOTS_Alpha"
WOTS_DETAIL_NRRC_TEXTURE = "Detail_NRRC"
WOTS_DETAIL_MASK_TEXTURE = "DetailMaskMap"
WOTS_STCM_TEXTURE = "SSSTranslucentCavityDetailMaskMap"
WOTS_HAIR_FLOW_TEXTURE = "HairFlowMap"
WOTS_HAIR_HSS_TEXTURE = "Hair_Height_SpecMask_Shift_Map"
WOTS_EMISSIVE_TEXTURE = "WOTS_Emissive"
WOTS_SECOND_ALPHA_TEXTURE = "WOTS_SecondAlpha"
WOTS_GAME_VERSIONS = frozenset({"ONIMUSHAWOTS", "ONIWOTS", "WOTS"})
WOTS_HAIR_TEMPLATES = frozenset({
    "v_chara_pl_hair.mmtr",
    "v_chara_pl_eyebrow.mmtr",
})
WOTS_MATERIAL_FAMILY_GENERIC = 0
WOTS_MATERIAL_FAMILY_SKIN = 1
WOTS_MATERIAL_FAMILY_FACE = 2
WOTS_MATERIAL_FAMILY_MOUTH = 3
WOTS_MATERIAL_FAMILY_HAIR = 4
WOTS_MATERIAL_FAMILY_EYEBROW = 5
WOTS_MATERIAL_FAMILY_BODY_HAIR = 6
WOTS_MATERIAL_FAMILY_EYE = 7
WOTS_MATERIAL_FAMILY_CORNEA = 8
WOTS_MATERIAL_FAMILY_EYE_AO = 9
WOTS_MATERIAL_FAMILY_TEAR = 10
WOTS_MATERIAL_FAMILY_CLOTH = 11
WOTS_TEMPLATE_FAMILIES = {
    "v_chara_pl_skin.mmtr": WOTS_MATERIAL_FAMILY_SKIN,
    "v_chara_pl_face.mmtr": WOTS_MATERIAL_FAMILY_FACE,
    "v_chara_pl_mouth.mmtr": WOTS_MATERIAL_FAMILY_MOUTH,
    "v_chara_pl_hair.mmtr": WOTS_MATERIAL_FAMILY_HAIR,
    "v_chara_pl_eyebrow.mmtr": WOTS_MATERIAL_FAMILY_EYEBROW,
    "v_chara_pl_bodyhair.mmtr": WOTS_MATERIAL_FAMILY_BODY_HAIR,
    "v_chara_pl_eye.mmtr": WOTS_MATERIAL_FAMILY_EYE,
    "v_chara_cornea.mmtr": WOTS_MATERIAL_FAMILY_CORNEA,
    "v_chara_eye_ao.mmtr": WOTS_MATERIAL_FAMILY_EYE_AO,
    "v_chara_tear.mmtr": WOTS_MATERIAL_FAMILY_TEAR,
    "v_chara_pl_cloth_alp.mmtr": WOTS_MATERIAL_FAMILY_CLOTH,
    "v_chara_pl_cloth.mmtr": WOTS_MATERIAL_FAMILY_CLOTH,
}
WOTS_NRRO_ROLES = (
    "NormalRoughnessOcclusionMap",
    "WrinkleBlend_NRROMap",
    "TexChange_NRRO",
    "EyeAwake_Eyes_NRRO",
)
WOTS_ALPHA_ROLES = (
    "AlphaMap",
    "BaseAlphaMap",
)
WOTS_NORMAL_ROLES = (
    "Wrinkle_NRMMap01",
    "NormalMap",
)
WOTS_RCTO_ROLES = (
    "RoughnessCavityTranslucentOcclusionMap",
    "TexChange_RoughnessCavityTranslucentOcclusionMap",
)
WOTS_DETAIL_NRRC_ROLES = (
    "Detail_NRRC",
    "MicroSkin_NRRC",
)
WOTS_DETAIL_MASK_ROLES = (
    "DetailMaskMap",
    "MicroSkin_MaskTex",
)
WOTS_STCM_ROLES = (
    "SSSTranslucentCavityDetailMaskMap",
    "CavityMap",
)
WOTS_EMISSIVE_ROLES = (
    "EmissiveMap",
    "EyeAwake_Face_EMI",
    "FakeHigLightInGameMap",
    "FakeHighLightMap",
)
WOTS_SECOND_ALPHA_ROLES = (
    "SecondAlphaMap",
    "EventDissolve_AlphaMap",
)
# Texture parameters used by WOTS' specialised character shaders.  Keep the
# MDF parameter names here: the Unreal material bundle can then bind them
# without losing the distinction between the face, eye, hair and transparent
# shader families.  Common weather/VFX overlays are intentionally excluded.
WOTS_SPECIALIZED_TEXTURE_ROLES = (
    "MicroSkin_MaskTex",
    "MicroSkin_NRRC",
    "EyeAwake_Face_NRRM",
    "EyeAwake_Face_EMI",
    "EyeAwake_Face_ColorGradient",
    "SweatMaskMap",
    "Wrinkle_ALBMap01",
    "Wrinkle_ALBMap02",
    "Wrinkle_NRMMap01",
    "Wrinkle_NRMMap02",
    "Wrinkle_MaskMap01",
    "Wrinkle_MaskMap02",
    "CavityMap",
    "HairFlowMap",
    "Hair_Height_SpecMask_Shift_Map",
    "Eye_FlatHeightMap",
    "EyeAwake_Eyes_ALBD",
    "EyeAwake_Eyes_NRRO",
    "EmissiveMap",
    "SecondAlphaMap",
    "EventDissolve_AlphaMap",
    "FakeHigLightInGameMap",
    "FakeHighLightMap",
    "StealthMap",
)
WRINKLE_DIFFUSE_MAPS = tuple(
    f"WrinkleDiffuseMap{index}"
    for index in range(1, 4)
)
WRINKLE_NORMAL_MAPS = tuple(
    f"WrinkleNormalMap{index}"
    for index in range(1, 4)
)


class _TextureProfile(Protocol):
    texture_type: str
    texture_path: str


class _SurfaceProfile(Protocol):
    game_version: str
    mmtr_path: str
    textures: tuple[_TextureProfile, ...]
    parameter_names: tuple[str, ...]
    parameters: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class RigLogicMaterialEffect:
    name: str
    game_versions: frozenset[str]
    templates: frozenset[str]
    weight_count: int
    texture_types: tuple[str, ...]

    def matches(self, surface: _SurfaceProfile) -> bool:
        return (
            surface.game_version.upper() in self.game_versions
            and _template_name(surface.mmtr_path) in self.templates
        )

    def validate(self, surface: _SurfaceProfile) -> None:
        _require_unique(
            self.texture_types,
            (texture.texture_type for texture in surface.textures),
            f"{self.name} texture role",
        )
        _require_unique(
            self.weight_names,
            surface.parameter_names,
            f"{self.name} parameter",
        )
        expected_weights = set(self.weight_names)
        unexpected_weights = sorted(
            name
            for name in surface.parameter_names
            if name.startswith("Weight")
            and name[6:].isdigit()
            and name not in expected_weights
        )
        if unexpected_weights:
            raise ValueError(
                f"{self.name} material has unexpected "
                f"{unexpected_weights[0]}"
            )

    @property
    def weight_names(self) -> tuple[str, ...]:
        return tuple(
            f"Weight{index}"
            for index in range(1, self.weight_count + 1)
        )


DMC5_FULL_WRINKLE_EFFECT = RigLogicMaterialEffect(
    name="DMC5 RigLogic 41-weight wrinkles",
    game_versions=frozenset({"DMC5"}),
    templates=frozenset({
        "blendtexture_riglogic.mmtr",
        "blendtexture_riglogic_astral.mmtr",
        "blendtexture_riglogic_pl0200.mmtr",
        "blendtexture_riglogic_pl0200_wet.mmtr",
        "blendtexture_riglogic_trans.mmtr",
        "blendtexture_riglogic_wet.mmtr",
        "shader_03_cs_blendtexture_riglogic_wet_00_000.mmtr",
    }),
    weight_count=41,
    texture_types=(
        BASE_METAL_MAP,
        *WRINKLE_DIFFUSE_MAPS,
        NORMAL_ROUGHNESS_MAP,
        *WRINKLE_NORMAL_MAPS,
        ATLAS_WRINKLE_MASK_MAP,
    ),
)

DMC5_PEOPLE_WRINKLE_EFFECT = RigLogicMaterialEffect(
    name="DMC5 RigLogic 24-weight wrinkles",
    game_versions=frozenset({"DMC5"}),
    templates=frozenset({
        "blendtexture_riglogic_people.mmtr",
        "blendtexture_riglogic_people_wet.mmtr",
    }),
    weight_count=24,
    texture_types=(
        BASE_METAL_MAP,
        NORMAL_ROUGHNESS_MAP,
        WRINKLE_NORMAL_MAPS[0],
        WRINKLE_NORMAL_MAPS[1],
        ATLAS_WRINKLE_MASK_MAP,
    ),
)

DMC5_TEETH_OCCLUSION_EFFECT = RigLogicMaterialEffect(
    name="DMC5 RigLogic teeth occlusion",
    game_versions=frozenset({"DMC5"}),
    templates=frozenset({
        "blendtexture_riglogic_teeth.mmtr",
        "blendtexture_riglogic_teeth_trans.mmtr",
    }),
    weight_count=1,
    texture_types=(
        BASE_METAL_MAP,
        NORMAL_ROUGHNESS_MAP,
        ALPHA_TRANSLUCENT_OCCLUSION_SSS_MAP,
        BLEND_ATOS,
    ),
)

DMC5_RIGLOGIC_EFFECTS = (
    DMC5_FULL_WRINKLE_EFFECT,
    DMC5_PEOPLE_WRINKLE_EFFECT,
    DMC5_TEETH_OCCLUSION_EFFECT,
)


def riglogic_material_effect(
    surface: _SurfaceProfile | None,
) -> RigLogicMaterialEffect | None:
    if surface is None:
        return None
    for effect in DMC5_RIGLOGIC_EFFECTS:
        if effect.matches(surface):
            effect.validate(surface)
            return effect
    return None


def material_texture_key(
    material_key: str,
    texture_type: str,
) -> str:
    if texture_type == BASE_METAL_MAP:
        return material_key
    return f"{material_key}\x1f{texture_type}"


def surface_texture_paths(
    surface: _SurfaceProfile,
) -> dict[str, str]:
    return {
        texture.texture_type: texture.texture_path
        for texture in surface.textures
    }


def is_wots_surface(surface: _SurfaceProfile | None) -> bool:
    if surface is None:
        return False
    game = "".join(
        character
        for character in str(getattr(surface, "game_version", "")).upper()
        if character.isalnum()
    )
    return game in WOTS_GAME_VERSIONS


def wots_material_texture_paths(
    surface: _SurfaceProfile | None,
) -> dict[str, str]:
    """Map WOTS texture aliases onto stable preview roles.

    The role names in MDF files vary between character templates.  Selection
    remains deterministic and never guesses from a filename.
    """

    if not is_wots_surface(surface):
        return {}
    by_name = {
        texture.texture_type.casefold(): texture.texture_path
        for texture in surface.textures
        if texture.texture_type and texture.texture_path
    }

    def first(roles: tuple[str, ...]) -> str:
        return next(
            (by_name[role.casefold()] for role in roles if role.casefold() in by_name),
            "",
        )

    paths = {
        role: path
        for role, path in (
            (WOTS_NRRO_TEXTURE, first(WOTS_NRRO_ROLES)),
            (WOTS_NORMAL_TEXTURE, first(WOTS_NORMAL_ROLES)),
            (WOTS_RCTO_TEXTURE, first(WOTS_RCTO_ROLES)),
            (WOTS_ALPHA_TEXTURE, first(WOTS_ALPHA_ROLES)),
            (WOTS_DETAIL_NRRC_TEXTURE, first(WOTS_DETAIL_NRRC_ROLES)),
            (WOTS_DETAIL_MASK_TEXTURE, first(WOTS_DETAIL_MASK_ROLES)),
            (WOTS_STCM_TEXTURE, first(WOTS_STCM_ROLES)),
            (WOTS_EMISSIVE_TEXTURE, first(WOTS_EMISSIVE_ROLES)),
            (WOTS_SECOND_ALPHA_TEXTURE, first(WOTS_SECOND_ALPHA_ROLES)),
        )
        if path
    }
    paths.update(
        (role, path)
        for role in WOTS_SPECIALIZED_TEXTURE_ROLES
        if (path := first((role,)))
    )
    return paths


def wots_material_parameters(
    surface: _SurfaceProfile | None,
) -> dict[str, float | bool]:
    """Extract the verified common WOTS character-material controls."""

    if not is_wots_surface(surface):
        return {"wots_material": False}
    values = {
        str(name).casefold(): tuple(float(value) for value in components)
        for name, components in getattr(surface, "parameters", {}).items()
        if components
    }

    def scalar(name: str, default: float) -> float:
        components = values.get(name.casefold())
        return float(components[0]) if components else float(default)

    def first_scalar(names: tuple[str, ...], default: float) -> float:
        for name in names:
            components = values.get(name.casefold())
            if components:
                return float(components[0])
        return float(default)

    template = _template_name(surface.mmtr_path)
    hair_material = template in WOTS_HAIR_TEMPLATES
    material_family = WOTS_TEMPLATE_FAMILIES.get(
        template, WOTS_MATERIAL_FAMILY_GENERIC
    )
    micro_skin_rate = scalar("MicroSkin_BlendRate", 0.0)
    # WOTS face colour/normal textures are 2x2 expression atlases.  The MDF
    # stores their sub-UV scale as Dummy_UVScale (0.5 for the shipped face
    # materials).  Neutral preview uses the first tile; expression blending
    # can select the other tiles once runtime wrinkle weights are available.
    face_uv_scale = (
        scalar("Dummy_UVScale", 0.5)
        if material_family == WOTS_MATERIAL_FAMILY_FACE
        else 1.0
    )
    return {
        "wots_material": True,
        "hair_material": hair_material,
        "material_family": material_family,
        # The face atlas itself uses UV0.  UV1 is the continuous secondary UV
        # used by overlays such as blood/oil, not the wrinkle/base atlas.
        "use_secondary_uv": False,
        "face_uv_scale": face_uv_scale,
        "roughness_scale": scalar("RoughnessScale", 1.0),
        "occlusion_scale": scalar("OcclusionScale", 1.0),
        "alpha_adjust": scalar("AlphaAdjust", 1.0),
        "alpha_threshold": scalar("AlphaTestThreshold", 0.5),
        "alpha_test": scalar("IsAlphaTest", 0.0) >= 0.5,
        "use_separate_alpha": scalar("UseSeparateAlpha", 0.0),
        "use_detail": (
            scalar("UseDetail", 0.0) >= 0.5 or micro_skin_rate > 0.0
        ),
        "detail_tiling": first_scalar(
            ("Detail_Tiling", "MicroSkin_Tiling"), 1.0
        ),
        "normal_blend_rate": first_scalar(
            ("Normal_BlendRate", "MicroSkin_NormalScale"), 1.0
        ) * (micro_skin_rate if micro_skin_rate > 0.0 else 1.0),
        "roughness_blend_rate": first_scalar(
            ("Roughness_BlendRate", "MicroSkin_RoughnessScale"), 0.0
        ) * (micro_skin_rate if micro_skin_rate > 0.0 else 1.0),
        "cavity_blend_rate": first_scalar(
            ("Cavity_BlendRate", "MicroSkin_CavityScale"), 0.0
        ) * (micro_skin_rate if micro_skin_rate > 0.0 else 1.0),
        "sss_scale": scalar("SSSScale", 0.0),
        "use_flow_map": scalar("UseFlowMap", 0.0),
        "secondary_specular_intensity": scalar(
            "Secondary_Specular_Intensity", 1.0
        ),
        "primary_spec_sharpness": scalar("PrimarySpec_Sharpness", 50.0),
        "secondary_spec_sharpness": scalar(
            "SecondarySpec_Sharpness", 20.0
        ),
        "primary_specular_shift_offset": scalar(
            "Primary_Specular_ShiftOffset", 0.1
        ),
        "secondary_specular_shift_offset": scalar(
            "Secondary_Specular_ShiftOffset", 0.1
        ),
        "hair_height_depth": scalar("Hair_Height_Depth", 0.0),
        "specular": first_scalar(("Specular", "SpecularScale"), 0.5),
        "primary_specular_level": scalar("Primary_Specular_Color", 0.023529),
        "ao_exp": scalar("AOExp", 0.0),
        "emissive_intensity": scalar("EmissiveIntensity", 0.0),
        "translucent_scale": scalar("TranslucentScale", 0.0),
    }


def _template_name(path: str) -> str:
    normalized = str(path or "").replace("\\", "/").lower()
    return PurePosixPath(normalized).name


def _require_unique(required, available, label: str) -> None:
    values = tuple(available)
    for name in required:
        count = values.count(name)
        if count != 1:
            detail = "is missing" if count == 0 else f"appears {count} times"
            raise ValueError(f"{label} {name} {detail}")
