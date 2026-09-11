from __future__ import annotations


LEGACY_MATERIAL_SHADER_NAMES = (
    "Standard",
    "Decal",
    "DecalWithMetallic",
    "DecalNRMR",
    "Transparent",
    "Distortion",
    "PrimitiveMesh",
    "PrimitiveSolidMesh",
    "Water",
    "SpeedTree",
    "GUI",
    "GUIMesh",
    "GUIMeshTransparent",
    "ExpensiveTransparent",
    "Forward",
    "RenderTarget",
    "PostProcess",
    "PrimitiveMaterial",
    "PrimitiveSolidMaterial",
    "SpineMaterial",
    "ReflectiveTransparent",
)


# via.render.MaterialShadingType from the WOTS 1.0.1.0 type database.
# WOTS inserted several shading types before ExpensiveTransparent, so using
# the legacy MDF list makes every value from 2 onward display the wrong name.
WOTS_MATERIAL_SHADER_NAMES = (
    "Standard",
    "Decal",
    "DecalStencil",
    "SeparateAlphaDecal",
    "DecalWithMetallic",
    "DecalNRMR",
    "Transparent",
    "TransparentStencil",
    "Distortion",
    "PrimitiveMesh",
    "PrimitiveSolidMesh",
    "Water",
    "SpeedTree",
    "GUI",
    "GUIMesh",
    "GUIMeshTransparent",
    "ExpensiveTransparent",
    "Forward",
    "RenderTarget",
    "PostProcess",
    "PrimitiveMaterial",
    "PrimitiveSolidMaterial",
    "SpineMaterial",
    "VolumetricFog",
    "ShellFurMaterial",
    "VolumeDecal",
    "AlembicMesh",
    "AlembicMeshForward",
    "AlembicMeshTransparent",
    "MarchingCubes",
    "MarchingCubesForward",
    "MarchingCubesTransparent",
    "Strands",
    "Eyeball",
    "EyeballPostProcess",
    "NFXTransparent",
    "Cloudscape2",
    "CloudscapeReserved",
    "VolumeSolidMaterial",
    "VolumeDecalMetallic",
    "Max",
)


def material_shader_names(layout: str) -> tuple[str, ...]:
    if layout == "onimusha_wots":
        return WOTS_MATERIAL_SHADER_NAMES
    return LEGACY_MATERIAL_SHADER_NAMES
