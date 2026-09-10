from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ui.scene.mesh_scene import (
    build_mesh_scene,
    mesh_lod0_submeshes,
    mesh_scene_payloads,
)


_COMPONENT = {
    np.dtype("float32"): 5126,
    np.dtype("uint8"): 5121,
    np.dtype("uint16"): 5123,
    np.dtype("uint32"): 5125,
}


@dataclass(slots=True)
class GltfTextureImage:
    width: int
    height: int
    rgba: bytes
    source_path: str = ""


@dataclass(slots=True)
class GltfMaterialAsset:
    name: str
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    roughness_factor: float = 1.0
    metallic_factor: float = 0.0
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    base_color: GltfTextureImage | None = None
    normal: GltfTextureImage | None = None
    orm: GltfTextureImage | None = None
    wots_textures: dict[str, GltfTextureImage] = field(default_factory=dict)
    unreal_master_material: str = ""
    extras: dict = field(default_factory=dict)


class _GltfBuilder:
    def __init__(self):
        self.binary = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def accessor(
        self,
        values,
        kind: str,
        *,
        target: int | None = None,
        normalized: bool = False,
        bounds: bool = False,
    ) -> int:
        array = np.ascontiguousarray(values)
        dtype = array.dtype
        if dtype not in _COMPONENT:
            raise ValueError(f"unsupported glTF component dtype {dtype}")
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[kind]
        flat = array.reshape(-1, width) if width > 1 else array.reshape(-1)
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        payload = flat.tobytes(order="C")
        self.binary.extend(payload)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        view_index = len(self.views)
        self.views.append(view)
        accessor = {
            "bufferView": view_index,
            "componentType": _COMPONENT[dtype],
            "count": len(flat),
            "type": kind,
        }
        if normalized:
            accessor["normalized"] = True
        if bounds and len(flat):
            rows = flat.reshape(len(flat), -1)
            accessor["min"] = rows.min(axis=0).astype(float).tolist()
            accessor["max"] = rows.max(axis=0).astype(float).tolist()
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index

    def blob(self, payload: bytes) -> int:
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        self.binary.extend(payload)
        index = len(self.views)
        self.views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        )
        return index


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _encode_png(image: GltfTextureImage) -> bytes:
    width, height = int(image.width), int(image.height)
    expected = width * height * 4
    if width <= 0 or height <= 0 or len(image.rgba) != expected:
        raise ValueError(
            f"invalid RGBA texture {width}x{height}: {len(image.rgba)} bytes"
        )
    stride = width * 4
    rows = b"".join(
        b"\0" + image.rgba[offset : offset + stride]
        for offset in range(0, expected, stride)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, 6))
        + _png_chunk(b"IEND", b"")
    )


def _decoded_texture(handler, texture_path: str, cache: dict):
    from file_handlers.mesh.material_resolver import MeshMaterialResolver
    from file_handlers.tex.qt_image_utils import parse_tex_bytes
    from file_handlers.tex.texture_decoder import decode_tex_mip

    if not texture_path:
        return None
    cache_key = texture_path.replace("\\", "/").casefold()
    if cache_key in cache:
        return cache[cache_key]
    resolved = MeshMaterialResolver.resolve_texture_path(
        handler,
        texture_path,
        prefer_streaming=True,
    )
    if resolved is None:
        cache[cache_key] = None
        return None
    actual_path, data = resolved
    tex = parse_tex_bytes(data, raise_errors=True)
    decoded = decode_tex_mip(tex, 0, 0)
    result = GltfTextureImage(
        int(decoded.width),
        int(decoded.height),
        bytes(decoded.rgba),
        actual_path,
    )
    cache[cache_key] = result
    return result


_WOTS_UNREAL_MATERIAL_ROOT = "/Game/Prototype/Demo/Onimusha/Materials"


def _wots_unreal_master(material_name: str, mmtr_path: str, alpha_test: bool) -> str:
    identity = f"{material_name} {mmtr_path}".casefold()
    if any(token in identity for token in ("hair", "beard", "eyelash", "brow")):
        name = "M_WOTS_Hair"
    elif any(token in identity for token in ("skin", "face")):
        name = "M_WOTS_Skin"
    elif alpha_test:
        name = "M_WOTS_Masked"
    else:
        name = "M_WOTS_Surface"
    return f"{_WOTS_UNREAL_MATERIAL_ROOT}/{name}"


def resolve_gltf_materials(handler, resolved_mdf=None) -> dict[str, GltfMaterialAsset]:
    """Resolve MDF/TEX resources into portable glTF PBR material inputs."""
    from file_handlers.mesh.material_effects import (
        WOTS_ALPHA_TEXTURE,
        WOTS_DETAIL_MASK_TEXTURE,
        WOTS_DETAIL_NRRC_TEXTURE,
        WOTS_NORMAL_TEXTURE,
        WOTS_NRRO_TEXTURE,
        WOTS_RCTO_TEXTURE,
        WOTS_SPECIALIZED_TEXTURE_ROLES,
        WOTS_STCM_TEXTURE,
        wots_material_parameters,
        wots_material_texture_paths,
    )
    from file_handlers.mesh.material_resolver import MeshMaterialResolver

    if resolved_mdf is None:
        resolved_mdf = MeshMaterialResolver.resolve_mdf_for_handler(handler)
    if resolved_mdf is None:
        return {}

    texture_cache: dict[str, GltfTextureImage | None] = {}
    result = {}
    for name, surface in resolved_mdf.surfaces.items():
        role_paths = wots_material_texture_paths(surface)
        parameters = wots_material_parameters(surface)
        base = _decoded_texture(handler, surface.texture_path, texture_cache)
        alpha = _decoded_texture(
            handler, role_paths.get(WOTS_ALPHA_TEXTURE, ""), texture_cache
        )
        nrro = _decoded_texture(
            handler, role_paths.get(WOTS_NRRO_TEXTURE, ""), texture_cache
        )
        normal_source = _decoded_texture(
            handler, role_paths.get(WOTS_NORMAL_TEXTURE, ""), texture_cache
        )
        rcto = _decoded_texture(
            handler, role_paths.get(WOTS_RCTO_TEXTURE, ""), texture_cache
        )
        detail_nrrc = _decoded_texture(
            handler, role_paths.get(WOTS_DETAIL_NRRC_TEXTURE, ""), texture_cache
        )
        detail_mask = _decoded_texture(
            handler, role_paths.get(WOTS_DETAIL_MASK_TEXTURE, ""), texture_cache
        )
        stcm = _decoded_texture(
            handler, role_paths.get(WOTS_STCM_TEXTURE, ""), texture_cache
        )
        alpha_adjust = float(parameters.get("alpha_adjust", 1.0))
        roughness_scale = float(parameters.get("roughness_scale", 1.0))
        occlusion_scale = float(parameters.get("occlusion_scale", 1.0))
        alpha_test = bool(parameters.get("alpha_test", False))
        texture_metadata = {
            texture.texture_type: texture.texture_path
            for texture in surface.textures
        }
        missing = sorted({
            path
            for path in (surface.texture_path, *role_paths.values())
            if path and _decoded_texture(handler, path, texture_cache) is None
        })
        reasy_extras = {
            "gameVersion": surface.game_version,
            "mmtrPath": surface.mmtr_path,
            "textures": texture_metadata,
            "parameters": {
                key: list(values) for key, values in surface.parameters.items()
            },
        }
        if missing:
            reasy_extras["missingTextures"] = missing
        raw_textures = {
            role: image
            for role, image in (
                ("BaseColorTexture", base),
                ("NRROTexture", nrro),
                ("NormalTexture", normal_source),
                ("RCTOTexture", rcto),
                ("AlphaTexture", alpha),
                ("Detail_NRRC", detail_nrrc),
                ("DetailMaskMap", detail_mask),
                ("SSSTranslucentCavityDetailMaskMap", stcm),
            )
            if image is not None
        }
        for parameter_name in WOTS_SPECIALIZED_TEXTURE_ROLES:
            image = _decoded_texture(
                handler,
                role_paths.get(parameter_name, ""),
                texture_cache,
            )
            if image is not None:
                raw_textures[parameter_name] = image
        unreal_master = _wots_unreal_master(
            name,
            surface.mmtr_path,
            alpha_test,
        )
        reasy_extras["unreal"] = {
            "masterMaterial": unreal_master,
            "textureEncoding": "WOTS_RAW_RGBA",
            "textureParameters": tuple(raw_textures),
            "scalarParameters": {
                "RoughnessScale": roughness_scale,
                "OcclusionScale": occlusion_scale,
                "AlphaAdjust": alpha_adjust,
                "Detail_Tiling": float(parameters.get("detail_tiling", 1.0)),
                "Normal_BlendRate": float(parameters.get("normal_blend_rate", 1.0)),
                "Roughness_BlendRate": float(parameters.get("roughness_blend_rate", 0.0)),
                "Cavity_BlendRate": float(parameters.get("cavity_blend_rate", 0.0)),
                "SSSScale": float(parameters.get("sss_scale", 0.0)),
            },
            "staticSwitchParameters": {
                "UseRCTO": rcto is not None,
                "UseStandaloneNormal": normal_source is not None and nrro is None,
                "UseDetail": bool(parameters.get("use_detail", False)) and detail_nrrc is not None,
            },
            "vectorParameters": {
                "BaseColorTint": list(surface.tint),
            },
        }
        result[name] = GltfMaterialAsset(
            name=name,
            base_color_factor=tuple(float(value) for value in surface.tint),
            roughness_factor=1.0,
            metallic_factor=0.0,
            double_sided=bool(surface.two_sided),
            alpha_mode=("MASK" if alpha_test else ("BLEND" if alpha is not None else "OPAQUE")),
            alpha_cutoff=float(parameters.get("alpha_threshold", 0.5)),
            wots_textures=raw_textures,
            unreal_master_material=unreal_master,
            extras={"REasy": reasy_extras},
        )
    return result


def _converted_trs(transform):
    tx, ty, tz = transform.translation
    qx, qy, qz, qw = transform.rotation
    return (
        [float(tx), float(ty), float(tz)],
        [float(qx), float(qy), float(qz), float(qw)],
        [float(value) for value in transform.scale],
    )


def _inverse_bind_matrices(rig) -> np.ndarray:
    from file_handlers.motion.evaluation.math3d import compose_world_matrices

    worlds = compose_world_matrices(
        tuple(joint.rest for joint in rig.joints),
        tuple(joint.parent_index for joint in rig.joints),
    )
    matrices = []
    for index, joint in enumerate(rig.joints):
        if joint.inverse_bind_matrix is None:
            inverse_row = np.linalg.inv(
                np.asarray(worlds[index], dtype=np.float64).reshape(4, 4)
            )
        else:
            inverse_row = np.asarray(
                joint.inverse_bind_matrix, dtype=np.float64
            ).reshape(4, 4)
        # RE Engine's WOTS matrices are stored row-major. glTF MAT4 payloads
        # are column-major, so the same flat values represent the transposed
        # mathematical matrix without an additional axis reflection.
        matrices.append(inverse_row.astype(np.float32).reshape(16))
    return np.asarray(matrices, dtype=np.float32)


def _mesh_skin(
    mesh,
    vertex_count: int,
    *,
    include_auxiliary_groups: bool = False,
):
    remap = np.asarray(mesh.bone_remap_indices, dtype=np.int64)
    if not len(remap):
        return None
    influences = []
    weights = []
    submeshes = mesh_lod0_submeshes(
        mesh,
        include_auxiliary_groups=include_auxiliary_groups,
    )
    for record in mesh_scene_payloads(mesh, submeshes):
        streams = [
            stream
            for stream in (
                getattr(record.payload, "skin_weights", None),
                getattr(record.payload, "extra_skin_weights", None),
            )
            if stream is not None
        ]
        if not streams:
            return None
        di, dw = [], []
        for stream in streams:
            width = stream.influence_count
            di.append(
                np.asarray(stream.deform_indices, dtype=np.int64).reshape(-1, width)[
                    : record.vertex_count
                ]
            )
            dw.append(
                np.asarray(stream.weights, dtype=np.float32).reshape(-1, width)[
                    : record.vertex_count
                ]
            )
        influences.append(np.concatenate(di, axis=1))
        weights.append(np.concatenate(dw, axis=1))
    deform = np.concatenate(influences)
    weight = np.concatenate(weights)
    if len(deform) != vertex_count:
        raise ValueError("LOD0 skin weights do not match exported vertices")
    active = weight > 0
    if np.any(active & ((deform < 0) | (deform >= len(remap)))):
        bad = int(deform[active & ((deform < 0) | (deform >= len(remap)))][0])
        raise ValueError(f"skin references deform index {bad} outside remap table")
    safe = deform.copy()
    safe[~active] = 0
    joints = remap[safe]
    if np.any(active & ((joints < 0) | (joints >= mesh.joint_count))):
        bad = int(joints[active & ((joints < 0) | (joints >= mesh.joint_count))][0])
        raise ValueError(f"skin remap references joint {bad} outside skeleton")
    # glTF exposes at most eight influences through JOINTS_0/1. Keep the
    # strongest lanes when a resource contains a secondary weight stream.
    if weight.shape[1] > 8:
        order = np.argsort(-weight, axis=1)[:, :8]
        weight = np.take_along_axis(weight, order, axis=1)
        joints = np.take_along_axis(joints, order, axis=1)
    if weight.shape[1] < 8:
        pad = 8 - weight.shape[1]
        weight = np.pad(weight, ((0, 0), (0, pad)))
        joints = np.pad(joints, ((0, 0), (0, pad)))
    sums = weight.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("skin contains a vertex without positive weights")
    return joints.astype(np.uint16), (weight / sums).astype(np.float32)


def _add_texture(
    document: dict,
    builder: _GltfBuilder,
    image: GltfTextureImage,
    name: str,
    cache: dict[tuple[str, int, int, bytes], int] | None = None,
) -> int:
    source_key = str(image.source_path or "").replace("\\", "/").casefold()
    cache_key = (source_key, int(image.width), int(image.height), image.rgba)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    view = builder.blob(_encode_png(image))
    image_index = len(document["images"])
    document["images"].append(
        {
            "name": name,
            "mimeType": "image/png",
            "bufferView": view,
            "extras": {"source": image.source_path} if image.source_path else {},
        }
    )
    texture_index = len(document["textures"])
    document["textures"].append({"source": image_index})
    if cache is not None:
        cache[cache_key] = texture_index
    return texture_index


def _material_document(
    document: dict,
    builder: _GltfBuilder,
    asset: GltfMaterialAsset,
    texture_cache: dict[tuple[str, int, int, bytes], int] | None = None,
) -> dict:
    pbr = {
        "baseColorFactor": list(asset.base_color_factor),
        "metallicFactor": float(asset.metallic_factor),
        "roughnessFactor": float(asset.roughness_factor),
    }
    result = {"name": asset.name, "pbrMetallicRoughness": pbr}
    extras = dict(asset.extras)
    if asset.wots_textures:
        texture_indices = {}
        for parameter_name, image in asset.wots_textures.items():
            texture_indices[parameter_name] = _add_texture(
                document,
                builder,
                image,
                f"{asset.name}_{parameter_name}",
                texture_cache,
            )
        base_index = texture_indices.get("BaseColorTexture")
        if base_index is not None:
            pbr["baseColorTexture"] = {"index": base_index}
        reasy = dict(extras.get("REasy", {}))
        unreal = dict(reasy.get("unreal", {}))
        unreal["masterMaterial"] = asset.unreal_master_material
        unreal["textureIndices"] = texture_indices
        reasy["unreal"] = unreal
        extras["REasy"] = reasy
    elif asset.base_color is not None:
        index = _add_texture(document, builder, asset.base_color, f"{asset.name}_BaseColor")
        pbr["baseColorTexture"] = {"index": index}
    if asset.normal is not None:
        index = _add_texture(document, builder, asset.normal, f"{asset.name}_Normal")
        result["normalTexture"] = {"index": index}
    if asset.orm is not None:
        index = _add_texture(document, builder, asset.orm, f"{asset.name}_ORM")
        result["occlusionTexture"] = {"index": index}
        pbr["metallicRoughnessTexture"] = {"index": index}
    if asset.double_sided:
        result["doubleSided"] = True
    if asset.alpha_mode != "OPAQUE":
        result["alphaMode"] = asset.alpha_mode
        if asset.alpha_mode == "MASK":
            result["alphaCutoff"] = float(asset.alpha_cutoff)
    if extras:
        result["extras"] = extras
    return result


def _safe_export_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("._")
    return cleaned or "material"


def _source_texture_export_name(image: GltfTextureImage) -> str:
    """Keep the original RE texture basename while changing it to PNG."""
    source_name = Path(str(image.source_path or "")).name
    lower_name = source_name.casefold()
    tex_marker = lower_name.find(".tex")
    if tex_marker > 0:
        source_name = source_name[:tex_marker]
    elif source_name:
        source_name = Path(source_name).stem
    return f"{_safe_export_name(source_name)}.png" if source_name else "texture.png"


_WOTS_SRGB_TEXTURE_PARAMETERS = frozenset({
    "BaseColorTexture",
    "Wrinkle_ALBMap01",
    "Wrinkle_ALBMap02",
    "EyeAwake_Eyes_ALBD",
    "EyeAwake_Face_ColorGradient",
    "EmissiveMap",
    "FakeHigLightInGameMap",
})


def _write_wots_unreal_bundle(
    gltf_path: Path,
    material_assets: dict[str, GltfMaterialAsset] | None,
) -> Path | None:
    assets = material_assets or {}
    if not any(asset.wots_textures for asset in assets.values()):
        return None

    texture_dir = gltf_path.with_name(f"{gltf_path.stem}_wots_textures")
    texture_dir.mkdir(parents=True, exist_ok=True)
    exported_files: dict[str, tuple[str, bytes]] = {}
    materials = []
    for material_name, asset in assets.items():
        if not asset.wots_textures:
            continue
        texture_files = {}
        for parameter_name, image in asset.wots_textures.items():
            payload = _encode_png(image)
            filename = _source_texture_export_name(image)
            source_key = str(image.source_path or "").replace("\\", "/").casefold()
            previous = exported_files.get(filename.casefold())
            if previous is not None and previous != (source_key, payload):
                # Preserve the original basename in the ordinary case. Only a
                # genuine same-name collision receives a stable content suffix.
                suffix = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
                filename = f"{Path(filename).stem}__{suffix}.png"
            if filename.casefold() not in exported_files:
                (texture_dir / filename).write_bytes(payload)
                exported_files[filename.casefold()] = (source_key, payload)
            texture_files[parameter_name] = {
                "file": f"{texture_dir.name}/{filename}",
                "source": image.source_path,
                "sRGB": parameter_name in _WOTS_SRGB_TEXTURE_PARAMETERS,
                "compression": (
                    "TC_Default"
                    if parameter_name in _WOTS_SRGB_TEXTURE_PARAMETERS
                    else "TC_Masks"
                ),
                "flipGreenChannel": False,
            }
        reasy = asset.extras.get("REasy", {})
        unreal = reasy.get("unreal", {})
        materials.append(
            {
                "name": material_name,
                "masterMaterial": asset.unreal_master_material,
                "textures": texture_files,
                "scalarParameters": unreal.get("scalarParameters", {}),
                "staticSwitchParameters": unreal.get(
                    "staticSwitchParameters", {}
                ),
                "vectorParameters": unreal.get("vectorParameters", {}),
                "alphaMode": asset.alpha_mode,
                "alphaCutoff": asset.alpha_cutoff,
                "doubleSided": asset.double_sided,
            }
        )

    manifest_path = gltf_path.with_name(f"{gltf_path.stem}.wots-materials.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "REasy.WOTS.UnrealMaterials/1",
                "model": gltf_path.name,
                "materials": materials,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _base_document(
    mesh=None,
    rig=None,
    material_assets=None,
    *,
    include_auxiliary_groups: bool = False,
):
    builder = _GltfBuilder()
    document = {
        "asset": {"version": "2.0", "generator": "REasy WOTS exporter"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "images": [],
        "textures": [],
        "skins": [],
        "animations": [],
    }
    texture_cache: dict[tuple[str, int, int, bytes], int] = {}
    joint_nodes = []
    if rig is not None:
        for joint in rig.joints:
            translation, rotation, scale = _converted_trs(joint.rest)
            node = {
                "name": joint.name,
                "translation": translation,
                "rotation": rotation,
                "scale": scale,
            }
            joint_nodes.append(len(document["nodes"]))
            document["nodes"].append(node)
        for index, joint in enumerate(rig.joints):
            if joint.parent_index is None:
                document["scenes"][0]["nodes"].append(joint_nodes[index])
            else:
                document["nodes"][joint_nodes[joint.parent_index]].setdefault(
                    "children", []
                ).append(joint_nodes[index])

    if mesh is not None:
        scenes = build_mesh_scene(
            mesh,
            key="export",
            include_auxiliary_groups=include_auxiliary_groups,
        )
        if not scenes:
            raise ValueError("mesh has no LOD0 geometry")
        scene = scenes[0]
        positions = np.asarray(scene.vertices, dtype=np.float32).copy()
        attributes = {
            "POSITION": builder.accessor(positions, "VEC3", target=34962, bounds=True)
        }
        if scene.normals is not None:
            normals = np.asarray(scene.normals, dtype=np.float32).copy()
            attributes["NORMAL"] = builder.accessor(normals, "VEC3", target=34962)
        if scene.uvs is not None:
            attributes["TEXCOORD_0"] = builder.accessor(
                np.asarray(scene.uvs, dtype=np.float32),
                "VEC2",
                target=34962,
            )
        if scene.colors is not None:
            attributes["COLOR_0"] = builder.accessor(
                np.asarray(scene.colors, dtype=np.float32),
                "VEC4",
                target=34962,
            )
        if rig is not None and mesh.joint_count:
            skin = _mesh_skin(
                mesh,
                len(positions),
                include_auxiliary_groups=include_auxiliary_groups,
            )
            if skin is not None:
                joints, weights = skin
                attributes["JOINTS_0"] = builder.accessor(
                    joints[:, :4], "VEC4", target=34962
                )
                attributes["WEIGHTS_0"] = builder.accessor(
                    weights[:, :4], "VEC4", target=34962
                )
                attributes["JOINTS_1"] = builder.accessor(
                    joints[:, 4:8], "VEC4", target=34962
                )
                attributes["WEIGHTS_1"] = builder.accessor(
                    weights[:, 4:8], "VEC4", target=34962
                )
                inverse = builder.accessor(_inverse_bind_matrices(rig), "MAT4")
                document["skins"].append(
                    {
                        "joints": joint_nodes,
                        "inverseBindMatrices": inverse,
                    }
                )
        material_indices = {}
        primitives = []
        for batch in scene.batches:
            indices = np.asarray(batch.indices, dtype=np.uint32)
            primitive = {
                "attributes": attributes,
                "indices": builder.accessor(indices, "SCALAR", target=34963),
                "mode": 4,
            }
            if batch.part_index is not None:
                primitive["extras"] = {
                    "REasy": {"meshGroupId": int(batch.part_index)}
                }
            name = batch.material_name or "Material"
            if name not in material_indices:
                material_indices[name] = len(document["materials"])
                asset = (material_assets or {}).get(name)
                document["materials"].append(
                    _material_document(document, builder, asset, texture_cache)
                    if asset is not None
                    else {"name": name}
                )
            primitive["material"] = material_indices[name]
            primitives.append(primitive)
        document["meshes"].append({"name": "LOD0", "primitives": primitives})
        node = {"name": "LOD0", "mesh": 0}
        if document["skins"]:
            node["skin"] = 0
        mesh_node = len(document["nodes"])
        document["nodes"].append(node)
        document["scenes"][0]["nodes"].append(mesh_node)
    return document, builder, joint_nodes


def _add_animation(document, builder, motion, rig, profile, joint_nodes):
    from file_handlers.motion.evaluation.binding import bind_motion
    from file_handlers.motion.evaluation.sampling import MotionEvaluator

    binding = bind_motion(motion, rig, profile.joint_binding)
    evaluator = MotionEvaluator(
        binding,
        profile.sampling_policy,
        profile.pose_composition_policy,
    )
    frames = np.arange(0, math.ceil(motion.end_frame) + 1, dtype=np.float32)
    times = frames / np.float32(60.0)
    time_accessor = builder.accessor(times, "SCALAR", bounds=True)
    samples = [
        evaluator.sample_local_frame(float(frame), wrap_looping=False)
        for frame in frames
    ]
    animation = {"name": motion.name, "samplers": [], "channels": []}
    for joint_index in range(len(rig.joints)):
        translations, rotations, scales = [], [], []
        for sample in samples:
            translation, rotation, scale = _converted_trs(
                sample.local_transforms[joint_index]
            )
            translations.append(translation)
            rotations.append(rotation)
            scales.append(scale)
        for path, values in (
            ("translation", translations),
            ("rotation", rotations),
            ("scale", scales),
        ):
            output = builder.accessor(
                np.asarray(values, dtype=np.float32),
                "VEC4" if path == "rotation" else "VEC3",
            )
            sampler = len(animation["samplers"])
            animation["samplers"].append(
                {
                    "input": time_accessor,
                    "output": output,
                    "interpolation": "LINEAR",
                }
            )
            animation["channels"].append(
                {
                    "sampler": sampler,
                    "target": {
                        "node": joint_nodes[joint_index],
                        "path": path,
                    },
                }
            )
    document["animations"].append(animation)


def export_gltf(
    path: str | Path,
    *,
    mesh=None,
    rig=None,
    motion=None,
    evaluation_profile=None,
    material_handler=None,
    resolved_mdf=None,
    material_assets: dict[str, GltfMaterialAsset] | None = None,
    include_auxiliary_groups: bool = False,
) -> Path:
    path = Path(path)
    if mesh is None and rig is None:
        raise ValueError("glTF export needs a mesh or rig")
    if material_assets is None and material_handler is not None and mesh is not None:
        material_assets = resolve_gltf_materials(material_handler, resolved_mdf)
    document, builder, joint_nodes = _base_document(
        mesh,
        rig,
        material_assets,
        include_auxiliary_groups=include_auxiliary_groups,
    )
    if motion is not None:
        if rig is None or evaluation_profile is None:
            raise ValueError("animation export needs a rig and evaluation profile")
        _add_animation(document, builder, motion, rig, evaluation_profile, joint_nodes)
    document["buffers"] = [{"byteLength": len(builder.binary)}]
    document["bufferViews"] = builder.views
    document["accessors"] = builder.accessors
    if not document["materials"]:
        document.pop("materials")
    if not document["images"]:
        document.pop("images")
    if not document["textures"]:
        document.pop("textures")
    if not document["meshes"]:
        document.pop("meshes")
    if not document["skins"]:
        document.pop("skins")
    if not document["animations"]:
        document.pop("animations")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gltf":
        bin_path = path.with_suffix(".bin")
        document["buffers"][0]["uri"] = bin_path.name
        bin_path.write_bytes(builder.binary)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_wots_unreal_bundle(path, material_assets)
        return path
    if path.suffix.lower() != ".glb":
        path = path.with_suffix(".glb")
    json_bytes = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary = bytes(builder.binary) + b"\0" * ((-len(builder.binary)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    glb = bytearray(struct.pack("<4sII", b"glTF", 2, total))
    glb.extend(struct.pack("<I4s", len(json_bytes), b"JSON"))
    glb.extend(json_bytes)
    glb.extend(struct.pack("<I4s", len(binary), b"BIN\0"))
    glb.extend(binary)
    path.write_bytes(glb)
    _write_wots_unreal_bundle(path, material_assets)
    return path


__all__ = [
    "GltfMaterialAsset",
    "GltfTextureImage",
    "export_gltf",
    "resolve_gltf_materials",
]
