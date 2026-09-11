from __future__ import annotations

from contextlib import suppress
from ctypes import c_void_p
from dataclasses import dataclass

import numpy as np
from OpenGL.arrays import vbo
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_MAX_VERTEX_ATTRIBS,
    GL_MAX_VERTEX_UNIFORM_COMPONENTS,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBufferSubData,
    glDeleteProgram,
    glDisableVertexAttribArray,
    glEnableVertexAttribArray,
    glGetAttribLocation,
    glGetIntegerv,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform4f,
    glUniform4fv,
    glUseProgram,
    glVertexAttrib2f,
    glVertexAttrib3f,
    glVertexAttrib4f,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from .scene_model import SceneSkinningBinding
from .wots_material import WOTS_FRAGMENT_SHADER, WOTS_MATERIAL_UNIFORM_NAMES


MAX_SKIN_INFLUENCES = 16
_COMPONENTS = "xyzw"
_GENERIC_VERTEX_UNIFORM_COMPONENTS = 9


class GpuSkinningError(ValueError):
    pass


@dataclass(slots=True)
class _State:
    binding: SceneSkinningBinding
    joints: np.ndarray
    palette_joints: np.ndarray
    positions: np.ndarray
    normals: np.ndarray | None
    palette_rows: np.ndarray
    source_revision: int = 0
    palette_revision: int = 0

    @property
    def palette_size(self) -> int:
        return len(self.palette_joints)


@dataclass(slots=True)
class _Source:
    state: _State
    positions: object
    normals: object | None
    tangents: object | None
    influences: object
    source_revision: int = -1


@dataclass(frozen=True, slots=True)
class _Inputs:
    attributes: tuple[int, ...]
    palette: int
    material: tuple[int, ...] | None


class GpuSkinningDeformer:
    """Upload semantic skinning once and bind it directly for scene draws."""

    def __init__(self) -> None:
        self._states: dict[str, _State] = {}
        self._sources: dict[str, _Source] = {}
        self._programs: dict[tuple[int, int], int] = {}
        self._inputs: dict[tuple[int, int, bool], _Inputs] = {}
        self._shader_variants: dict[tuple[int, int], tuple[str, str]] = {}
        self._palette_bindings: dict[int, tuple[str, int]] = {}
        self._bound: tuple[int, ...] | None = None

    @property
    def keys(self) -> set[str]:
        return set(self._states)

    @staticmethod
    def binding_requirements(binding: SceneSkinningBinding) -> tuple[int, int]:
        """Return the vertex-attribute and matrix-palette requirements."""
        active = binding.weights > 0.0
        palette_size = len(np.unique(binding.joint_indices[active]))
        return 6 + binding.group_count * 2, palette_size

    def supports_binding(self, binding: SceneSkinningBinding) -> bool:
        """Whether the current OpenGL context can draw one skin binding."""
        required_attributes, required_palette = self.binding_requirements(binding)
        palette_limit = max(
            0,
            (
                self._integer_limit(GL_MAX_VERTEX_UNIFORM_COMPONENTS)
                - _GENERIC_VERTEX_UNIFORM_COMPONENTS
            )
            // 12,
        )
        return (
            self._integer_limit(GL_MAX_VERTEX_ATTRIBS) >= required_attributes
            and palette_limit >= required_palette
        )

    def vertex_count(self, key: str) -> int:
        return len(self._state(key).binding.positions)

    def set_binding(self, key: str, binding: SceneSkinningBinding) -> None:
        influence_count = binding.joint_indices.shape[1]
        if influence_count > MAX_SKIN_INFLUENCES:
            raise GpuSkinningError(
                f"GPU skinning supports at most {MAX_SKIN_INFLUENCES} "
                f"influences; got {influence_count}"
            )
        active = binding.weights > 0.0
        palette_joints = np.unique(binding.joint_indices[active])
        dense = np.zeros(int(binding.joint_indices.max()) + 1, dtype=np.uint16)
        dense[palette_joints] = np.arange(len(palette_joints), dtype=np.uint16)
        identity = np.tile(np.eye(4, dtype=np.float32), (len(palette_joints), 1, 1))
        self._states[str(key)] = _State(
            binding,
            dense[binding.joint_indices],
            palette_joints,
            binding.positions,
            binding.normals,
            self._affine_rows(identity),
        )
        self._palette_bindings.clear()

    def update_source(
        self,
        key: str,
        positions: np.ndarray,
        normals: np.ndarray | None,
    ) -> None:
        state = self._state(key)
        positions = self._vec3(positions)
        normals = self._vec3(normals) if normals is not None else None
        if len(positions) != len(state.binding.positions):
            raise GpuSkinningError("skinning source vertex count changed")
        if (normals is None) != (state.binding.normals is None):
            raise GpuSkinningError("skinning source normal layout changed")
        if normals is not None and len(normals) != len(positions):
            raise GpuSkinningError("skinning source normals do not match positions")
        if state.positions is positions and state.normals is normals:
            return
        if not np.isfinite(positions).all() or (
            normals is not None and not np.isfinite(normals).all()
        ):
            raise GpuSkinningError("skinning source contains a nonfinite value")
        state.positions, state.normals = positions, normals
        state.source_revision += 1

    def update_palette(self, key: str, matrices: np.ndarray) -> None:
        state = self._state(key)
        matrices = np.asarray(matrices, dtype=np.float32).reshape(-1, 4, 4)
        if not len(matrices) or not np.isfinite(matrices).all():
            raise GpuSkinningError("skin matrix palette is empty or non-finite")
        if int(state.palette_joints[-1]) >= len(matrices):
            raise GpuSkinningError(
                f"skinned mesh {key!r} references a joint outside its palette"
            )
        state.palette_rows = self._affine_rows(matrices[state.palette_joints])
        state.palette_revision += 1

    def remove(self, keys: set[str] | None = None) -> set[str]:
        removed = set(self._states) if keys is None else self.keys & set(keys)
        for key in removed:
            self._states.pop(key, None)
        if removed:
            self._palette_bindings.clear()
        return removed

    def clear(self) -> None:
        self._states.clear()
        self._palette_bindings.clear()

    def dispose_gl(self) -> None:
        self.unbind()
        for source in self._sources.values():
            self._dispose_source(source)
        for program in self._programs.values():
            with suppress(Exception):
                glDeleteProgram(program)
        self._sources.clear()
        self._programs.clear()
        self._inputs.clear()
        self._shader_variants.clear()
        self._palette_bindings.clear()

    def shader_variant(self, key: str) -> tuple[str, str]:
        state = self._state(key)
        variant = state.binding.group_count, state.palette_size
        return self._shader_variants.setdefault(
            variant,
            (
                f"skinning-{variant[0]}-{variant[1]}",
                _vertex_shader(*variant, riglogic=True),
            ),
        )

    def prepare(self) -> None:
        self._sync_sources()
        if not self._states:
            return
        required_attributes = 6 + max(
            state.binding.group_count * 2 for state in self._states.values()
        )
        palette_limit = max(
            0,
            (
                self._integer_limit(GL_MAX_VERTEX_UNIFORM_COMPONENTS)
                - _GENERIC_VERTEX_UNIFORM_COMPONENTS
            )
            // 12,
        )
        required_palette = max(state.palette_size for state in self._states.values())
        problems = []
        if self._integer_limit(GL_MAX_VERTEX_ATTRIBS) < required_attributes:
            problems.append(f"{required_attributes} vertex attributes")
        if palette_limit < required_palette:
            problems.append(f"{required_palette} skin matrices (limit {palette_limit})")
        if problems:
            raise GpuSkinningError(
                "OpenGL cannot draw skinned meshes: " + ", ".join(problems)
            )

    def bind(
        self,
        key: str,
        *,
        uvs_vbo=None,
        uvs1_vbo=None,
        colors_vbo=None,
        vertex_offset: int = 0,
        program: int = 0,
        tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        ambient: float = 1.0,
        diffuse: float = 0.0,
        exposure: float = 1.0,
        gamma: float = 1.0,
        lit: bool = False,
        textured: bool = False,
        texture_id: int = 0,
        nrro_texture_id: int = 0,
        normal_texture_id: int = 0,
        rcto_texture_id: int = 0,
        alpha_texture_id: int = 0,
        hair_flow_texture_id: int = 0,
        hair_hss_texture_id: int = 0,
        detail_nrrc_texture_id: int = 0,
        detail_mask_texture_id: int = 0,
        stcm_texture_id: int = 0,
        emissive_texture_id: int = 0,
        second_alpha_texture_id: int = 0,
        wots_material: bool = False,
        roughness_scale: float = 1.0,
        occlusion_scale: float = 1.0,
        alpha_adjust: float = 1.0,
        alpha_threshold: float = 0.5,
        alpha_test: bool = False,
        hair_material: bool = False,
        material_family: int = 0,
        use_secondary_uv: bool = False,
        use_separate_alpha: float = 0.0,
        use_flow_map: float = 0.0,
        secondary_specular_intensity: float = 1.0,
        primary_spec_sharpness: float = 50.0,
        secondary_spec_sharpness: float = 20.0,
        primary_specular_shift_offset: float = 0.1,
        secondary_specular_shift_offset: float = 0.1,
        hair_height_depth: float = 0.0,
        specular: float = 0.5,
        primary_specular_level: float = 0.023529,
        ao_exp: float = 0.0,
        sss_scale: float = 0.0,
        use_detail: bool = False,
        detail_tiling: float = 1.0,
        normal_blend_rate: float = 1.0,
        roughness_blend_rate: float = 0.0,
        cavity_blend_rate: float = 0.0,
        emissive_intensity: float = 0.0,
        translucent_scale: float = 0.0,
        face_uv_scale: float = 1.0,
        vertex_colors: bool = True,
    ) -> None:
        state = self._state(key)
        source = self._sources.get(str(key))
        if source is None or source.state is not state:
            self.prepare()
            source = self._sources[str(key)]
        generic = not program
        program = program or self._program(
            state.binding.group_count,
            state.palette_size,
        )
        inputs = self._program_inputs(program, state.binding.group_count, generic)
        self._upload_source(source)
        glUseProgram(program)
        self._upload_palette(str(key), program, inputs.palette, state)
        self._bind_attributes(
            source,
            inputs.attributes,
            uvs_vbo,
            uvs1_vbo,
            colors_vbo if vertex_colors else None,
            int(vertex_offset),
            generic=generic,
        )
        if inputs.material is not None:
            for unit, texture in enumerate((
                texture_id if textured else 0,
                nrro_texture_id,
                normal_texture_id,
                rcto_texture_id,
                alpha_texture_id,
                hair_flow_texture_id,
                hair_hss_texture_id,
                detail_nrrc_texture_id,
                detail_mask_texture_id,
                stcm_texture_id,
                emissive_texture_id,
                second_alpha_texture_id,
            )):
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, int(texture or 0))
            glActiveTexture(GL_TEXTURE0)
            glUniform4f(inputs.material[0], *tint)
            glUniform1f(inputs.material[1], float(ambient))
            glUniform1f(inputs.material[2], float(diffuse))
            glUniform1f(inputs.material[3], float(exposure))
            glUniform1f(inputs.material[4], float(gamma))
            glUniform1i(inputs.material[5], int(lit))
            glUniform1i(inputs.material[6], 0)
            glUniform1i(inputs.material[7], int(textured))
            glUniform1i(inputs.material[8], 1)
            glUniform1i(inputs.material[9], int(bool(nrro_texture_id)))
            glUniform1i(inputs.material[10], 2)
            glUniform1i(inputs.material[11], int(bool(normal_texture_id)))
            glUniform1i(inputs.material[12], 3)
            glUniform1i(inputs.material[13], int(bool(rcto_texture_id)))
            glUniform1i(inputs.material[14], 4)
            glUniform1i(inputs.material[15], int(bool(alpha_texture_id)))
            glUniform1i(inputs.material[16], 5)
            glUniform1i(inputs.material[17], int(bool(hair_flow_texture_id)))
            glUniform1i(inputs.material[18], 6)
            glUniform1i(inputs.material[19], int(bool(hair_hss_texture_id)))
            glUniform1i(inputs.material[20], 7)
            glUniform1i(inputs.material[21], int(bool(detail_nrrc_texture_id)))
            glUniform1i(inputs.material[22], 8)
            glUniform1i(inputs.material[23], int(bool(detail_mask_texture_id)))
            glUniform1i(inputs.material[24], 9)
            glUniform1i(inputs.material[25], int(bool(stcm_texture_id)))
            glUniform1i(inputs.material[26], 10)
            glUniform1i(inputs.material[27], int(bool(emissive_texture_id)))
            glUniform1i(inputs.material[28], 11)
            glUniform1i(inputs.material[29], int(bool(second_alpha_texture_id)))
            glUniform1i(inputs.material[30], int(wots_material))
            glUniform1i(inputs.material[31], int(hair_material))
            glUniform1i(inputs.material[32], int(material_family))
            glUniform1i(inputs.material[33], int(use_secondary_uv))
            glUniform1f(inputs.material[34], float(roughness_scale))
            glUniform1f(inputs.material[35], float(occlusion_scale))
            glUniform1f(inputs.material[36], float(alpha_adjust))
            glUniform1f(inputs.material[37], float(alpha_threshold))
            glUniform1i(inputs.material[38], int(alpha_test))
            for index, value in enumerate((
                use_separate_alpha,
                use_flow_map,
                secondary_specular_intensity,
                primary_spec_sharpness,
                secondary_spec_sharpness,
                primary_specular_shift_offset,
                secondary_specular_shift_offset,
                hair_height_depth,
                specular,
                primary_specular_level,
                ao_exp,
                sss_scale,
            ), start=39):
                glUniform1f(inputs.material[index], float(value))
            glUniform1i(inputs.material[51], int(use_detail))
            for index, value in enumerate((
                detail_tiling,
                normal_blend_rate,
                roughness_blend_rate,
                cavity_blend_rate,
                emissive_intensity,
                translucent_scale,
            ), start=52):
                glUniform1f(inputs.material[index], float(value))
            glUniform1f(inputs.material[58], float(face_uv_scale))
        self._bound = inputs.attributes

    def unbind(self) -> None:
        if self._bound is None:
            return
        for location in self._bound:
            glDisableVertexAttribArray(location)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        for unit in range(12):
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glUseProgram(0)
        self._bound = None

    def _state(self, key: str) -> _State:
        try:
            return self._states[str(key)]
        except KeyError as exc:
            raise GpuSkinningError(f"skinned mesh {key!r} is not registered") from exc

    def _sync_sources(self) -> None:
        for key in set(self._sources) - set(self._states):
            self._dispose_source(self._sources.pop(key))
        for key, state in self._states.items():
            source = self._sources.get(key)
            if source is None or source.state is not state:
                if source is not None:
                    self._dispose_source(source)
                self._sources[key] = self._build_source(state)

    def _build_source(self, state: _State) -> _Source:
        groups = state.binding.group_count
        width = groups * 4
        joints = np.zeros((len(state.joints), width), dtype=np.float32)
        weights = np.zeros_like(joints)
        joints[:, : state.joints.shape[1]] = state.joints
        weights[:, : state.binding.weights.shape[1]] = state.binding.weights
        influences = np.concatenate(
            (joints.reshape(-1, groups, 4), weights.reshape(-1, groups, 4)),
            axis=2,
        ).reshape(len(joints), -1)
        return _Source(
            state,
            self._array_vbo(state.positions, dynamic=True),
            self._array_vbo(state.normals, dynamic=True) if state.normals is not None else None,
            self._array_vbo(state.binding.tangents) if state.binding.tangents is not None else None,
            self._array_vbo(influences),
        )

    def _bind_attributes(
        self,
        source: _Source,
        locations: tuple[int, ...],
        uvs_vbo,
        uvs1_vbo,
        colors_vbo,
        vertex_offset: int,
        *,
        generic: bool,
    ) -> None:
        self._bind_attribute(source.positions, locations[0], 3)
        if source.normals is None:
            glDisableVertexAttribArray(locations[1])
            glVertexAttrib3f(locations[1], 0.0, 0.0, 1.0)
        else:
            self._bind_attribute(source.normals, locations[1], 3)
        for handle, location, width, offset, default in (
            (uvs_vbo, locations[2], 2, vertex_offset * 8, (0.0, 0.0)),
            (colors_vbo, locations[3], 4, vertex_offset * 16, (1.0,) * 4),
        ):
            if handle is None:
                glDisableVertexAttribArray(location)
                (glVertexAttrib2f if width == 2 else glVertexAttrib4f)(location, *default)
            else:
                self._bind_attribute(handle, location, width, offset=offset)
        influence_base = 4
        if generic:
            tangent_location = locations[4]
            if source.tangents is None:
                glDisableVertexAttribArray(tangent_location)
                glVertexAttrib4f(tangent_location, 0.0, 0.0, 0.0, 1.0)
            else:
                self._bind_attribute(source.tangents, tangent_location, 4)
            uv1_location = locations[5]
            if uvs1_vbo is None:
                glDisableVertexAttribArray(uv1_location)
                glVertexAttrib2f(uv1_location, 0.0, 0.0)
            else:
                self._bind_attribute(
                    uvs1_vbo,
                    uv1_location,
                    2,
                    offset=vertex_offset * 8,
                )
            influence_base = 6
        source.influences.bind()
        stride = source.state.binding.group_count * 32
        for group in range(source.state.binding.group_count):
            for lane, offset in enumerate((group * 32, group * 32 + 16)):
                location = locations[influence_base + group * 2 + lane]
                glEnableVertexAttribArray(location)
                glVertexAttribPointer(location, 4, GL_FLOAT, False, stride, c_void_p(offset))

    @staticmethod
    def _bind_attribute(handle, location: int, width: int, *, offset: int = 0) -> None:
        handle.bind()
        glEnableVertexAttribArray(location)
        glVertexAttribPointer(location, width, GL_FLOAT, False, 0, c_void_p(offset))

    @staticmethod
    def _upload_source(source: _Source) -> None:
        state = source.state
        if source.source_revision == state.source_revision:
            return
        for handle, data in ((source.positions, state.positions), (source.normals, state.normals)):
            if handle is not None and data is not None:
                handle.bind()
                glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        source.source_revision = state.source_revision

    def _upload_palette(
        self,
        key: str,
        program: int,
        location: int,
        state: _State,
    ) -> None:
        revision = key, state.palette_revision
        if self._palette_bindings.get(program) == revision:
            return
        glUniform4fv(location, state.palette_size * 3, state.palette_rows)
        self._palette_bindings[program] = revision

    def _program(self, groups: int, palette_size: int) -> int:
        key = groups, palette_size
        program = self._programs.get(key)
        if program is None:
            program = compileProgram(
                compileShader(
                    _vertex_shader(groups, palette_size, riglogic=False),
                    GL_VERTEX_SHADER,
                ),
                compileShader(_GENERIC_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
            )
            self._programs[key] = program
        return program

    def _program_inputs(self, program: int, groups: int, generic: bool) -> _Inputs:
        key = int(program), groups, generic
        inputs = self._inputs.get(key)
        if inputs is not None:
            return inputs
        names = ["a_position", "a_normal", "a_uv", "a_color"]
        if generic:
            names.extend(("a_tangent", "a_uv1"))
        names += [
            name
            for group in range(groups)
            for name in (f"a_joints{group}", f"a_weights{group}")
        ]
        attributes = tuple(int(glGetAttribLocation(program, name)) for name in names)
        palette = int(glGetUniformLocation(program, "u_palette[0]"))
        material = (
            tuple(
                int(glGetUniformLocation(program, name))
                for name in WOTS_MATERIAL_UNIFORM_NAMES
            )
            if generic
            else None
        )
        if min((*attributes, palette, *(material or ()))) < 0:
            raise GpuSkinningError("GPU skinning shader omitted a required input")
        inputs = _Inputs(attributes, palette, material)
        self._inputs[key] = inputs
        return inputs

    @staticmethod
    def _vec3(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.ascontiguousarray(
            array if array.ndim == 2 and array.shape[1] == 3 else array.reshape(-1, 3)
        )

    @staticmethod
    def _affine_rows(matrices: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.transpose(matrices[:, :, :3], (0, 2, 1)))

    @staticmethod
    def _array_vbo(data: np.ndarray, *, dynamic: bool = False):
        return vbo.VBO(
            np.ascontiguousarray(data),
            usage=GL_DYNAMIC_DRAW if dynamic else GL_STATIC_DRAW,
            target=GL_ARRAY_BUFFER,
        )

    @staticmethod
    def _integer_limit(parameter: int) -> int:
        return int(np.asarray(glGetIntegerv(parameter)).reshape(-1)[0])

    @staticmethod
    def _dispose_source(source: _Source) -> None:
        for handle in (
            source.positions,
            source.normals,
            source.tangents,
            source.influences,
        ):
            if handle is not None:
                with suppress(Exception):
                    handle.delete()


def _vertex_shader(groups: int, palette_size: int, *, riglogic: bool) -> str:
    tangent_attribute = "" if riglogic else "attribute vec4 a_tangent;"
    uv1_attribute = "" if riglogic else "attribute vec2 a_uv1;"
    attributes = "\n".join(
        f"attribute vec4 a_joints{i};\nattribute vec4 a_weights{i};"
        for i in range(groups)
    )
    influences = "\n".join(
        (
            f"    applyInfluence(position, normal, a_joints{i}.{c}, a_weights{i}.{c});"
            if riglogic
            else f"    applyInfluence(position, normal, tangent, a_joints{i}.{c}, a_weights{i}.{c});"
        )
        for i in range(groups)
        for c in _COMPONENTS
    )
    outputs = (
        """
varying vec2 v_uv;
varying vec3 v_eye_position;
varying vec3 v_eye_normal;
varying vec4 v_color;
"""
        if riglogic
        else """
uniform vec4 u_tint;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_exposure;
uniform float u_gamma;
uniform bool u_lit;
varying vec2 v_uv;
varying vec2 v_uv1;
varying vec3 v_eye_position;
varying vec3 v_eye_normal;
varying vec4 v_eye_tangent;
varying vec4 v_color;
"""
    )
    result = (
        """
    v_uv = a_uv;
    v_eye_position = eye.xyz;
    v_eye_normal = eyeNormal;
    v_color = a_color;
"""
        if riglogic
        else """
    v_uv = a_uv;
    v_uv1 = a_uv1;
    v_eye_position = eye.xyz;
    v_eye_normal = eyeNormal;
    vec3 eyeTangent = gl_NormalMatrix * tangent;
    v_eye_tangent = vec4(
        dot(eyeTangent, eyeTangent) > 0.000000000001
            ? normalize(eyeTangent)
            : vec3(0.0),
        a_tangent.w
    );
    v_color = a_color;
"""
    )
    tangent_parameter = "" if riglogic else "    inout vec3 tangent,\n"
    tangent_skin = "" if riglogic else """
    tangent += vec3(
        dot(u_palette[base].xyz, a_tangent.xyz),
        dot(u_palette[base + 1].xyz, a_tangent.xyz),
        dot(u_palette[base + 2].xyz, a_tangent.xyz)
    ) * weight;
"""
    tangent_initial = "" if riglogic else "    vec3 tangent = vec3(0.0);"
    tangent_normalize = "" if riglogic else """
    tangent -= normal * dot(normal, tangent);
    if (dot(tangent, tangent) > 0.000000000001) tangent = normalize(tangent);
"""
    return f"""
#version 120

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_uv;
attribute vec4 a_color;
{tangent_attribute}
{uv1_attribute}
{attributes}
uniform vec4 u_palette[{palette_size * 3}];
{outputs}

void applyInfluence(
    inout vec3 position,
    inout vec3 normal,
{tangent_parameter}    float jointIndex,
    float weight
) {{
    if (weight <= 0.0) return;
    int base = int(floor(jointIndex + 0.5)) * 3;
    vec4 source = vec4(a_position, 1.0);
    position += vec3(
        dot(u_palette[base], source),
        dot(u_palette[base + 1], source),
        dot(u_palette[base + 2], source)
    ) * weight;
    normal += vec3(
        dot(u_palette[base].xyz, a_normal),
        dot(u_palette[base + 1].xyz, a_normal),
        dot(u_palette[base + 2].xyz, a_normal)
    ) * weight;
{tangent_skin}
}}

void main() {{
    vec3 position = vec3(0.0);
    vec3 normal = vec3(0.0);
{tangent_initial}
{influences}
    if (dot(normal, normal) > 0.000000000001) normal = normalize(normal);
{tangent_normalize}
    vec4 eye = gl_ModelViewMatrix * vec4(position, 1.0);
    vec3 eyeNormal = normalize(gl_NormalMatrix * normal);
    gl_Position = gl_ProjectionMatrix * eye;
{result}
}}
"""


_GENERIC_FRAGMENT_SHADER = WOTS_FRAGMENT_SHADER
