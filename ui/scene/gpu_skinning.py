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
        return 4 + binding.group_count * 2, palette_size

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
        required_attributes = 4 + max(
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
        wots_material: bool = False,
        roughness_scale: float = 1.0,
        occlusion_scale: float = 1.0,
        alpha_adjust: float = 1.0,
        alpha_threshold: float = 0.5,
        alpha_test: bool = False,
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
            colors_vbo if vertex_colors else None,
            int(vertex_offset),
        )
        if inputs.material is not None:
            for unit, texture in enumerate((
                texture_id if textured else 0,
                nrro_texture_id,
                normal_texture_id,
                rcto_texture_id,
                alpha_texture_id,
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
            glUniform1i(inputs.material[16], int(wots_material))
            glUniform1f(inputs.material[17], float(roughness_scale))
            glUniform1f(inputs.material[18], float(occlusion_scale))
            glUniform1f(inputs.material[19], float(alpha_adjust))
            glUniform1f(inputs.material[20], float(alpha_threshold))
            glUniform1i(inputs.material[21], int(alpha_test))
        self._bound = inputs.attributes

    def unbind(self) -> None:
        if self._bound is None:
            return
        for location in self._bound:
            glDisableVertexAttribArray(location)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        for unit in range(5):
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
            self._array_vbo(influences),
        )

    def _bind_attributes(
        self,
        source: _Source,
        locations: tuple[int, ...],
        uvs_vbo,
        colors_vbo,
        vertex_offset: int,
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
        source.influences.bind()
        stride = source.state.binding.group_count * 32
        for group in range(source.state.binding.group_count):
            for lane, offset in enumerate((group * 32, group * 32 + 16)):
                location = locations[4 + group * 2 + lane]
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
        names = ["a_position", "a_normal", "a_uv", "a_color"] + [
            name
            for group in range(groups)
            for name in (f"a_joints{group}", f"a_weights{group}")
        ]
        attributes = tuple(int(glGetAttribLocation(program, name)) for name in names)
        palette = int(glGetUniformLocation(program, "u_palette[0]"))
        material = (
            tuple(
                int(glGetUniformLocation(program, name))
                for name in (
                    "u_tint", "u_ambient", "u_diffuse", "u_exposure",
                    "u_gamma", "u_lit", "u_texture", "u_textured",
                    "u_nrro_texture", "u_has_nrro", "u_normal_texture",
                    "u_has_normal", "u_rcto_texture", "u_has_rcto",
                    "u_alpha_texture", "u_has_alpha", "u_wots_material", "u_roughness_scale",
                    "u_occlusion_scale", "u_alpha_adjust",
                    "u_alpha_threshold", "u_alpha_test",
                )
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
        for handle in (source.positions, source.normals, source.influences):
            if handle is not None:
                with suppress(Exception):
                    handle.delete()


def _vertex_shader(groups: int, palette_size: int, *, riglogic: bool) -> str:
    attributes = "\n".join(
        f"attribute vec4 a_joints{i};\nattribute vec4 a_weights{i};"
        for i in range(groups)
    )
    influences = "\n".join(
        f"    applyInfluence(position, normal, a_joints{i}.{c}, a_weights{i}.{c});"
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
varying vec3 v_eye_position;
varying vec3 v_eye_normal;
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
    v_eye_position = eye.xyz;
    v_eye_normal = eyeNormal;
    v_color = a_color;
"""
    )
    return f"""
#version 120

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_uv;
attribute vec4 a_color;
{attributes}
uniform vec4 u_palette[{palette_size * 3}];
{outputs}

void applyInfluence(
    inout vec3 position,
    inout vec3 normal,
    float jointIndex,
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
}}

void main() {{
    vec3 position = vec3(0.0);
    vec3 normal = vec3(0.0);
{influences}
    if (dot(normal, normal) > 0.000000000001) normal = normalize(normal);
    vec4 eye = gl_ModelViewMatrix * vec4(position, 1.0);
    vec3 eyeNormal = normalize(gl_NormalMatrix * normal);
    gl_Position = gl_ProjectionMatrix * eye;
{result}
}}
"""


_GENERIC_FRAGMENT_SHADER = """
#version 120

uniform sampler2D u_texture;
uniform bool u_textured;
uniform sampler2D u_nrro_texture;
uniform bool u_has_nrro;
uniform sampler2D u_normal_texture;
uniform bool u_has_normal;
uniform sampler2D u_rcto_texture;
uniform bool u_has_rcto;
uniform sampler2D u_alpha_texture;
uniform bool u_has_alpha;
uniform vec4 u_tint;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_exposure;
uniform float u_gamma;
uniform bool u_lit;
uniform bool u_wots_material;
uniform float u_roughness_scale;
uniform float u_occlusion_scale;
uniform float u_alpha_adjust;
uniform float u_alpha_threshold;
uniform bool u_alpha_test;

varying vec2 v_uv;
varying vec3 v_eye_position;
varying vec3 v_eye_normal;
varying vec4 v_color;

vec3 mappedNormal(vec3 tangentNormal) {
    vec3 normal = normalize(v_eye_normal);
    vec3 positionX = dFdx(v_eye_position);
    vec3 positionY = dFdy(v_eye_position);
    vec2 uvX = dFdx(v_uv);
    vec2 uvY = dFdy(v_uv);
    vec3 positionYPerp = cross(positionY, normal);
    vec3 positionXPerp = cross(normal, positionX);
    vec3 tangent = positionYPerp * uvX.x + positionXPerp * uvY.x;
    vec3 bitangent = positionYPerp * uvX.y + positionXPerp * uvY.y;
    float scale = max(dot(tangent, tangent), dot(bitangent, bitangent));
    if (scale < 1e-12) return normal;
    float inverseScale = inversesqrt(scale);
    return normalize(
        tangent * (tangentNormal.x * inverseScale)
        + bitangent * (tangentNormal.y * inverseScale)
        + normal * tangentNormal.z
    );
}

void main() {
    vec4 texel = u_textured ? texture2D(u_texture, v_uv) : vec4(1.0);
    vec4 surface = texel * v_color * u_tint;
    vec3 normal = normalize(v_eye_normal);
    float roughness = 1.0;
    float occlusion = 1.0;
    if (u_wots_material && u_has_nrro) {
        vec4 nrro = texture2D(u_nrro_texture, v_uv);
        vec2 normalXY = nrro.ag * 2.0 - 1.0;
        vec3 tangentNormal = vec3(
            normalXY,
            sqrt(max(1.0 - dot(normalXY, normalXY), 0.0))
        );
        normal = mappedNormal(normalize(tangentNormal));
        roughness = clamp(nrro.r * max(u_roughness_scale, 0.0), 0.04, 1.0);
        occlusion = clamp(
            mix(1.0, nrro.b, max(u_occlusion_scale, 0.0)),
            0.0,
            1.0
        );
    } else if (u_wots_material && u_has_normal) {
        normal = mappedNormal(
            normalize(texture2D(u_normal_texture, v_uv).rgb * 2.0 - 1.0)
        );
    }
    if (u_wots_material && u_has_rcto) {
        vec4 rcto = texture2D(u_rcto_texture, v_uv);
        roughness = clamp(rcto.r * max(u_roughness_scale, 0.0), 0.04, 1.0);
        occlusion = clamp(
            mix(1.0, rcto.a, max(u_occlusion_scale, 0.0)),
            0.0,
            1.0
        );
    }
    float alpha = surface.a;
    if (u_wots_material) {
        float alphaMask = u_has_alpha
            ? texture2D(u_alpha_texture, v_uv).r
            : texel.a;
        alpha = v_color.a * u_tint.a * pow(
            clamp(alphaMask, 0.0, 1.0),
            max(u_alpha_adjust, 0.0001)
        );
        if (u_alpha_test && alpha < u_alpha_threshold) discard;
    }
    float hemisphere = 0.35 + 0.65 * clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 keyDirection = normalize(vec3(0.45, 0.75, 0.55));
    float key = max(dot(normal, keyDirection), 0.0);
    float fill = max(dot(normal, normalize(vec3(-0.70, 0.30, 0.45))), 0.0);
    float rim = pow(max(dot(normal, normalize(vec3(0.10, 0.35, -0.95))), 0.0), 2.0);
    float light = u_ambient * hemisphere * occlusion
        + u_diffuse * (0.72 * key + 0.28 * fill + 0.35 * rim);
    vec3 litSurface = surface.rgb * light;
    if (u_wots_material) {
        vec3 viewDirection = normalize(-v_eye_position);
        vec3 halfDirection = normalize(keyDirection + viewDirection);
        float exponent = mix(96.0, 4.0, roughness);
        float strength = mix(0.42, 0.04, roughness);
        float specular = pow(max(dot(normal, halfDirection), 0.0), exponent);
        litSurface += vec3(specular * strength * u_diffuse * occlusion);
    }
    vec3 rgb = u_lit
        ? pow(
            max(litSurface * max(u_exposure, 0.0), vec3(0.0)),
            vec3(1.0 / max(u_gamma, 0.01))
        )
        : surface.rgb;
    gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), alpha);
}
"""
