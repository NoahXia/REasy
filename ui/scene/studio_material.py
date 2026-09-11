from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from OpenGL.GL import (
    GL_FRAGMENT_SHADER,
    GL_FLOAT,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindTexture,
    glDeleteProgram,
    glDisableVertexAttribArray,
    glEnableVertexAttribArray,
    glGetAttribLocation,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform4f,
    glUseProgram,
    glVertexAttrib2f,
    glVertexAttrib4f,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from .wots_material import WOTS_FRAGMENT_SHADER


@dataclass(slots=True)
class _Program:
    handle: int
    texture: int
    textured: int
    nrro_texture: int
    has_nrro: int
    normal_texture: int
    has_normal: int
    rcto_texture: int
    has_rcto: int
    alpha_texture: int
    has_alpha: int
    hair_flow_texture: int
    has_hair_flow: int
    hair_hss_texture: int
    has_hair_hss: int
    detail_nrrc_texture: int
    has_detail_nrrc: int
    detail_mask_texture: int
    has_detail_mask: int
    stcm_texture: int
    has_stcm: int
    emissive_texture: int
    has_emissive: int
    second_alpha_texture: int
    has_second_alpha: int
    tint: int
    ambient: int
    diffuse: int
    exposure: int
    gamma: int
    lit: int
    wots_material: int
    roughness_scale: int
    occlusion_scale: int
    alpha_adjust: int
    alpha_threshold: int
    alpha_test: int
    hair_material: int
    material_family: int
    use_secondary_uv: int
    use_separate_alpha: int
    use_flow_map: int
    secondary_specular_intensity: int
    primary_spec_sharpness: int
    secondary_spec_sharpness: int
    primary_specular_shift_offset: int
    secondary_specular_shift_offset: int
    hair_height_depth: int
    specular: int
    primary_specular_level: int
    ao_exp: int
    sss_scale: int
    use_detail: int
    detail_tiling: int
    normal_blend_rate: int
    roughness_blend_rate: int
    cavity_blend_rate: int
    emissive_intensity: int
    translucent_scale: int
    face_uv_scale: int
    uv1: int
    tangent: int


class StudioMaterialRenderer:
    """Apply preview lighting and display controls after texture sampling."""

    def __init__(self) -> None:
        self._program: _Program | None = None

    def bind(
        self,
        *,
        tint: tuple[float, float, float, float],
        ambient: float,
        diffuse: float,
        exposure: float,
        gamma: float,
        textured: bool,
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
        uvs1_vbo=None,
        tangents_vbo=None,
        lit: bool = True,
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
    ) -> int:
        program = self._get_program()
        glUseProgram(program.handle)
        for unit, (location, texture) in enumerate((
            (program.texture, texture_id if textured else 0),
            (program.nrro_texture, nrro_texture_id),
            (program.normal_texture, normal_texture_id),
            (program.rcto_texture, rcto_texture_id),
            (program.alpha_texture, alpha_texture_id),
            (program.hair_flow_texture, hair_flow_texture_id),
            (program.hair_hss_texture, hair_hss_texture_id),
            (program.detail_nrrc_texture, detail_nrrc_texture_id),
            (program.detail_mask_texture, detail_mask_texture_id),
            (program.stcm_texture, stcm_texture_id),
            (program.emissive_texture, emissive_texture_id),
            (program.second_alpha_texture, second_alpha_texture_id),
        )):
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, int(texture or 0))
            glUniform1i(location, unit)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(program.textured, int(textured))
        glUniform1i(program.has_nrro, int(bool(nrro_texture_id)))
        glUniform1i(program.has_normal, int(bool(normal_texture_id)))
        glUniform1i(program.has_rcto, int(bool(rcto_texture_id)))
        glUniform1i(program.has_alpha, int(bool(alpha_texture_id)))
        glUniform1i(program.has_hair_flow, int(bool(hair_flow_texture_id)))
        glUniform1i(program.has_hair_hss, int(bool(hair_hss_texture_id)))
        glUniform1i(program.has_detail_nrrc, int(bool(detail_nrrc_texture_id)))
        glUniform1i(program.has_detail_mask, int(bool(detail_mask_texture_id)))
        glUniform1i(program.has_stcm, int(bool(stcm_texture_id)))
        glUniform1i(program.has_emissive, int(bool(emissive_texture_id)))
        glUniform1i(program.has_second_alpha, int(bool(second_alpha_texture_id)))
        glUniform4f(program.tint, *tint)
        glUniform1f(program.ambient, float(ambient))
        glUniform1f(program.diffuse, float(diffuse))
        glUniform1f(program.exposure, float(exposure))
        glUniform1f(program.gamma, float(gamma))
        glUniform1i(program.lit, int(lit))
        glUniform1i(program.wots_material, int(wots_material))
        glUniform1f(program.roughness_scale, float(roughness_scale))
        glUniform1f(program.occlusion_scale, float(occlusion_scale))
        glUniform1f(program.alpha_adjust, float(alpha_adjust))
        glUniform1f(program.alpha_threshold, float(alpha_threshold))
        glUniform1i(program.alpha_test, int(alpha_test))
        glUniform1i(program.hair_material, int(hair_material))
        glUniform1i(program.material_family, int(material_family))
        glUniform1i(program.use_secondary_uv, int(use_secondary_uv))
        glUniform1f(program.use_separate_alpha, float(use_separate_alpha))
        glUniform1f(program.use_flow_map, float(use_flow_map))
        glUniform1f(
            program.secondary_specular_intensity,
            float(secondary_specular_intensity),
        )
        glUniform1f(program.primary_spec_sharpness, float(primary_spec_sharpness))
        glUniform1f(program.secondary_spec_sharpness, float(secondary_spec_sharpness))
        glUniform1f(
            program.primary_specular_shift_offset,
            float(primary_specular_shift_offset),
        )
        glUniform1f(
            program.secondary_specular_shift_offset,
            float(secondary_specular_shift_offset),
        )
        glUniform1f(program.hair_height_depth, float(hair_height_depth))
        glUniform1f(program.specular, float(specular))
        glUniform1f(program.primary_specular_level, float(primary_specular_level))
        glUniform1f(program.ao_exp, float(ao_exp))
        glUniform1f(program.sss_scale, float(sss_scale))
        glUniform1i(program.use_detail, int(use_detail))
        glUniform1f(program.detail_tiling, float(detail_tiling))
        glUniform1f(program.normal_blend_rate, float(normal_blend_rate))
        glUniform1f(program.roughness_blend_rate, float(roughness_blend_rate))
        glUniform1f(program.cavity_blend_rate, float(cavity_blend_rate))
        glUniform1f(program.emissive_intensity, float(emissive_intensity))
        glUniform1f(program.translucent_scale, float(translucent_scale))
        glUniform1f(program.face_uv_scale, float(face_uv_scale))
        if uvs1_vbo is None:
            glDisableVertexAttribArray(program.uv1)
            glVertexAttrib2f(program.uv1, 0.0, 0.0)
        else:
            uvs1_vbo.bind()
            glEnableVertexAttribArray(program.uv1)
            glVertexAttribPointer(program.uv1, 2, GL_FLOAT, False, 0, None)
        if tangents_vbo is None:
            glDisableVertexAttribArray(program.tangent)
            glVertexAttrib4f(program.tangent, 0.0, 0.0, 0.0, 1.0)
        else:
            tangents_vbo.bind()
            glEnableVertexAttribArray(program.tangent)
            glVertexAttribPointer(program.tangent, 4, GL_FLOAT, False, 0, None)
        return program.handle

    def unbind(self) -> None:
        if self._program is not None:
            glDisableVertexAttribArray(self._program.uv1)
            glDisableVertexAttribArray(self._program.tangent)
        for unit in range(12):
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glUseProgram(0)

    def dispose_gl(self) -> None:
        if self._program is not None:
            with suppress(Exception):
                glDeleteProgram(self._program.handle)
        self._program = None

    def _get_program(self) -> _Program:
        if self._program is not None:
            return self._program
        handle = compileProgram(
            compileShader(_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        names = (
            "u_texture", "u_textured", "u_nrro_texture", "u_has_nrro",
            "u_normal_texture", "u_has_normal", "u_rcto_texture", "u_has_rcto",
            "u_alpha_texture", "u_has_alpha",
            "u_hair_flow_texture", "u_has_hair_flow",
            "u_hair_hss_texture", "u_has_hair_hss",
            "u_detail_nrrc_texture", "u_has_detail_nrrc",
            "u_detail_mask_texture", "u_has_detail_mask",
            "u_stcm_texture", "u_has_stcm",
            "u_emissive_texture", "u_has_emissive",
            "u_second_alpha_texture", "u_has_second_alpha",
            "u_tint", "u_ambient",
            "u_diffuse", "u_exposure", "u_gamma", "u_lit",
            "u_wots_material", "u_roughness_scale", "u_occlusion_scale",
            "u_alpha_adjust", "u_alpha_threshold", "u_alpha_test",
            "u_hair_material", "u_material_family", "u_use_secondary_uv", "u_use_separate_alpha", "u_use_flow_map",
            "u_secondary_specular_intensity", "u_primary_spec_sharpness",
            "u_secondary_spec_sharpness", "u_primary_specular_shift_offset",
            "u_secondary_specular_shift_offset", "u_hair_height_depth",
            "u_specular", "u_primary_specular_level", "u_ao_exp", "u_sss_scale",
            "u_use_detail", "u_detail_tiling", "u_normal_blend_rate",
            "u_roughness_blend_rate", "u_cavity_blend_rate",
            "u_emissive_intensity", "u_translucent_scale",
            "u_face_uv_scale",
        )
        locations = tuple(glGetUniformLocation(handle, name) for name in names)
        uv1 = int(glGetAttribLocation(handle, "a_uv1"))
        tangent = int(glGetAttribLocation(handle, "a_tangent"))
        if min((*locations, uv1, tangent)) < 0:
            glDeleteProgram(handle)
            raise RuntimeError("Studio preview shader omitted a required uniform")
        self._program = _Program(handle, *locations, uv1, tangent)
        return self._program


_VERTEX_SHADER = """
#version 120

varying vec2 v_uv;
varying vec2 v_uv1;
varying vec3 v_eye_position;
varying vec3 v_eye_normal;
varying vec4 v_eye_tangent;
varying vec4 v_color;

attribute vec4 a_tangent;
attribute vec2 a_uv1;

void main() {
    vec4 eye = gl_ModelViewMatrix * gl_Vertex;
    gl_Position = gl_ProjectionMatrix * eye;
    v_uv = gl_MultiTexCoord0.xy;
    v_uv1 = a_uv1;
    v_eye_position = eye.xyz;
    v_eye_normal = normalize(gl_NormalMatrix * gl_Normal);
    vec3 eyeTangent = gl_NormalMatrix * a_tangent.xyz;
    v_eye_tangent = vec4(
        dot(eyeTangent, eyeTangent) > 0.000000000001
            ? normalize(eyeTangent)
            : vec3(0.0),
        a_tangent.w
    );
    v_color = gl_Color;
}
"""


_FRAGMENT_SHADER = WOTS_FRAGMENT_SHADER
