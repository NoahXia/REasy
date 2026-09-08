from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from OpenGL.GL import (
    GL_FRAGMENT_SHADER,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindTexture,
    glDeleteProgram,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform4f,
    glUseProgram,
)
from OpenGL.GL.shaders import compileProgram, compileShader


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
        lit: bool = True,
        wots_material: bool = False,
        roughness_scale: float = 1.0,
        occlusion_scale: float = 1.0,
        alpha_adjust: float = 1.0,
        alpha_threshold: float = 0.5,
        alpha_test: bool = False,
    ) -> int:
        program = self._get_program()
        glUseProgram(program.handle)
        for unit, (location, texture) in enumerate((
            (program.texture, texture_id if textured else 0),
            (program.nrro_texture, nrro_texture_id),
            (program.normal_texture, normal_texture_id),
            (program.rcto_texture, rcto_texture_id),
            (program.alpha_texture, alpha_texture_id),
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
        return program.handle

    @staticmethod
    def unbind() -> None:
        for unit in range(5):
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
            "u_alpha_texture", "u_has_alpha", "u_tint", "u_ambient",
            "u_diffuse", "u_exposure", "u_gamma", "u_lit",
            "u_wots_material", "u_roughness_scale", "u_occlusion_scale",
            "u_alpha_adjust", "u_alpha_threshold", "u_alpha_test",
        )
        locations = tuple(glGetUniformLocation(handle, name) for name in names)
        if min(locations) < 0:
            glDeleteProgram(handle)
            raise RuntimeError("Studio preview shader omitted a required uniform")
        self._program = _Program(handle, *locations)
        return self._program


_VERTEX_SHADER = """
#version 120

varying vec2 v_uv;
varying vec3 v_eye_position;
varying vec3 v_eye_normal;
varying vec4 v_color;

void main() {
    vec4 eye = gl_ModelViewMatrix * gl_Vertex;
    gl_Position = gl_ProjectionMatrix * eye;
    v_uv = gl_MultiTexCoord0.xy;
    v_eye_position = eye.xyz;
    v_eye_normal = normalize(gl_NormalMatrix * gl_Normal);
    v_color = gl_Color;
}
"""


_FRAGMENT_SHADER = """
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
        vec2 normalXY = nrro.rg * 2.0 - 1.0;
        vec3 tangentNormal = vec3(
            normalXY,
            sqrt(max(1.0 - dot(normalXY, normalXY), 0.0))
        );
        normal = mappedNormal(normalize(tangentNormal));
        roughness = clamp(nrro.b * max(u_roughness_scale, 0.0), 0.04, 1.0);
        occlusion = clamp(
            mix(1.0, nrro.a, max(u_occlusion_scale, 0.0)),
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
    vec3 fillDirection = normalize(vec3(-0.70, 0.30, 0.45));
    vec3 rimDirection = normalize(vec3(0.10, 0.35, -0.95));
    float key = max(dot(normal, keyDirection), 0.0);
    float fill = max(dot(normal, fillDirection), 0.0);
    float rim = pow(max(dot(normal, rimDirection), 0.0), 2.0);
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
