from __future__ import annotations


# Keep the ordinary and GPU-skinned preview paths on the same fragment
# contract.  The hair branch mirrors the final M_WOTS_Hair graph used by the
# Unreal reference project: HSS.G blends sharpness/specular, HSS.B shifts the
# decoded flow normal, and NRRO.B supplies the AO exponent input.
WOTS_FRAGMENT_SHADER = r"""
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
uniform sampler2D u_hair_flow_texture;
uniform bool u_has_hair_flow;
uniform sampler2D u_hair_hss_texture;
uniform bool u_has_hair_hss;
uniform sampler2D u_detail_nrrc_texture;
uniform bool u_has_detail_nrrc;
uniform sampler2D u_detail_mask_texture;
uniform bool u_has_detail_mask;
uniform sampler2D u_stcm_texture;
uniform bool u_has_stcm;
uniform sampler2D u_emissive_texture;
uniform bool u_has_emissive;
uniform sampler2D u_second_alpha_texture;
uniform bool u_has_second_alpha;
uniform vec4 u_tint;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_exposure;
uniform float u_gamma;
uniform bool u_lit;
uniform bool u_wots_material;
uniform bool u_hair_material;
uniform int u_material_family;
uniform bool u_use_secondary_uv;
uniform float u_face_uv_scale;
uniform float u_roughness_scale;
uniform float u_occlusion_scale;
uniform float u_alpha_adjust;
uniform float u_alpha_threshold;
uniform bool u_alpha_test;
uniform float u_use_separate_alpha;
uniform float u_use_flow_map;
uniform float u_secondary_specular_intensity;
uniform float u_primary_spec_sharpness;
uniform float u_secondary_spec_sharpness;
uniform float u_primary_specular_shift_offset;
uniform float u_secondary_specular_shift_offset;
uniform float u_hair_height_depth;
uniform float u_specular;
uniform float u_primary_specular_level;
uniform float u_ao_exp;
uniform float u_sss_scale;
uniform bool u_use_detail;
uniform float u_detail_tiling;
uniform float u_normal_blend_rate;
uniform float u_roughness_blend_rate;
uniform float u_cavity_blend_rate;
uniform float u_emissive_intensity;
uniform float u_translucent_scale;

varying vec2 v_uv;
varying vec2 v_uv1;
varying vec3 v_eye_position;
varying vec3 v_eye_normal;
varying vec4 v_eye_tangent;
varying vec4 v_color;

vec3 derivativeMappedNormal(vec3 tangentNormal, vec3 normal, vec2 uv) {
    vec3 positionX = dFdx(v_eye_position);
    vec3 positionY = dFdy(v_eye_position);
    vec2 uvX = dFdx(uv);
    vec2 uvY = dFdy(uv);
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

vec3 mappedNormal(vec3 tangentNormal, vec2 uv) {
    vec3 normal = normalize(v_eye_normal);
    vec3 tangent = v_eye_tangent.xyz;
    if (dot(tangent, tangent) < 1e-12) {
        return derivativeMappedNormal(tangentNormal, normal, uv);
    }
    tangent = normalize(tangent - normal * dot(normal, tangent));
    vec3 bitangent = normalize(cross(normal, tangent)) * v_eye_tangent.w;
    return normalize(
        tangent * tangentNormal.x
        + bitangent * tangentNormal.y
        + normal * tangentNormal.z
    );
}

vec3 decodeAG(vec4 packedNormal) {
    vec2 xy = packedNormal.ag * 2.0 - 1.0;
    return normalize(vec3(xy, sqrt(max(1.0 - dot(xy, xy), 0.0))));
}

vec3 blendAngleCorrectedNormals(vec3 baseNormal, vec3 detailNormal) {
    vec3 t = baseNormal + vec3(0.0, 0.0, 1.0);
    vec3 u = detailNormal * vec3(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) - u * t.z);
}

void main() {
    vec2 surfaceUv = u_use_secondary_uv ? v_uv1 : v_uv;
    // Face ALBD/NRM resources contain four complete expression tiles.  UV0
    // addresses one face and Dummy_UVScale maps it into the neutral tile.
    // RCTO and overlay textures are ordinary full-size maps and therefore
    // continue to use surfaceUv below.
    vec2 primaryUv = u_material_family == 2
        ? v_uv * u_face_uv_scale
        : surfaceUv;
    vec4 texel = u_textured ? texture2D(u_texture, primaryUv) : vec4(1.0);
    vec4 surface = texel * v_color * u_tint;
    vec3 normal = normalize(v_eye_normal);
    float roughness = 1.0;
    float occlusion = 1.0;
    float specularStrength = 1.0;
    float subsurfaceAmount = 0.0;
    vec4 nrro = vec4(1.0);
    vec4 hss = vec4(0.0);
    vec3 tangentNormal = vec3(0.0, 0.0, 1.0);
    vec3 subsurfaceColor = vec3(0.0);
    vec3 emissiveColor = vec3(0.0);

    if (u_wots_material) {
        // RE character shaders use vertex alpha as a mask, not vertex RGB as
        // an albedo multiplier. This matches the M_WOTS material graphs.
        surface = texel * u_tint;
    }

    if (u_wots_material && u_has_nrro) {
        nrro = texture2D(u_nrro_texture, surfaceUv);
        tangentNormal = decodeAG(nrro);
        roughness = nrro.r;
        occlusion = clamp(
            mix(1.0, nrro.b, max(u_occlusion_scale, 0.0)),
            0.0,
            1.0
        );
    } else if (u_wots_material && u_has_normal) {
        tangentNormal = normalize(
            texture2D(u_normal_texture, primaryUv).rgb * 2.0 - 1.0
        );
    }
    if (u_wots_material && u_has_rcto) {
        vec4 rcto = texture2D(u_rcto_texture, surfaceUv);
        roughness = rcto.r;
        occlusion = clamp(
            mix(1.0, rcto.a, max(u_occlusion_scale, 0.0)),
            0.0,
            1.0
        );
    }

    if (u_wots_material && u_use_detail && u_has_detail_nrrc) {
        vec4 detail = texture2D(
            u_detail_nrrc_texture,
            v_uv * max(u_detail_tiling, 0.0001)
        );
        float detailMask = u_has_detail_mask
            ? texture2D(u_detail_mask_texture, surfaceUv).r
            : 1.0;
        float normalStrength = clamp(
            u_normal_blend_rate * detailMask, 0.0, 1.0
        );
        vec3 detailNormal = normalize(mix(
            vec3(0.0, 0.0, 1.0),
            decodeAG(detail),
            normalStrength
        ));
        tangentNormal = blendAngleCorrectedNormals(tangentNormal, detailNormal);
        roughness = mix(
            roughness,
            detail.r,
            clamp(u_roughness_blend_rate * detailMask, 0.0, 1.0)
        );
        occlusion = mix(
            occlusion,
            occlusion * detail.b,
            clamp(u_cavity_blend_rate * detailMask, 0.0, 1.0)
        );
    }
    if (u_wots_material) {
        normal = mappedNormal(tangentNormal, surfaceUv);
        roughness = clamp(
            roughness * max(u_roughness_scale, 0.0), 0.04, 1.0
        );
        if (u_has_stcm) {
            vec4 stcm = texture2D(u_stcm_texture, surfaceUv);
            subsurfaceAmount = max(stcm.r * u_sss_scale, 0.0);
            subsurfaceColor = surface.rgb * subsurfaceAmount;
            if (u_material_family == 3) {
                occlusion *= clamp(stcm.r, 0.0, 1.0);
            }
        }
        if (u_has_emissive) {
            float emissiveScale = max(u_emissive_intensity, 0.0);
            if (u_material_family == 8 && emissiveScale == 0.0) {
                emissiveScale = 0.35;
            }
            emissiveColor = texture2D(u_emissive_texture, surfaceUv).rgb
                * emissiveScale;
        }
        if (u_material_family == 9) {
            surface.rgb = vec3(0.0);
        }
    }

    if (u_hair_material) {
        // The final UE graph does not multiply hair BaseColor by vertex color.
        surface = texel * u_tint;
        hss = u_has_hair_hss
            ? texture2D(u_hair_hss_texture, surfaceUv)
            : vec4(0.0);
        float secondaryBlend = clamp(
            hss.g * max(u_secondary_specular_intensity, 0.0),
            0.0,
            1.0
        );
        float sharpness = mix(
            max(u_primary_spec_sharpness, 0.0),
            max(u_secondary_spec_sharpness, 0.0),
            secondaryBlend
        );
        roughness = clamp(
            sqrt(2.0 / max(sharpness + 2.0, 0.0001))
                * max(u_roughness_scale, 0.0),
            0.04,
            1.0
        );
        specularStrength = max(u_specular, 0.0) * mix(
            max(u_primary_specular_level + 0.4, 0.0),
            max(u_secondary_specular_intensity, 0.0),
            clamp(hss.g, 0.0, 1.0)
        );
        if (u_has_nrro) {
            occlusion = pow(
                max(clamp(nrro.b, 0.0, 1.0), 0.0001),
                max(u_ao_exp, 0.0)
            );
            surface.rgb *= occlusion;
        }
        if (u_has_hair_flow) {
            vec3 flowNormal = texture2D(u_hair_flow_texture, surfaceUv).rgb * 2.0
                - vec3(254.0 / 255.0);
            float specularShift = mix(
                u_primary_specular_shift_offset,
                u_secondary_specular_shift_offset,
                secondaryBlend
            );
            float heightShift = (hss.b - 0.5) * u_hair_height_depth;
            flowNormal.x += specularShift + heightShift;
            flowNormal = normalize(mix(
                vec3(0.0, 1.0, 0.0),
                flowNormal,
                clamp(u_use_flow_map, 0.0, 1.0)
            ));
            normal = mappedNormal(flowNormal, surfaceUv);
        }
        subsurfaceAmount = max(hss.r * u_sss_scale, subsurfaceAmount);
        subsurfaceColor = surface.rgb * subsurfaceAmount;
    }

    float alpha = surface.a;
    if (u_wots_material) {
        bool familyUsesAlpha = u_material_family == 5
            || u_material_family == 6
            || u_material_family == 8
            || u_material_family == 9
            || u_material_family == 10;
        float alphaMask = (
            u_has_alpha && (familyUsesAlpha || u_use_separate_alpha >= 0.5)
        ) ? texture2D(u_alpha_texture, surfaceUv).r : texel.a;
        if (u_has_second_alpha && (u_material_family == 8 || u_material_family == 9)) {
            alphaMask *= texture2D(u_second_alpha_texture, surfaceUv).r;
        }
        bool vertexAlphaMask = u_material_family == 5
            || u_material_family == 6
            || u_material_family == 11;
        alpha = (vertexAlphaMask ? v_color.a : 1.0) * u_tint.a * pow(
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
        float strength = mix(0.42, 0.04, roughness) * specularStrength;
        float specular = pow(max(dot(normal, halfDirection), 0.0), exponent);
        litSurface += vec3(specular * strength * u_diffuse * occlusion);
        litSurface += subsurfaceColor * (0.12 + 0.18 * rim);
        litSurface += emissiveColor;
        if (u_material_family == 10) {
            litSurface = mix(
                litSurface,
                surface.rgb + vec3(specular * strength),
                clamp(u_translucent_scale, 0.0, 1.0)
            );
        }
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


WOTS_MATERIAL_UNIFORM_NAMES = (
    "u_tint", "u_ambient", "u_diffuse", "u_exposure", "u_gamma", "u_lit",
    "u_texture", "u_textured", "u_nrro_texture", "u_has_nrro",
    "u_normal_texture", "u_has_normal", "u_rcto_texture", "u_has_rcto",
    "u_alpha_texture", "u_has_alpha", "u_hair_flow_texture",
    "u_has_hair_flow", "u_hair_hss_texture", "u_has_hair_hss",
    "u_detail_nrrc_texture", "u_has_detail_nrrc",
    "u_detail_mask_texture", "u_has_detail_mask",
    "u_stcm_texture", "u_has_stcm", "u_emissive_texture",
    "u_has_emissive", "u_second_alpha_texture", "u_has_second_alpha",
    "u_wots_material", "u_hair_material", "u_material_family",
    "u_use_secondary_uv", "u_roughness_scale",
    "u_occlusion_scale", "u_alpha_adjust", "u_alpha_threshold",
    "u_alpha_test", "u_use_separate_alpha", "u_use_flow_map",
    "u_secondary_specular_intensity", "u_primary_spec_sharpness",
    "u_secondary_spec_sharpness", "u_primary_specular_shift_offset",
    "u_secondary_specular_shift_offset", "u_hair_height_depth", "u_specular",
    "u_primary_specular_level", "u_ao_exp", "u_sss_scale",
    "u_use_detail", "u_detail_tiling", "u_normal_blend_rate",
    "u_roughness_blend_rate", "u_cavity_blend_rate",
    "u_emissive_intensity", "u_translucent_scale",
    "u_face_uv_scale",
)
