#version 330

#define saturate(x) clamp(x, 0.0, 1.0)
#define PI 3.14159265359

uniform float u_cascadesNear[4];
uniform float u_cascadesFar[4];
uniform vec2 u_cascadesPlane[4];
uniform mat4 u_cascadesMatrix[4];

uniform sampler2DArray s_shadowMap;

uniform vec3 u_lightDirection;
uniform float u_expC;
uniform float u_showCascades;

in vec4 v_position;
in vec3 v_normal;
in vec3 v_vs_position;
in vec2 v_texcoord;

out vec4 f_color;

vec4 get_cascade_weights(float depth, vec4 splitNear, vec4 splitFar)
{
    return step(splitNear, vec4(depth)) * step(depth, splitFar); // near * far
}

mat4 get_cascade_viewproj(vec4 weights, mat4 viewProj[4])
{
    return viewProj[0] * weights.x + viewProj[1] * weights.y + viewProj[2] * weights.z + viewProj[3] * weights.w;
}

float get_cascade_layer(vec4 weights) 
{
    return 0.0 * weights.x + 1.0 * weights.y + 2.0 * weights.z + 3.0 * weights.w;   
}

float get_cascade_near(vec4 weights) 
{
    return u_cascadesNear[0] * weights.x + u_cascadesNear[1] * weights.y + u_cascadesNear[2] * weights.z + u_cascadesNear[3] * weights.w;
}

float get_cascade_far(vec4 weights) 
{
    return u_cascadesFar[0] * weights.x + u_cascadesFar[1] * weights.y + u_cascadesFar[2] * weights.z + u_cascadesFar[3] * weights.w;
}

vec3 get_cascade_weighted_color(vec4 weights) 
{
    return vec3(1,0,0) * weights.x + vec3(0,1,0) * weights.y + vec3(0,0,1) * weights.z + vec3(1,0,1) * weights.w;
}

// OrenNayar diffuse term
vec3 get_diffuse(vec3 diffuseColor, float roughness4, float NoV, float NoL, float VoH)
{
    float VoL = 2 * VoH - 1;
    float c1 = 1 - 0.5 * roughness4 / (roughness4 + 0.33);
    float cosri = VoL - NoV * NoL;
    float c2 = 0.45 * roughness4 / (roughness4 + 0.09) * cosri * (cosri >= 0 ? min(1, NoL / NoV) : NoL);
    return diffuseColor / PI * (NoL * c1 + c2);
}

// GGX Normal distribution
float get_normal_distrib(float roughness4, float NoH)
{
    float d = (NoH * roughness4 - NoH) * NoH + 1;
    return roughness4 / (d*d);
}

// Smith GGX geometric shadowing from "Physically-Based Shading at Disney"
float geometric_shadowing(float roughness4, float NoV, float NoL, float VoH, vec3 L, vec3 V)
{   
    float gSmithV = NoV + sqrt(NoV * (NoV - NoV * roughness4) + roughness4);
    float gSmithL = NoL + sqrt(NoL * (NoL - NoL * roughness4) + roughness4);
    return 1.0 / (gSmithV * gSmithL);
}

vec3 fresnel(vec3 specularColor, float VoH)
{
    vec3 specularColorSqrt = sqrt(clamp(vec3(0, 0, 0), vec3(0.99, 0.99, 0.99), specularColor));
    vec3 n = (1 + specularColorSqrt) / (1 - specularColorSqrt);
    vec3 g = sqrt(n * n + VoH * VoH - 1);
    return 0.5 * pow((g - VoH) / (g + VoH), vec3(2.0)) * (1 + pow(((g+VoH)*VoH - 1) / ((g-VoH)*VoH + 1), vec3(2.0)));
}

void main() 
{
    vec3 N = normalize(v_normal);
    vec3 L = normalize(-u_lightDirection);
    vec3 V = normalize(-v_vs_position);
    vec3 H = normalize(V + L);

    float NoL = saturate(dot(N, L));
    float NoV = saturate(dot(N, V));
    float VoH = saturate(dot(V, H));
    float NoH = saturate(dot(N, H));

    // Find frustum
    vec4 cascadeWeights = get_cascade_weights(
            -v_vs_position.z, 
            vec4(u_cascadesPlane[0].x, u_cascadesPlane[1].x, u_cascadesPlane[2].x, u_cascadesPlane[3].x), 
            vec4(u_cascadesPlane[0].y, u_cascadesPlane[1].y, u_cascadesPlane[2].y, u_cascadesPlane[3].y)
        );

    // Shadow coords
    mat4 viewProj = get_cascade_viewproj(cascadeWeights, u_cascadesMatrix);
    vec4 coord = viewProj * v_position;

    // Shadow term (ESM)
    float shadows = 1.0;
    if (coord.z > 0.0 && coord.x > 0.0 && coord.y > 0 && coord.x <= 1 && coord.y <= 1) 
    {
        float near = get_cascade_near(cascadeWeights);
        float far = get_cascade_far(cascadeWeights);
        float depth = coord.z - 0.0052;
        float occluderDepth = texture(s_shadowMap, vec3(coord.xy, get_cascade_layer(cascadeWeights))).r;
        float occluder = exp(u_expC * occluderDepth);
        float receiver = exp(-u_expC * depth);
        shadows = clamp(occluder * receiver, 0.0, 1.0);
    }

    vec3 baseColor = vec3(1.0);
    float metallicTerm = 0.0;
    float roughnessTerm = 0.72;
    float specularTerm = 1.0;
    vec3 diffuseColor = baseColor - baseColor * metallicTerm;
    vec3 specularColor = mix(vec3(0.08 * specularTerm), baseColor, metallicTerm);
    
    // Compute BRDF terms
    float distribution = get_normal_distrib(roughnessTerm, NoH);
    vec3 fresnel = fresnel(specularColor, VoH);
    float geom = geometric_shadowing(roughnessTerm, NoV, NoL, VoH, L, V);

    // Compute specular and diffuse and combine them
    vec3 diffuse = get_diffuse(diffuseColor, roughnessTerm, NoV, NoL, VoH);
    vec3 specular = NoL * (distribution * fresnel * geom);
    vec3 color = saturate(vec3(0.05) + shadows * 3.0 * (diffuse + specular));

    // Gamma correct
    color = pow(color, vec3(1.0f / 2.2f));

    f_color = vec4(color, 1.0);
    f_color.rgb = mix(f_color.rgb, f_color.rgb * get_cascade_weighted_color(cascadeWeights), u_showCascades);
}