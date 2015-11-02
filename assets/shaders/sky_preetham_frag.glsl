#version 330

vec3 gamma(vec3 v)
{
    return pow(v, vec3(1 / 2.2));
}

out vec4 out_color;

in vec3 direction;

uniform vec3 A, B, C, D, E, Z;
uniform vec3 SunDirection;

vec3 perez(float cos_theta, float gamma, float cos_gamma, vec3 A, vec3 B, vec3 C, vec3 D, vec3 E)
{
    return (1 + A * exp(B / (cos_theta + 0.01))) * (1 + C * exp(D * gamma) + E * cos_gamma * cos_gamma);
}

void main()
{
    vec3 V = normalize(direction);
    
    float cos_theta = clamp(V.y, 0, 1);
    float cos_gamma = dot(V, SunDirection);
    float gamma_ = acos(cos_gamma);
    
    vec3 R_xyY = Z * perez(cos_theta, gamma_, cos_gamma, A, B, C, D, E);
    
    vec3 R_XYZ = vec3(R_xyY.x, R_xyY.y, 1 - R_xyY.x - R_xyY.y) * R_xyY.z / R_xyY.y;
    
    float R_r = dot(vec3( 3.240479, -1.537150, -0.498535), R_XYZ);
    float R_g = dot(vec3(-0.969256,  1.875992,  0.041556), R_XYZ);
    float R_b = dot(vec3( 0.055648, -0.204043,  1.057311), R_XYZ);
    
    vec3 R = vec3(R_r, R_g, R_b);
    
    out_color = vec4(gamma(clamp(R, 0, 1)), 1);
}