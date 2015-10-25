#version 330

vec3 gamma(vec3 v)
{
    return pow(v, vec3(1 / 2.2));
}

out vec4 out_color;

in vec3 direction;

uniform vec3 A, B, C, D, E, F, G, H, I, Z;
uniform vec3 SunDirection;

vec3 perezExt(float cos_theta, float gamma, float cos_gamma)
{
	vec3 chi = (1 + cos_gamma * cos_gamma) / pow(1 + H * H - 2 * cos_gamma * H, vec3(1.5));

    return (1 + A * exp(B / (cos_theta + 0.01))) * (C + D * exp(E * gamma) + F * (cos_gamma * cos_gamma) + G * chi + I * sqrt(cos_theta));
}

void main()
{
	vec3 V = normalize(direction);

	float cos_theta = clamp(V.z, 0, 1);
	float cos_gamma = dot(V, SunDirection);
	float gamma_ = acos(cos_gamma);

	vec3 R = Z * perezExt(cos_theta, gamma_, cos_gamma);
    
	out_color = vec4(gamma(clamp(R, 0, 1)), 1);
}