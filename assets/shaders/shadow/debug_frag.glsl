#version 330 core

uniform int u_section;
uniform sampler2DArray s_shadowSampler;

in vec2 v_texcoord;
out vec4 f_color;

void main() 
{
    f_color = texture(s_shadowSampler, vec3(v_texcoord, float(u_section)));
}