#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexcoord;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewProjMatrix;

out vec2 v_texcoord;

void main() 
{
    v_texcoord = inTexcoord;
    vec3 worldPosition = (u_modelMatrix * vec4(inPosition, 1.0)).xyz;
    gl_Position = u_viewProjMatrix * vec4(worldPosition, 1.0); // clip-space
} 