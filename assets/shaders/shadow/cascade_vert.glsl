#version 330

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 3) in vec2 inTexcoord;

uniform mat4 u_modelMatrix;
uniform mat4 u_modelMatrixIT;
uniform mat4 u_viewProjMatrix;
uniform mat4 u_modelViewMatrix;
uniform mat4 u_projMatrix;
uniform mat3 u_normalMatrix;

out vec4 v_position;
out vec3 v_normal;
out vec3 v_vs_position;
out vec2 v_texcoord;

void main() 
{
    v_texcoord = inTexcoord;
    vec4 viewSpacePosition = (u_modelViewMatrix * vec4(inPosition, 1.0));
    v_position = vec4(inPosition, 1.0);
    v_normal = normalize(u_normalMatrix * inNormal);
    v_vs_position = viewSpacePosition.xyz;
    gl_Position = u_projMatrix * viewSpacePosition; 
} 