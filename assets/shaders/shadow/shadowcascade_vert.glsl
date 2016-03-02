#version 330
#extension GL_ARB_gpu_shader5: require

layout(location = 0) in vec4 inPosition;

void main()
{
    gl_Position = inPosition;
}