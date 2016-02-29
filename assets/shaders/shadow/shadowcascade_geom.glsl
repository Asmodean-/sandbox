#version 330
#extension GL_ARB_gpu_shader5: require

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 u_cascadeViewMatrixArray[4];
uniform mat4 u_cascadeProjMatrixArray[4];

out float g_layer;
out vec3 vs_position;

void main() 
{
    for (int i = 0; i < gl_in.length(); ++i) 
    {
        vec4 pos = (u_cascadeViewMatrixArray[gl_InvocationID] * gl_in[i].gl_Position );
        gl_Position = u_cascadeProjMatrixArray[gl_InvocationID] * pos;
        vs_position = pos.xyz;
        gl_Layer = gl_InvocationID;
        g_layer = float(gl_InvocationID);
        EmitVertex();
    }
    EndPrimitive();
} 