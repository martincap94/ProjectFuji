#version 330 core

uniform mat4  u_ModelViewMatrix;
uniform mat4  u_ProjectionMatrix;

layout (location = 0) in vec4 a_Vertex;

out vec4 v_LightSpacePos;

void main(void) {
    v_LightSpacePos = u_ModelViewMatrix * a_Vertex; // v_LightSpacePos is not projected in this case!
    gl_Position = u_ProjectionMatrix * v_LightSpacePos;
}
