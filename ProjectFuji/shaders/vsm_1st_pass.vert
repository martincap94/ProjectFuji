#version 400 core

uniform mat4  u_ModelViewMatrix;
uniform mat4  u_ProjectionMatrix;

layout (location = 0) in vec3 a_Vertex;
layout (location = 1) in vec3 a_Normal;

out vec4 v_LightSpacePos;

void main(void) {
	v_LightSpacePos = u_ProjectionMatrix * u_ModelViewMatrix * vec4(a_Vertex, 1.0);
	gl_Position = v_LightSpacePos;
}
