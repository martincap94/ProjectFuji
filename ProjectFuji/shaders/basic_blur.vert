#version 330 core

//uniform mat4  u_ModelViewMatrix;
//uniform mat4  u_ProjectionMatrix;
uniform int   u_UserVariableInt;
uniform float u_UserVariableFloat;

layout (location = 0) in vec4 a_Vertex;
layout (location = 1) in vec2 a_TexCoords;

out vec2 v_TexCoords;


void main(void) {
	gl_Position = vec4(a_Vertex.x, a_Vertex.y, 0.0, 1.0);
	v_TexCoords = a_TexCoords;
}
