#version 330 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec2 a_TexCoords;

out vec2 v_TexCoords;

void main(void) {
	gl_Position = a_Pos;
	v_TexCoords = a_TexCoords;
}
