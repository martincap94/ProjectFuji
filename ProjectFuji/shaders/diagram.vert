#version 330 core


layout (location = 0) in vec4 a_Vertex;
layout (location = 1) in vec2 a_TexCoords;

out vec2 v_TexCoords;


void main(void) {
	gl_Position = vec4(a_Vertex.x / 2.0, a_Vertex.y / 2.0, 0.0, 1.0);
	v_TexCoords = a_TexCoords;
}
