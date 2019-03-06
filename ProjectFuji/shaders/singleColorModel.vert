#version 330 core

layout (location = 0) in vec4 aPos;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_Model;

void main() {
	gl_Position = u_Projection * u_View * u_Model * aPos;
}

