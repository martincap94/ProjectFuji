#version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 u_View;
uniform mat4 u_Projection;

void main() {
	gl_Position = u_Projection * u_View * vec4(aPos, 1.0);
}

