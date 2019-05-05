#version 330 core

layout (location = 0) in vec4 a_Pos;

uniform mat4 u_View;
uniform mat4 u_Projection;

void main() {
	gl_Position = u_Projection * u_View  * a_Pos;
}

