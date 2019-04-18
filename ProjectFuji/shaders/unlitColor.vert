#version 400 core

layout (location = 0) in vec3 a_Pos;
layout (location = 1) in vec3 a_Color;

uniform mat4 u_View;
uniform mat4 u_Projection;

out vec3 v_Color;

void main() {
	v_Color = a_Color;
	gl_Position = u_Projection * u_View * vec4(a_Pos, 1.0);
}

