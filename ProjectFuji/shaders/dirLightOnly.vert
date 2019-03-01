#version 330 core

layout (location = 0) in vec3 a_Pos;
layout (location = 1) in vec3 a_Normal;

out vec3 v_FragPos;
out vec3 v_Normal;


uniform mat4 u_View;
uniform mat4 u_Projection;

void main() {
	v_Normal = a_Normal;
	v_FragPos = a_Pos;
	gl_Position = u_Projection * u_View * vec4(a_Pos, 1.0);
}

