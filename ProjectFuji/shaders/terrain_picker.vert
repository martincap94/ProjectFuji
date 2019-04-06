#version 330 core

layout (location = 0) in vec4 a_Pos;

out vec4 v_FragPos;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

void main() {
	v_FragPos = u_Model * a_Pos;
	gl_Position = u_Projection * u_View * v_FragPos;
}

