#version 330 core

layout (location = 0) in vec4 a_Pos;

uniform mat4  u_View;
uniform mat4  u_Projection;

out vec4 v_LightSpacePos;

void main(void) {
	v_LightSpacePos = u_Projection * u_View * a_Pos;
	gl_Position = v_LightSpacePos;
}
