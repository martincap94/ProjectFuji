#version 330 core

layout (location = 0) in vec4 a_Pos;
layout (location = 5) in mat4 a_InstanceModelMatrix;

uniform mat4  u_Model;
uniform mat4  u_View;
uniform mat4  u_Projection;

uniform bool  u_IsInstanced = false;

out vec4 v_LightSpacePos;

void main(void) {
	if (u_IsInstanced) {
		v_LightSpacePos = u_Projection * u_View * u_Model * a_InstanceModelMatrix * a_Pos;
	} else {
		v_LightSpacePos = u_Projection * u_View * u_Model * a_Pos;
	}
	gl_Position = v_LightSpacePos;
}
