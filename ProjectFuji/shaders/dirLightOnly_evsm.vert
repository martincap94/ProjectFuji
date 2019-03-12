#version 330 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TexCoords;

out vec4 v_FragPos;
out vec3 v_Normal;
out vec4 v_LightSpacePos;
out vec2 v_TexCoords;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_LightSpaceMatrix;

void main() {
	v_Normal = a_Normal;
	v_FragPos = a_Pos;

	v_TexCoords = a_TexCoords;

	v_LightSpacePos = u_LightSpaceMatrix * a_Pos;

	gl_Position = u_Projection * u_View * a_Pos;
}

