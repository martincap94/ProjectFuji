#version 330 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TexCoords;
layout (location = 3) in vec3 a_Tangent;
layout (location = 4) in vec3 a_Bitangent;

out vec4 v_FragPos;
out vec3 v_Normal;
out vec4 v_LightSpacePos;
out vec2 v_TexCoords;

out mat3 v_TBN;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_LightSpaceMatrix;

void main() {

	vec3 T = normalize(a_Tangent);
    vec3 B = normalize(a_Bitangent);
    vec3 N = normalize(a_Normal);
    v_TBN = mat3(T, B, N);

	v_Normal = a_Normal;
	v_FragPos = a_Pos;

	v_TexCoords = a_TexCoords;

	v_LightSpacePos = u_LightSpaceMatrix * u_Model * a_Pos;

	gl_Position = u_Projection * u_View * u_Model * a_Pos;
}

