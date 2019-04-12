#version 330 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TexCoords;
layout (location = 3) in vec3 a_Tangent;
layout (location = 4) in vec3 a_Bitangent;

out vec4 v_FragPos;
out vec3 v_Normal;
out vec4 v_LightSpacePos;
out vec4 v_PrevLightSpacePos;
out vec2 v_TexCoords;

out mat3 v_TBN;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_LightSpaceMatrix;
uniform mat4 u_PrevLightSpaceMatrix;

void main() {

	vec3 T = normalize(vec3(u_Model * vec4(a_Tangent, 0.0)));
    vec3 B = normalize(vec3(u_Model * vec4(a_Bitangent, 0.0)));
    vec3 N = normalize(vec3(u_Model * vec4(a_Normal, 0.0)));
    v_TBN = mat3(T, B, N);

	v_Normal = normalize(vec3(u_Model * vec4(a_Normal, 0.0)));
	v_FragPos = u_Model * a_Pos;

	v_TexCoords = a_TexCoords;

	v_LightSpacePos = u_LightSpaceMatrix * v_FragPos;
	v_PrevLightSpacePos = u_PrevLightSpaceMatrix * v_FragPos;

	gl_Position = u_Projection * u_View * v_FragPos;
}

