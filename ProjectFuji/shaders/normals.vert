#version 400 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TexCoords;
layout (location = 3) in vec3 a_Tangent;
layout (location = 4) in vec3 a_Bitangent;

out vec2 v_TexCoords;
out vec3 v_FragPos;
out mat3 v_TBN;
out vec4 v_LightSpacePos;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_LightSpaceMatrix;

void main() {
	vec3 T = normalize(vec3(u_Model * vec4(a_Tangent, 0.0)));
    vec3 B = normalize(vec3(u_Model * vec4(a_Bitangent, 0.0)));
    vec3 N = normalize(vec3(u_Model * vec4(a_Normal, 0.0)));
    v_TBN = mat3(T, B, N);
	
    v_TexCoords = a_TexCoords;
    v_FragPos = vec3(u_Model * a_Pos);
	v_LightSpacePos = u_LightSpaceMatrix * u_Model * a_Pos;
	
	gl_Position = u_Projection * u_View * u_Model * a_Pos;

}