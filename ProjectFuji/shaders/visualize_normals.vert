#version 400 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TexCoords;
layout (location = 3) in vec3 a_Tangent;
layout (location = 4) in vec3 a_Bitangent;


out NormalData {
	vec3 normal;
} normalData;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;


void main() {
	
	mat3 normalMatrix = mat3(transpose(inverse(u_View * u_Model)));
	normalData.normal = normalize(vec3(u_Projection * vec4(normalMatrix * a_Normal, 0.0))); 
	gl_Position = u_Projection * u_View * u_Model * a_Pos;

}