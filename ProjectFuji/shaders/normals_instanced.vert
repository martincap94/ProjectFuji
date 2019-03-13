#version 400 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TexCoords;
layout (location = 3) in vec3 a_Tangent;
layout (location = 4) in vec3 a_Bitangent;
layout (location = 5) in mat4 a_InstanceModelMatrix;

out vec2 v_TexCoords;
//out vec3 v_Normal;
out vec3 v_FragPos;
out mat3 v_TBN;
out vec4 v_FragPosLightSpace;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_LightSpaceMatrix;

void main() {
	mat4 finalModelMatrix = a_InstanceModelMatrix * u_Model;
	vec3 T = normalize(vec3(finalModelMatrix * vec4(a_Tangent, 0.0)));
    vec3 B = normalize(vec3(finalModelMatrix * vec4(a_Bitangent, 0.0)));
    vec3 N = normalize(vec3(finalModelMatrix * vec4(a_Normal, 0.0)));
    v_TBN = mat3(T, B, N);
	
    v_TexCoords = a_TexCoords;
    //v_Normal = mat3(transpose(inverse(u_Model))) * a_Normal;
    v_FragPos = vec3(u_Model * a_Pos);
	v_FragPosLightSpace = u_LightSpaceMatrix * vec4(v_FragPos, 1.0);
	
	gl_Position = u_Projection * u_View * finalModelMatrix * a_Pos;

}