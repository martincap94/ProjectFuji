#version 450 core

layout (location = 0) in vec4 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TexCoords;
layout (location = 3) in vec3 a_Tangent;
layout (location = 4) in vec3 a_Bitangent;
layout (location = 5) in mat4 a_InstanceModelMatrix;

out vec3 v_FragPos;
out vec2 v_TexCoords;
out vec3 v_Normal;
out vec4 v_LightSpacePos;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_LightSpaceMatrix;

uniform vec3 u_CameraPos;
uniform float u_CullDistance = 500.0;

void main() {

	mat4 finalModelMatrix = u_Model * a_InstanceModelMatrix;
	v_FragPos = vec3(finalModelMatrix * a_Pos);

	float camDist = distance(u_CameraPos, v_FragPos);
	if (camDist > u_CullDistance) {
		gl_CullDistance[0] = -1;
	}

	
    v_TexCoords = a_TexCoords;
	v_LightSpacePos = u_LightSpaceMatrix * vec4(v_FragPos, 1.0);
	
	gl_Position = u_Projection * u_View * vec4(v_FragPos, 1.0);

}