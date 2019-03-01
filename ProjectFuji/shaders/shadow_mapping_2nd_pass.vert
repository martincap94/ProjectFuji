#version 330 core

in vec4 a_Vertex;
in vec3 a_Normal;
in vec2 a_TexCoord;

out vec3 v_Normal;
out vec2 v_TexCoord;
out vec4 v_Vertex;
out vec4 v_LightSpacePos;

uniform mat4  u_View;
uniform mat4  u_Projection;
uniform mat4  u_LightViewMatrix;		// Use these two matrixes to calculate vertex position in ...
uniform mat4  u_LightProjectionMatrix;  // ...light view space, or
uniform mat4  u_LightSpaceMatrix;
uniform int   u_UserVariableInt;
uniform float u_UserVariableFloat;

uniform int   u_PCFMode;



void main() {
    v_Vertex   = u_View * a_Vertex;
    v_Normal   = mat3(u_View) * a_Normal;
    v_TexCoord = a_TexCoord;

    mat4 shadowTransform;

	switch(u_PCFMode) {
		case 0:
			shadowTransform = u_LightViewMatrix;
			break;
		//case 1:
		//case 2:
		//case 3:
		//case 4:
		default:
			shadowTransform = u_LightSpaceMatrix;
			break;
	}
	v_LightSpacePos = shadowTransform * a_Vertex;

    gl_Position = u_Projection * v_Vertex;
}
