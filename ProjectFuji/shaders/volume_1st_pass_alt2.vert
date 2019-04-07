#version 330 core

layout (location = 0) in vec4 a_Pos;

layout (location = 5) in int a_ProfileIndex;


uniform mat4  u_View;
uniform mat4  u_Projection;

uniform vec3 u_LightPos;
uniform vec3 u_CameraPos;
uniform float u_WorldPointSize;

uniform int u_Mode;

#define MAX_PROFILES 1000
uniform float u_ProfileCCLs[MAX_PROFILES];
uniform int u_NumProfiles;

flat out int g_DiscardParticle;


void main(void) {
	g_DiscardParticle = 0;
	if (a_ProfileIndex < u_NumProfiles) {
		float CCL = u_ProfileCCLs[a_ProfileIndex];
		float diff = a_Pos.y - CCL;
		if (a_Pos.y - CCL < 0.0) {
			g_DiscardParticle = 1;
		}
	}
	gl_Position = a_Pos;
}
