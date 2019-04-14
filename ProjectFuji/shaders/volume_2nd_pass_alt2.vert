#version 430 core

layout (location = 0) in vec4 a_Pos;
layout (location = 5) in int a_ProfileIndex;

out int v_ParticleTextureIdx;


#define MAX_PROFILES 1000
uniform float u_ProfileCCLs[MAX_PROFILES];
uniform int u_NumProfiles;

flat out int g_DiscardParticle;

const float discardThreshold = 10.0; // 10 meters

void main() {
	g_DiscardParticle = 0;
	if (a_ProfileIndex < u_NumProfiles) {
		float CCL = u_ProfileCCLs[a_ProfileIndex];
		float diff = a_Pos.y - CCL;
		if (a_Pos.y - CCL < discardThreshold) {
			g_DiscardParticle = 1;
		}
	}

	gl_Position = a_Pos;
	v_ParticleTextureIdx = int(mod(abs(a_Pos.y), 4));
}


