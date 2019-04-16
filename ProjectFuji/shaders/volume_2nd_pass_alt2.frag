#version 330 core

in vec4 v_LightSpacePos;
in vec4 g_LightSpacePos;
in vec2 g_TexCoords;

flat in int g_ParticleTextureIdx;


out vec4 fragColor;

uniform bool u_ShowParticleTextureIdx;
uniform bool u_UseAtlasTexture;

uniform vec3 u_TintColor;

uniform sampler2D u_Texture;		// 0
uniform sampler2D u_ShadowTexture;	// 1
uniform sampler2D u_AtlasTexture;	// 2

uniform float u_Opacity;
uniform bool u_ShowHiddenParticles;

uniform float u_ScreenWidth;
uniform float u_ScreenHeight;

uniform mat4 u_ShadowMatrix;

uniform int u_Mode = 0;

// phase function info
uniform vec3 u_CameraPos;
uniform vec3 u_LightPos;
in vec3 g_WorldSpacePos;

uniform int u_PhaseFunction = 0;
uniform bool u_MultiplyPhaseByShadow = true;
uniform float u_g;	// also substitutes k for Schlick approximation (as k = -u_g)
uniform float u_g2; // g2 for Double Henyey-Greenstein
uniform float u_f;  // interpolation parameter for Double Henyey-Greenstein

#define PI 3.1415926538


float calculateRayleigh(float cosphi);
float calculateHenyeyGreenstein(float cosphi, float g);
float calculateDoubleHenyeyGreenstein(float cosphi, float g1, float g2, float f);
float calculateSchlick(float cosphi, float k);
float calculateCornetteShanks(float cosphi, float g);

void main() {

	if (u_ShowParticleTextureIdx) {
		if (g_ParticleTextureIdx == 0) {
			fragColor = vec4(1.0, 0.0, 0.0, 1.0);
		} else if (g_ParticleTextureIdx == 1) {
			fragColor = vec4(0.0, 1.0, 0.0, 1.0);
		} else if (g_ParticleTextureIdx == 2) {
			fragColor = vec4(0.0, 0.0, 1.0, 1.0);
		} else {
			fragColor = vec4(0.0, 1.0, 1.0, 1.0);
		}
		return;
	}
	if (u_UseAtlasTexture) {
		fragColor = texture(u_AtlasTexture, g_TexCoords);
	} else {
		fragColor = texture(u_Texture, g_TexCoords);
	}

	fragColor.xyz *= u_TintColor;

	

	vec3 projCoords = g_LightSpacePos.xyz / g_LightSpacePos.w;
	projCoords = projCoords * 0.5 + vec3(0.5);
	vec3 shadow = clamp(vec3(1.0) - texture(u_ShadowTexture, projCoords.xy).xyz, 0.0, 1.0);

	//fragColor = vec4(shadow, fragColor.a);

	fragColor.w *= u_Opacity;
	fragColor.xyz *= shadow * fragColor.w;


	// Phase functions
	if (u_PhaseFunction != 0) {

		vec3 sunToParticle = normalize(g_WorldSpacePos - u_LightPos);
		vec3 particleToCamera = normalize(u_CameraPos - g_WorldSpacePos);
		float cosphi = dot(sunToParticle, particleToCamera);

		float phaseFunc;

		// I prefer using if/else in shaders instead of switch since it could be potentially computationally less optimized
		if (u_PhaseFunction == 1) {

			// Rayleigh phase function
			phaseFunc = calculateRayleigh(cosphi);

		} else if (u_PhaseFunction == 2) {
			
			// Henyey-Greenstein
			phaseFunc = calculateHenyeyGreenstein(cosphi, u_g);

		} else if (u_PhaseFunction == 3) {

			// Double Henyey-Greenstein
			phaseFunc = calculateDoubleHenyeyGreenstein(cosphi, u_g, u_g2, u_f);
		
		} else if (u_PhaseFunction == 4) {

			// Schlick
			phaseFunc = calculateSchlick(cosphi, -u_g);

		} else if (u_PhaseFunction == 5) {

			// Cornette-Shanks
			phaseFunc = calculateCornetteShanks(cosphi, u_g);

		}

		
		if (u_MultiplyPhaseByShadow) {
			fragColor.xyz *= (vec3(1.0) + shadow * phaseFunc);
		} else {
			fragColor.xyz *= vec3(1.0 + phaseFunc);
		}


	}




	

}


float calculateRayleigh(float cosphi) {
	return (3.0 / (16.0 * PI) * (1.0 + cosphi * cosphi));

}


float calculateHenyeyGreenstein(float cosphi, float g) {
	float henyeyGreenstein = 1.0 / (4.0 * PI);
	henyeyGreenstein *= (1.0 - g * g);
	henyeyGreenstein /= pow((1.0 - 2.0 * g * cosphi + g * g), 3.0 / 2.0);
	return henyeyGreenstein;
}

float calculateDoubleHenyeyGreenstein(float cosphi, float g1, float g2, float f) {
	return (1.0 - f) * calculateHenyeyGreenstein(cosphi, g1) + f * calculateHenyeyGreenstein(cosphi, g2);
}


float calculateSchlick(float cosphi, float k) {
	float schlick = 1.0 / (4.0 * PI);
	schlick *= (1.0 - k) / ((1.0 + k * cosphi) * (1.0 + k * cosphi));
	return schlick;
}

float calculateCornetteShanks(float cosphi, float g) {
	float cornetteShanks = 1.0 / (4.0 * PI);
	cornetteShanks *= (3.0 / 2.0);
	cornetteShanks *= (1.0 - g * g) / (2.0 + g * g);
	cornetteShanks *= (1.0 + cosphi * cosphi);
	cornetteShanks /= pow((1.0 + g * g - 2.0 * g * cosphi), 3.0 / 2.0);
	return cornetteShanks;
}