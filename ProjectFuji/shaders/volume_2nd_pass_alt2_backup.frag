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
uniform float u_SymmetryParameter;

#define PI 3.1415926538

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


	// Rayleigh scattering
	if (u_PhaseFunction != 0) {

		vec3 sunToParticle = normalize(g_WorldSpacePos - u_LightPos);
		vec3 particleToCamera = normalize(u_CameraPos - g_WorldSpacePos);
		float cosphi = dot(sunToParticle, particleToCamera);

		
		if (u_PhaseFunction == 1) {

			float rayleighPhase = 3.0 / (16.0 * PI) * (1.0 + cosphi * cosphi);

			// playing around with ideas
			/*
			if (length(shadow) > 1.0 && cosphi > 0.0) {
				fragColor.xyz *= vec3(rayleighPhase * 2.0 + 1.0);
				//fragColor = vec4(1.0, 0.0, 0.0, 1.0);
				return;
			}
			*/


			fragColor.xyz *= vec3(rayleighPhase * 2.0 + 1.0);
			return;


			/*
			if (u_PhaseFunction == 2) {
				fragColor.xyz *= rayleighPhase;
			} else if (u_PhaseFunction == 3) {
				fragColor = vec4(vec3(rayleighPhase), 1.0);
			} else if (u_PhaseFunction == 4) {
				fragColor.xyz += vec3(rayleighPhase / 100.0);
			}
			*/
		} else if (u_PhaseFunction <= 5) {
			
			float henyeyGreenstein = 1.0 / (4.0 * PI);

			float rightSide = (1.0 - u_SymmetryParameter * u_SymmetryParameter) / pow((1.0 - 2.0 * u_SymmetryParameter * cosphi + u_SymmetryParameter * u_SymmetryParameter), 3.0 / 2.0);
			//rightSide = pow(rightSide, 3.0 / 2.0); // incorrect pow!
			henyeyGreenstein *= rightSide;

			if (u_PhaseFunction == 2) {
				fragColor = vec4(vec3(henyeyGreenstein), 1.0);
			} else if (u_PhaseFunction == 3) {
				fragColor.xyz *= (1.0 + henyeyGreenstein);
			} else if (u_PhaseFunction == 4) {
				
				fragColor.xyz *= (1.0 + henyeyGreenstein * length(shadow));

			} else {
				fragColor.xyz *= (vec3(1.0) + shadow * henyeyGreenstein);

			}

		} else if (u_PhaseFunction <= 6) {
			// Schlick
			float schlick = 1.0 / (4.0 * PI);
			float k = -u_SymmetryParameter;
			schlick *= (1.0 - k) / ((1.0 + k * cosphi) * (1.0 + k * cosphi));

			fragColor.xyz *= (vec3(1.0) + shadow * schlick);

		} else if (u_PhaseFunction <= 7) {
			// Cornette-Shanks

			float g = u_SymmetryParameter;
			float cornetteShanks = 1.0 / (4.0 * PI);
			cornetteShanks *= (3.0 / 2.0);
			cornetteShanks *= (1.0 - g * g) / (2.0 + g * g);
			cornetteShanks *= (1.0 + cosphi * cosphi);
			cornetteShanks /= pow((1.0 + g * g - 2.0 * g * cosphi), 3.0 / 2.0);

			fragColor.xyz *= (vec3(1.0) + shadow * cornetteShanks);

		}



	}




	

}
