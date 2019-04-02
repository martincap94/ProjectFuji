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

	

}
