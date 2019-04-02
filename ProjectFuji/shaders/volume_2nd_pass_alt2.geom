#version 400 core

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

in int v_ParticleTextureIdx[];
flat out int g_ParticleTextureIdx;

uniform mat4  u_View;
uniform mat4  u_Projection;

uniform mat4 u_LightSpaceMatrix;

uniform bool u_UseAtlasTexture;

uniform vec3 u_CameraPos;
uniform float u_WorldPointSize;

uniform int u_Mode;

const vec3 worldup = vec3(0.0, 1.0, 0.0);

out vec2 g_TexCoords;
out vec3 g_LightSpacePos;

float goldNoise(in vec2 coordinate, in float seed);

void main() {
	g_ParticleTextureIdx = v_ParticleTextureIdx[0];

	vec3 pos = gl_in[0].gl_Position.xyz;
	mat4 VP = u_Projection * u_View;

	float tmpscale = u_WorldPointSize / 10.0;

	vec3 right;
	vec3 up;
	if (u_Mode == 0) {
		vec3 toCamera = normalize(u_CameraPos - pos);
		right = normalize(cross(toCamera, worldup)) * tmpscale;
		up = normalize(cross(right, toCamera)) * tmpscale;
	} else {
		right = normalize(vec3(goldNoise(pos.xy, pos.x), goldNoise(pos.xy, pos.y), goldNoise(pos.xy, pos.z))) * tmpscale;
		up = normalize(cross(right, worldup)) * tmpscale;
	}


	/*
		testing - texture array or just multiple textures would be more effective than changing
		texture coordinates in geometry shader!
	*/
	
	if (u_UseAtlasTexture) {
		// maybe beware half pixel offsets: https://gamedev.stackexchange.com/questions/46963/how-to-avoid-texture-bleeding-in-a-texture-atlas
		// testing for now, total correctness not very important

		vec2 texOffset = vec2(0.0, 0.0);
		
		// 0 - no offset
		// 1 - offset right
		// 2 - offset down
		// 3 - offset right and down
		if (g_ParticleTextureIdx == 1) {
			texOffset.s += 0.5;
		} else if (g_ParticleTextureIdx == 2) {
			texOffset.t += 0.5;
		} else if (g_ParticleTextureIdx == 3) {
			texOffset += vec2(0.5, 0.5);
		}
		vec4 tmppos = vec4(pos - right - up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = texOffset;
		EmitVertex();

		tmppos = vec4(pos - right + up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = vec2(0.0, 0.5) + texOffset;
		EmitVertex();

		tmppos = vec4(pos + right - up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = vec2(0.5, 0.0) + texOffset;
		EmitVertex();

		tmppos = vec4(pos + right + up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = vec2(0.5, 0.5) + texOffset;
		EmitVertex();
		
	} else {

		vec4 tmppos = vec4(pos - right - up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = vec2(0.0, 0.0);
		EmitVertex();

		tmppos = vec4(pos - right + up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = vec2(0.0, 1.0);
		EmitVertex();

		tmppos = vec4(pos + right - up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = vec2(1.0, 0.0);
		EmitVertex();

		tmppos = vec4(pos + right + up, 1.0);
		gl_Position = VP * tmppos;
		g_LightSpacePos = vec3(u_LightSpaceMatrix * tmppos);
		g_TexCoords = vec2(1.0, 1.0);
		EmitVertex();
	}

	EndPrimitive();

}

// Taken from: https://www.shadertoy.com/view/ltB3zD
// or can be found: https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// Gold Noise ©2015 dcerisano@standard3d.com
// - based on the Golden Ratio, PI and the Square Root of Two
// - superior distribution
// - fastest static noise generator function
// - works with all chipsets (including low precision)

float PHI = 1.61803398874989484820459 * 00000.1; // Golden Ratio   
float PI  = 3.14159265358979323846264 * 00000.1; // PI
float SQ2 = 1.41421356237309504880169 * 10000.0; // Square Root of Two

float goldNoise(in vec2 coordinate, in float seed) {
    return fract(tan(distance(coordinate*(seed+PHI), vec2(PHI, PI)))*SQ2);
}



