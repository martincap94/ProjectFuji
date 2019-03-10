#version 430 core

layout (location = 0) in vec3 a_Pos;


layout (location = 5) in int a_ProfileIndex;

//struct ProfileData {
//	float CCL;
//};

//layout (binding = 0, std430) buffer STLPStorage {
//	//int numProfiles;
//	ProfileData profileData[];
//};

#define MAX_PROFILES 1000
uniform float u_ProfileCCLs[MAX_PROFILES];
uniform int u_NumProfiles;

uniform mat4 u_View;
uniform mat4 u_Projection;

uniform vec3 u_CameraPos;
uniform float u_PointSizeModifier;

uniform int u_OpacityBlendMode = 0;
uniform float u_OpacityBlendRange = 10.0;


out float v_ParticleOpacityMultiplier;
out float v_ComputedOpacity;

float goldNoise(in vec2 coordinate, in float seed);

void main() {
	float cameraDist = distance(a_Pos, u_CameraPos);
	float pointScale = u_PointSizeModifier * 100.0 / cameraDist;
	gl_Position = u_Projection * u_View * vec4(a_Pos, 1.0);
	gl_PointSize = pointScale;
	//v_ParticleOpacityMultiplier = goldNoise(gl_Position.xy, gl_Position.z);
	//v_DiscardFragments = (a_ProfileIndex > 50) ? 1.0 : 0.0;
	//v_DiscardFragments = a_ProfileIndex;

	// check validity (may be omitted later when we are sure it won't be broken)
	if (a_ProfileIndex < u_NumProfiles) {
		float CCL = u_ProfileCCLs[a_ProfileIndex];
		
		float diff = a_Pos.y - CCL;

		if (u_OpacityBlendMode == 0) {
			// This approach changes opacity only above CCL
			if (diff <= 0.0) {
				v_ComputedOpacity = 0.0;
			} else {
				v_ComputedOpacity = max(diff / u_OpacityBlendRange, 1.0);
			}
		} else {
			// This approach changes opacity from (CCL - range) to (CCL + range)
			if (abs(diff) <= u_OpacityBlendRange) {
				v_ComputedOpacity = (diff / u_OpacityBlendRange) * 0.5 + 0.5;
			} else {
				v_ComputedOpacity = (diff < 0.0) ? 0.0 : 1.0;
			}
		}




		
	}


	/*
	// This may be better looking in fragment shader
	// testing opacity fade
	float CCL = 20.0;
	float range = 10.0;

	float diff = a_Pos.y - CCL;
	if (abs(diff) <= range) {
		v_ComputedOpacity = (diff / range) * 0.5 + 0.5;
	} else {
		v_ComputedOpacity = (diff < 0.0) ? 0.0 : 1.0;
	}
	*/
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

float goldNoise(in vec2 coordinate, in float seed){
    return fract(tan(distance(coordinate*(seed+PHI), vec2(PHI, PI)))*SQ2);
}
