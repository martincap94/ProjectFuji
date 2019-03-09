#version 330 core

in float v_ParticleOpacityMultiplier;
in float v_ComputedOpacity;

out vec4 fragColor;

uniform vec3 u_Color;

uniform sampler2D u_Tex;

uniform float u_OpacityMultiplier;


float goldNoise(in vec2 coordinate, in float seed);


void main() {

	if (v_ComputedOpacity == 0.0) {
		discard;
	}


	//fragColor = vec4(u_Color, 1.0);
	//fragColor = vec4(gl_PointCoord, 0.0, 1.0);
	fragColor = texture(u_Tex, gl_PointCoord);
	fragColor.a *= u_OpacityMultiplier * v_ComputedOpacity;

	//fragColor.a *= v_ParticleOpacityMultiplier;
	//fragColor *= goldNoise(gl_FragCoord.xy, gl_FragCoord.z);


	//if (fragColor.a < 1.0) {
	//	fragColor = vec4(1.0, 0.0, 0.0, 1.0);
	//} else {
	//	fragColor = vec4(0.0, 1.0, 0.0, 1.0);
	//}


	//if (fragColor.a <= 0.05) {
	//	discard;
	//}
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

//void mainImage(out vec4 fragColor, in vec2 fragCoord){    
//	fragColor  = vec4(gold_noise(fragCoord, iTime));
//}