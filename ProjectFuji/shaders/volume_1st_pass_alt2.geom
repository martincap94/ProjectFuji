#version 400 core

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

uniform mat4  u_View;
uniform mat4  u_Projection;

uniform mat4 u_LightSpaceMatrix;


uniform vec3 u_CameraPos;
uniform vec3 u_LightPos;

uniform int u_Mode;

uniform float u_WorldPointSize;

const vec3 worldup = vec3(0.0, 1.0, 0.0);

out vec2 g_TexCoords;

float goldNoise(in vec2 coordinate, in float seed);

void main() {
	vec3 pos = gl_in[0].gl_Position.xyz;
	mat4 VP = u_Projection * u_View;

	float tmpscale = u_WorldPointSize / 10.0;

	vec3 right;
	vec3 up;

	if (u_Mode == 0) {
		right = normalize(vec3(goldNoise(pos.xy, pos.x), goldNoise(pos.xy, pos.y), goldNoise(pos.xy, pos.z))) * tmpscale;
		up = normalize(cross(right, worldup)) * tmpscale;
	} else {
		vec3 toLight = normalize(u_LightPos - pos);
		right = normalize(cross(toLight, worldup)) * tmpscale;
		up = normalize(cross(right, toLight)) * tmpscale;
	}


	gl_Position = VP * vec4(pos - right - up, 1.0);
	g_TexCoords = vec2(0.0, 0.0);
	EmitVertex();

	gl_Position = VP * vec4(pos - right + up, 1.0);
	g_TexCoords = vec2(0.0, 1.0);
	EmitVertex();

	gl_Position = VP * vec4(pos + right - up, 1.0);
	g_TexCoords = vec2(1.0, 0.0);
	EmitVertex();

	gl_Position = VP * vec4(pos + right + up, 1.0);
	g_TexCoords = vec2(1.0, 1.0);
	EmitVertex();

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



