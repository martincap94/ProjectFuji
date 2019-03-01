#version 400 core

layout (location = 0) out vec4 FragColor;

in vec4 v_LightSpacePos;

void main(void) {
	FragColor = vec4(length(v_LightSpacePos.xyz), dot(v_LightSpacePos.xyz, v_LightSpacePos.xyz), gl_FragCoord.z, 1.0); // based on seminar
	//gl_FragDepth = gl_FragCoord.z; // compilator does by itself - if defined: prevents early z-test
}
