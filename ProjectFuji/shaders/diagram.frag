#version 400 core

uniform sampler2D u_InputTexture;

layout (location = 0) out vec4 FragColor;

in vec2 v_TexCoords;


void main(void) {
	vec3 finalColor = vec3(0.0, 0.0, 0.0);
	finalColor += texture(u_InputTexture, v_TexCoords).rgb;
	FragColor = vec4(finalColor, 1.0);
}
