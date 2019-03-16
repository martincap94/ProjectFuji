#version 400 core

uniform sampler2D u_Texture;

layout (location = 0) out vec4 FragColor;

in vec2 v_TexCoords;


void main(void) {
	//vec3 finalColor = vec3(1.0, 0.0, 0.0);
	vec3 finalColor = vec3(0.0);
	finalColor += texture(u_Texture, v_TexCoords).rgb;
	//finalColor += vec3(v_TexCoords, 0.0);
	FragColor = vec4(finalColor, 1.0);
}
