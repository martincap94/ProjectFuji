#version 400 core

uniform sampler2D u_InputTexture;

uniform vec2 u_TexelSize;

layout (location = 0) out vec4 fragColor;

in vec2 v_TexCoords;


void main(void) {
	vec4 finalColor;
	finalColor = texture(u_InputTexture, v_TexCoords + vec2(0.5, 0.5) * u_TexelSize);
	finalColor += texture(u_InputTexture, v_TexCoords + vec2(-0.5, 0.5) * u_TexelSize);
	finalColor += texture(u_InputTexture, v_TexCoords + vec2(0.5, -0.5) * u_TexelSize);
	finalColor += texture(u_InputTexture, v_TexCoords + vec2(-0.5, -0.5) * u_TexelSize);
	fragColor = vec4(finalColor * 0.25);
}
