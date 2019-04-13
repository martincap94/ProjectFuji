#version 400 core

uniform sampler2D u_InputTexture;

uniform vec2 u_TexelSize;

uniform float u_BlurAmount;

layout (location = 0) out vec4 fragColor;

in vec2 v_TexCoords;


void main(void) {
	vec4 finalColor;

	//fragColor = texture(u_InputTexture, v_TexCoords);
	//return;

	finalColor = texture(u_InputTexture, v_TexCoords + vec2(0.5, 0.5) * u_TexelSize * u_BlurAmount);
	finalColor += texture(u_InputTexture, v_TexCoords + vec2(-0.5, 0.5) * u_TexelSize * u_BlurAmount);
	finalColor += texture(u_InputTexture, v_TexCoords + vec2(0.5, -0.5) * u_TexelSize * u_BlurAmount);
	finalColor += texture(u_InputTexture, v_TexCoords + vec2(-0.5, -0.5) * u_TexelSize * u_BlurAmount);
	fragColor = finalColor * 0.25;
}
