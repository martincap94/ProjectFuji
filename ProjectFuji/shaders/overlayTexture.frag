#version 400 core

uniform sampler2D u_Texture;

uniform bool u_ShowAlphaChannel = true;

layout (location = 0) out vec4 FragColor;

in vec2 v_TexCoords;


void main(void) {
	if (u_ShowAlphaChannel) {
		FragColor = texture(u_Texture, v_TexCoords);
	} else {
		FragColor = vec4(texture(u_Texture, v_TexCoords).rgb, 1.0);
	}
}
