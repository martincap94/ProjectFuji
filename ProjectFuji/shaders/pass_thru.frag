#version 330 core

layout (location = 0) out vec4 fragColor;

in vec2 v_TexCoords;

uniform sampler2D u_Texture;


void main(void) {
	fragColor = vec4(texture(u_Texture, v_TexCoords).rgb, 1.0);
}
