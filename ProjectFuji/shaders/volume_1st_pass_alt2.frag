#version 400 core


layout (location = 0) out vec4 fragColor;

in vec2 g_TexCoords;

uniform sampler2D u_Texture;
uniform vec4 u_Color;

uniform float u_Opacity; // misleading (same name as in second pass even though they have completely different purpose)
uniform float u_ShadowAlpha = 0.005;


void main(void) {

	fragColor = vec4(1.0, 0.0, 0.0, 1.0);

	vec4 texColor = texture(u_Texture, g_TexCoords);
	fragColor = vec4(texColor.rgb * texColor.a, texColor.a) * u_Opacity;
	fragColor.xyz *= u_ShadowAlpha;

}
