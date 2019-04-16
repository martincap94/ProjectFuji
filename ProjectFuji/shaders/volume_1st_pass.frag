#version 400 core


layout (location = 0) out vec4 fragColor;



uniform sampler2D u_Texture;
uniform vec4 u_Color;

uniform float u_Opacity; // misleading (same name as in second pass even though they have completely different purpose)
uniform float u_ShadowAlpha = 0.005;


void main(void) {
	vec4 texColor = texture(u_Texture, gl_PointCoord);
	fragColor = vec4(texColor.rgb * texColor.a, texColor.a) * u_Opacity;
	fragColor.xyz *= u_ShadowAlpha;
}
