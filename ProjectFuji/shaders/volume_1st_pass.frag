#version 400 core


layout (location = 0) out vec4 fragColor;

in vec4 v_LightSpacePos;

uniform sampler2D u_Texture;


uniform float u_Opacity;


void main(void) {

	vec4 texColor = texture(u_Texture, gl_PointCoord);
	fragColor = vec4(texColor.rgb * texColor.a, texColor.a) * u_Opacity;

}
