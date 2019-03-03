#version 330 core

out vec4 fragColor;

uniform vec3 u_Color;

uniform sampler2D uTex;


void main() {
	//fragColor = vec4(u_Color, 1.0);
	//fragColor = vec4(gl_PointCoord, 0.0, 1.0);
	fragColor = texture(uTex, gl_PointCoord);
	if (fragColor.a <= 0.4) {
		discard;
	}
}
