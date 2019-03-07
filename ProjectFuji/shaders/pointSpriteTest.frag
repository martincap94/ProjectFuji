#version 330 core

out vec4 fragColor;

uniform vec3 u_Color;

uniform sampler2D u_Tex;

uniform float u_OpacityMultiplier;

void main() {
	//fragColor = vec4(u_Color, 1.0);
	//fragColor = vec4(gl_PointCoord, 0.0, 1.0);
	fragColor = texture(u_Tex, gl_PointCoord);
	fragColor.a *= u_OpacityMultiplier;


	//if (fragColor.a < 1.0) {
	//	fragColor = vec4(1.0, 0.0, 0.0, 1.0);
	//} else {
	//	fragColor = vec4(0.0, 1.0, 0.0, 1.0);
	//}


	//if (fragColor.a <= 0.05) {
	//	discard;
	//}
}
