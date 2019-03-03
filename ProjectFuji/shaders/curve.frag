#version 330 core

in vec3 v_Pos;

out vec4 fragColor;

uniform vec3 color;
uniform bool u_CropBounds;


void main() {

	if (u_CropBounds && (v_Pos.x < 0.0 || v_Pos.x > 1.0)) {
		discard;
	}

	fragColor = vec4(color, 1.0);
}
