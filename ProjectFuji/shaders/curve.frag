#version 330 core

in vec3 vPos;

out vec4 fragColor;

uniform vec3 color;


void main() {

	if (vPos.x < 0.0 || vPos.x > 1.0) {
		//discard;
	}

	fragColor = vec4(color, 1.0);
}
