#version 400 core

out vec4 fragColor;

in vec3 v_Color;


void main() {
	fragColor = vec4(v_Color, 1.0);
}
