#version 330 core

out vec4 fragColor;

uniform vec3 u_Color;


void main() {
	fragColor = vec4(u_Color, 1.0);
}
