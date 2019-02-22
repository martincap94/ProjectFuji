#version 330 core


uniform vec3 uColor;

in vec3 vColor;

out vec4 fragColor;


void main() {
	fragColor = vec4(vColor, 1.0);
}
