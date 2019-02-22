#version 400 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vColor;

void main() {
	vColor = aColor;
	gl_Position = uProjection * uView * vec4(aPos, 1.0);
}

