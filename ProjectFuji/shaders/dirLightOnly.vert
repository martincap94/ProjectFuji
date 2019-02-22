#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 vFragPos;
out vec3 vNormal;


uniform mat4 uView;
uniform mat4 uProjection;

void main() {
	vNormal = aNormal;
	vFragPos = aPos;
	gl_Position = uProjection * uView * vec4(aPos, 1.0);
}

