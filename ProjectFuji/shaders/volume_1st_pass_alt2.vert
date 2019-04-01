#version 330 core

layout (location = 0) in vec4 a_Pos;

layout (location = 5) in int a_ProfileIndex;


uniform mat4  u_View;
uniform mat4  u_Projection;

uniform vec3 u_LightPos;
uniform vec3 u_CameraPos;
uniform float u_WorldPointSize;

uniform int u_Mode;


void main(void) {
	gl_Position = a_Pos;
}
