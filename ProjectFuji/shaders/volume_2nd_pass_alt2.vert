#version 430 core

layout (location = 0) in vec4 a_Pos;
layout (location = 5) in int a_ProfileIndex;


void main() {
	gl_Position = a_Pos;
}


