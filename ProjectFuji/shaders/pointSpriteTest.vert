#version 330 core

layout (location = 0) in vec3 a_Pos;

uniform mat4 u_View;
uniform mat4 u_Projection;

uniform vec3 u_CameraPos;
uniform float u_PointSizeModifier;


void main() {
	float cameraDist = distance(a_Pos, u_CameraPos);
	float pointScale = u_PointSizeModifier * 100.0 / cameraDist;
	gl_Position = u_Projection * u_View * vec4(a_Pos, 1.0);
	gl_PointSize = pointScale;

}

