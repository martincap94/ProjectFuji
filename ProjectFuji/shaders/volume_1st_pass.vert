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
	
	float pointScale;
	if (u_Mode == 0) {
		float lightDist = distance(vec3(a_Pos), u_LightPos);
		pointScale = u_WorldPointSize * 100.0 / lightDist;
	} else {
		float cameraDist = distance(vec3(a_Pos), u_CameraPos);
		pointScale = u_WorldPointSize * 100.0 / cameraDist;
	}

	gl_Position = u_Projection * u_View * a_Pos;
	gl_PointSize = pointScale;


}
