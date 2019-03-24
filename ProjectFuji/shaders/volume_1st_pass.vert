#version 330 core

layout (location = 0) in vec4 a_Pos;

layout (location = 5) in int a_ProfileIndex;


uniform mat4  u_View;
uniform mat4  u_Projection;

uniform vec3 u_LightPos;
uniform float u_WorldPointSize;


out vec4 v_LightSpacePos;

void main(void) {
	
	float lightDist = distance(vec3(a_Pos), u_LightPos);
	float pointScale = u_WorldPointSize * 100.0 / lightDist;

	v_LightSpacePos = u_Projection * u_View * a_Pos;
	gl_Position = v_LightSpacePos;
	gl_PointSize = pointScale;


}
