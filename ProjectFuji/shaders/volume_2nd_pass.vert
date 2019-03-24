#version 430 core

layout (location = 0) in vec4 a_Pos;


layout (location = 5) in int a_ProfileIndex;


uniform mat4 u_View;
uniform mat4 u_Projection;

uniform vec3 u_CameraPos;
uniform float u_WorldPointSize;

//uniform mat4 u_LightSpaceView;
//uniform mat4 u_LightSpaceProjection;

uniform mat4 u_LightSpaceMatrix;

out vec4 v_LightSpacePos;


void main() {
	float cameraDist = distance(vec3(a_Pos), u_CameraPos);
	float pointScale = u_WorldPointSize * 100.0 / cameraDist;
	//v_LightSpacePos = u_LightSpaceProjection * u_LightSpaceView * a_Pos; // matrix multiplication can be moved to CPU
	v_LightSpacePos = u_LightSpaceMatrix * a_Pos;
	//gl_Position = v_LightSpacePos; // testing correctness

	gl_Position = u_Projection * u_View * a_Pos;
	gl_PointSize = pointScale;
	
}


