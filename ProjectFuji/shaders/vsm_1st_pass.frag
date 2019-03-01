#version 400 core

layout (location = 0) out vec4 FragColor;

in vec4 v_LightSpacePos;

void main(void) {

	float depth = v_LightSpacePos.z / v_LightSpacePos.w;
	depth = depth * 0.5 + 0.5; // move to [0,1]

	float dx = dFdx(depth);
	float dy = dFdy(depth);
	
	//float firstMoment = depth; // redundant
	float secondMoment = depth * depth + 0.25 * (dx * dx + dy * dy);

	FragColor = vec4(depth, secondMoment, 0.0, 1.0);
	//FragColor = vec4(depth * 10.0); // testing
	//FragColor = vec4(10.0f * vec3(depth), 1.0);


}
