#version 400 core

uniform vec2  u_Exponents = vec2(4.0, 4.0);

layout (location = 0) out vec4 FragColor;

in vec4 v_LightSpacePos;

void main(void) {

	float depth = v_LightSpacePos.z / v_LightSpacePos.w;
	//depth = depth * 0.5 + 0.5; // move to [0,1]
	// here, we want depth in range [-1, 1]

	// ===================================================
	float pos = exp(u_Exponents.x * depth);
	float neg = -exp(-u_Exponents.y * depth);
	// ===================================================

	FragColor = vec4(pos, pos * pos, neg, neg * neg);
	//FragColor = vec4(pos, pos * pos, depth, depth); // testing

}
