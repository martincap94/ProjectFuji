#version 400 core


layout (location = 0) out vec4 FragColor;

in vec4 v_LightSpacePos;

uniform sampler2D u_Texture;

const float solidAngle = 0.001;

void main(void) {


	//FragColor = vec4(0.0, 0.0, 0.0, 1.0);

	FragColor = texture(u_Texture, gl_PointCoord);

}
