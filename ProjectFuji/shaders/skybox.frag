#version 330 core

in vec3 texCoord;

out vec4 fragColor;

uniform samplerCube skybox;



void main() {

	//fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    fragColor = texture(skybox, texCoord);
    
}
























