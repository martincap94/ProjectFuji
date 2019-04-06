#version 330 core

out vec4 fragColor;

in vec4 v_FragPos;

uniform vec3 u_ViewPos;



void main() {
	
	fragColor = vec4(v_FragPos.xyz, 1.0);

}

