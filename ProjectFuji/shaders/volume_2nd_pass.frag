#version 330 core

in float v_ParticleOpacityMultiplier;

out vec4 fragColor;

uniform vec3 u_TintColor;

uniform sampler2D u_Texture;

uniform float u_Opacity;
uniform bool u_ShowHiddenParticles;



void main() {


	fragColor = texture(u_Texture, gl_PointCoord);
	fragColor.xyz *= u_TintColor;

	fragColor.a *= u_Opacity;

}
