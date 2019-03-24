#version 330 core

in vec4 v_LightSpacePos;

out vec4 fragColor;

uniform vec3 u_TintColor;

uniform sampler2D u_Texture;
uniform sampler2D u_ShadowTexture;

uniform float u_Opacity;
uniform bool u_ShowHiddenParticles;



void main() {


	fragColor = texture(u_Texture, gl_PointCoord);
	fragColor.xyz *= u_TintColor;

	fragColor.w *= u_Opacity;


	vec3 projCoords = v_LightSpacePos.xyz / v_LightSpacePos.w;
	projCoords = projCoords * 0.5 + vec3(0.5);
	vec3 shadow = vec3(1.0) - texture(u_ShadowTexture, projCoords.xy).xyz;

	//fragColor = vec4(shadow, fragColor.a);
	fragColor.xyz *= shadow * fragColor.w;


}
