#version 330 core

in vec4 v_LightSpacePos;

out vec4 fragColor;

uniform vec3 u_TintColor;

uniform sampler2D u_Texture;
uniform sampler2D u_ShadowTexture;

uniform float u_Opacity;
uniform bool u_ShowHiddenParticles;

uniform int u_Mode = 0;

void main() {

	if (u_Mode == 0) {
		fragColor = texture(u_Texture, gl_PointCoord);
		fragColor.xyz *= u_TintColor;

		fragColor.w *= u_Opacity;

		// this is probably the source of our problems since each fragment accesses the same shadow texel -> we need to transform the fragment position, not the vertex position (into the light space), do we need a geometry shader??? or reverse transformation of the gl_FragCoord?
		vec3 projCoords = v_LightSpacePos.xyz / v_LightSpacePos.w;
		projCoords = projCoords * 0.5 + vec3(0.5);
		vec3 shadow = vec3(1.0) - texture(u_ShadowTexture, projCoords.xy).xyz;

		//fragColor = vec4(shadow, fragColor.a);
		fragColor.xyz *= shadow * fragColor.w;

	} else if (u_Mode == 1) {


		fragColor = texture(u_Texture, gl_PointCoord);


		
	}

}
