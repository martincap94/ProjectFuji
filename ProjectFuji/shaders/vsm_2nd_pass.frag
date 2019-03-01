#version 400 core

layout (location = 0) out vec4 FragColor;

in vec3 v_Normal;
in vec4 v_Vertex;
in vec4 v_LightSpacePos;

uniform int		  u_ShadowOnly = 1;
uniform float	  u_ShadowBias;

uniform float	  u_VarianceMinLimit;
uniform float	  u_LightBleedReduction;

uniform sampler2D u_DepthMapTexture;

float calcShadow(vec4 fragLightSpacePos);
float chebyshev(vec2 moments, float depth);
float reduceLightBleed(float p_max, float amount);
float linstep(float minVal, float maxVal, float val);

void main() {
// Compute fragment diffuse color
    //vec3 N = normalize(v_Normal);
    //vec3 L = normalize(u_LightPosition.xyz - v_Vertex.xyz);
    //float NdotL = max(dot(N, L), 0.0);
    //vec4 color = texture(u_SceneTexture, v_TexCoord) * NdotL;
		
	vec4 color = vec4(1.0);

    //vec4 shadow = vec4(1.0);
    // TODO: implement shadow generation 
        // 1. Compute correct tex. coordinates to depth map texture
        //      vec2 texCoord = -v_LightSpacePos.xy / v_LightSpacePos.z;
        // 2. Read depth from depth map 
        // 3. Compare fragment's depth with value from depth map texture
	float shadow = calcShadow(v_LightSpacePos);


	//vec3 projCoords = v_LightSpacePos.xyz / v_LightSpacePos.w;

	//projCoords.z -= u_ShadowBias; // z bias
	//projCoords = projCoords * 0.5 + 0.5;

	//FragColor = vec4(projCoords, 1.0);


    // Modulate fragment's color according to result of shadow test
	if (u_ShadowOnly == 1) {
		FragColor = vec4(vec3(shadow), 1.0);
	} else {
		FragColor = color * vec4(vec3(shadow), 1.0);
	}
}



float calcShadow(vec4 fragLightSpacePos) {
	
	vec3 projCoords = fragLightSpacePos.xyz / fragLightSpacePos.w;

	projCoords.z -= u_ShadowBias; // z bias
	projCoords = projCoords * 0.5 + 0.5;

	float shadow = 0.0;

	vec2 moments = texture(u_DepthMapTexture, projCoords.xy).rg;



	shadow = chebyshev(moments, projCoords.z);
	return shadow;

}

float chebyshev(vec2 moments, float depth) {

	if (depth <= moments.x) {
		return 1.0;
	}

	float variance = moments.y - (moments.x * moments.x);
	variance = max(variance, u_VarianceMinLimit / 1000.0);

	float d = depth - moments.x; // attenuation
	float p_max = variance / (variance + d * d);

	//return p_max;
	return reduceLightBleed(p_max, u_LightBleedReduction);
}

float reduceLightBleed(float p_max, float amount) {
	return linstep(amount, 1.0, p_max);
}

float linstep(float minVal, float maxVal, float val) {
	return clamp((val - minVal) / (maxVal - minVal), 0.0, 1.0);
}