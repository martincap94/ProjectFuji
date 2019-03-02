#version 330 core

out vec4 fragColor;

in vec4 v_FragPos;
in vec3 v_Normal;
in vec4 v_LightSpacePos;

struct DirLight {
	vec3 direction;

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

uniform DirLight dirLight;

uniform vec3 v_ViewPos;

uniform sampler2D u_DepthMapTexture;

uniform float u_VarianceMinLimit;
uniform float u_LightBleedReduction;
uniform vec2 u_Exponents = vec2(40.0, 40.0);
uniform int u_EVSMMode = 0;
uniform float u_ShadowBias;

vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir);

float calcShadowBasic(vec4 fragLightSpacePos);
float calcShadow(vec4 fragLightSpacePos);
float chebyshev(vec2 moments, float depth);
float reduceLightBleed(float p_max, float amount);
float linstep(float minVal, float maxVal, float val);

void main() {

	vec3 norm = normalize(v_Normal);
	vec3 viewDir = normalize(v_FragPos.xyz - v_ViewPos);
	

	vec3 color = calcDirLight(dirLight, norm, viewDir);

	float shadow = calcShadow(v_LightSpacePos);

	vec3 result;
	
	
	//result = color *  vec3(shadow);
	result = color * shadow;


	//vec3 projCoords = v_LightSpacePos.xyz / v_LightSpacePos.w;
	//projCoords = projCoords * 0.5 + vec3(0.5);
	//vec4 moments = texture(u_DepthMapTexture, projCoords.xy); // pos, pos^2, neg, neg^2

	//result = vec3(moments.xy, 0.0);




	fragColor = vec4(result, 1.0);
}


vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
	vec3 matColor = vec3(0.5, 0.4, 0.4);

    // combine results
    vec3 ambient  = light.ambient  * matColor;
    vec3 diffuse  = light.diffuse  * diff /** matColor*/;
    vec3 specular = light.specular * spec /** matColor */;

    
    return (ambient + diffuse + specular);
}


float calcShadowBasic(vec4 fragLightSpacePos) {
	
	vec3 projCoords = fragLightSpacePos.xyz / fragLightSpacePos.w;

	vec4 moments = texture(u_DepthMapTexture, projCoords.xy);


	return (moments.z < projCoords.z) ? 0.0 : 1.0;

}


float calcShadow(vec4 fragLightSpacePos) {
	
	vec3 projCoords = fragLightSpacePos.xyz / fragLightSpacePos.w;

	// compute pos and neg after bias adjustment
	//float pos = exp(u_Exponents.x * projCoords.z);
	//float neg = -exp(-u_Exponents.y * projCoords.z);

	//float bias = 0.0;
	projCoords.z -= u_ShadowBias; // z bias
	projCoords = projCoords * 0.5 + vec3(0.5);

	//if (projCoords.z >= 1.0) {
	//	return 1.0;
	//}
	float shadow = 0.0;

	vec4 moments = texture(u_DepthMapTexture, projCoords.xy); // pos, pos^2, neg, neg^2

	projCoords = projCoords* 2.0 - 1.0;

	float pos = exp(u_Exponents.x * projCoords.z);
	float neg = -exp(-u_Exponents.y * projCoords.z);

		
	if (u_EVSMMode == 0) {
		shadow = chebyshev(moments.xy, pos);
		return shadow;
	} else {
		float posShadow = chebyshev(moments.xy, pos);
		float negShadow = chebyshev(moments.zw, neg);
		shadow = min(posShadow, negShadow);
		return shadow;
	}
}

float chebyshev(vec2 moments, float depth) {

	if (depth <= moments.x) {
		return 1.0;
	}

	float variance = moments.y - (moments.x * moments.x);
	variance = max(variance, u_VarianceMinLimit / 1000.0);

	float d = depth - moments.x; // attenuation
	float p_max = variance / (variance + d * d);

	return reduceLightBleed(p_max, u_LightBleedReduction);
}

float reduceLightBleed(float p_max, float amount) {
	return linstep(amount, 1.0, p_max);
}

float linstep(float minVal, float maxVal, float val) {
	return clamp((val - minVal) / (maxVal - minVal), 0.0, 1.0);
}