#version 400 core

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec2 v_TexCoords;
in mat3 v_TBN;
in vec4 v_LightSpacePos;

out vec4 fragColor;

uniform vec3 u_ViewPos;

struct Material {
	sampler2D albedo;
	sampler2D metallicRoughness;
	sampler2D normalMap;
	sampler2D ao;
	float tiling;
};



uniform Material u_Material;

struct Fog {
	float intensity;
	float minDistance;
	float maxDistance;
	vec4 color; // alpha could be used instead of intensity

	int mode;
	float expFalloff;
};

uniform Fog u_Fog;

struct DirLight {
    vec3 direction;
    
	vec3 color;
	float intensity;
};

uniform DirLight u_DirLight;

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutoff;
    float outerCutoff;
    
	vec3 color;
	float intensity;
    
    float constant;
    float linear;
    float quadratic;
};

uniform SpotLight u_SpotLight;

struct PointLight {
    vec3 position;
    
	vec3 color;
	float intensity;
    
    float constant;
    float linear;
    float quadratic;
};



uniform sampler2D u_DepthMapTexture;

uniform float u_VarianceMinLimit;
uniform float u_LightBleedReduction;
uniform vec2 u_Exponents = vec2(40.0, 40.0);
uniform int u_EVSMMode = 0;
uniform float u_ShadowBias;

uniform bool u_ShadowOnly;
uniform float u_ShadowDamping;



#define NR_POINT_LIGHTS 8
uniform PointLight u_PointLights[NR_POINT_LIGHTS];

uniform int u_NumActivePointLights = 0;

uniform bool u_SpotLightEnabled;


float calcShadow(vec4 fragLightSpacePos);
float chebyshev(vec2 moments, float depth);
float reduceLightBleed(float p_max, float amount);
float linstep(float minVal, float maxVal, float val);


// Based on tutorial from learnopengl.com: https://learnopengl.com/PBR/Lighting
//		by Joey De Vries
float distributionGGX(vec3 N, vec3 H, float roughness);
float geometrySchlickGGX(float NdotV, float roughness);
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness);
vec3 fresnelSchlick(float cosTheta, vec3 F0);


const float PI = 3.14159265359;


void main() {

	
	//vec3 N = texture(u_Material.normalMap, v_TexCoords).rgb;
	vec3 N = texture(u_Material.normalMap, v_TexCoords).rgb;
	N = normalize(N * 2.0 - 1.0);
	N = normalize(v_TBN * N); 
	
	//vec3 N = normalize(v_Normal);

    vec3 V = normalize(u_ViewPos - v_WorldPos);

	float shadow = calcShadow(v_LightSpacePos); // directional light only

	if (u_ShadowOnly) {
		fragColor = vec4(vec3(shadow), 1.0);
		return;
	}

	vec3 result;


	vec3 albedo     = pow(texture(u_Material.albedo, v_TexCoords).rgb, vec3(2.2));
	//vec3 albedo = texture(albedoTex, v_TexCoords).rgb;
    float metallic  = texture(u_Material.metallicRoughness, v_TexCoords).r;
    float roughness = texture(u_Material.metallicRoughness, v_TexCoords).g;
    //float ao        = texture(aoTex, v_TexCoords).r;
	float ao        = pow(texture(u_Material.ao, v_TexCoords).r, 2.2);


	// PBR
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);

	vec3 L = normalize(-u_DirLight.direction);
	vec3 H = normalize(L + V);
	
	vec3 radiance = u_DirLight.color * u_DirLight.intensity; // no attenuation for directional light

	float NDF = distributionGGX(N, H, roughness);
	float G = geometrySmith(N, V, L, roughness);
	vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - metallic;

	vec3 numerator = NDF * G * F;
	float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
	vec3 specular = numerator / max(denominator, 0.001); // check the max later

	float NdotL = max(dot(N, L), 0.0);
	vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

	//vec3 ambient = vec3(0.03) * albedo * ao;
	//vec3 color = ambient + Lo;
	vec3 color = Lo * ao;




	result = color * min(shadow + u_ShadowDamping, 1.0);
	
    
    fragColor = vec4(result, 1.0);

	float dist = distance(v_WorldPos.xyz, u_ViewPos);
	if (u_Fog.mode == 0) {
		float t = (dist - u_Fog.minDistance) / (u_Fog.maxDistance - u_Fog.minDistance);
		fragColor = mix(fragColor, u_Fog.color, clamp(t * u_Fog.intensity, 0.0, 1.0));
	} else if (u_Fog.mode == 1) {
		float t = exp(-u_Fog.expFalloff * dist * u_Fog.expFalloff * dist);
		fragColor = mix(fragColor, u_Fog.color, clamp((1.0 - t) * u_Fog.intensity, 0.0, 1.0));
	}

	// gamma correction
	float gamma = 2.2;
    fragColor.rgb = pow(fragColor.rgb, vec3(1.0/gamma));
	
    
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




float distributionGGX(vec3 N, vec3 H, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;
	//float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return a2 / denom;
	
}


float geometrySchlickGGX(float NdotV, float roughness) {
	float r = (roughness + 1.0);
	float k = (r * r) * 0.125; // (r * r ) / 8.0; use MAD
	float num = NdotV;
	float denom = NdotV * (1.0 - k) + k;
	return num / denom;

} 


float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = geometrySchlickGGX(NdotV, roughness);
	float ggx1 = geometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);

}
































