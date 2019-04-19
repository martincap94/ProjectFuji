#version 330 core

out vec4 fragColor;

in vec4 v_FragPos;
in vec3 v_Normal;
in vec4 v_LightSpacePos;
in vec4 v_PrevLightSpacePos;
in vec2 v_TexCoords;

in mat3 v_TBN;

struct DirLight {
	vec3 direction;

	vec3 color;
	float intensity;
};

uniform DirLight u_DirLight;

struct Material {
	sampler2D albedo;
	sampler2D metallicRoughness;
	sampler2D normalMap;
	sampler2D ao;
	float tiling;
};


uniform Material[4] u_Materials;

uniform sampler2D u_MaterialMap;


struct Fog {
	float intensity;
	float minDistance;
	float maxDistance;
	vec4 color; // alpha could be used instead of intensity
	
	int mode;
	float expFalloff;
};

uniform Fog u_Fog;

uniform vec3 u_ViewPos;

uniform sampler2D u_DepthMapTexture;
uniform sampler2D u_CloudShadowTexture; // texture unit: TEXTURE_UNIT_CLOUD_SHADOW_MAP

uniform bool u_CloudsCastShadows;
uniform float u_CloudCastShadowAlphaMultiplier;

uniform sampler2D u_TerrainNormalMap;
uniform float u_GlobalNormalMapMixingRatio;
uniform float u_GlobalNormalMapTiling = 1.0;

uniform bool u_UseGrungeMap = false;
uniform sampler2D u_GrungeMap;
uniform float u_GrungeMapMin = 1.0;
uniform float u_GrungeMapTiling = 1.0;

uniform float u_UVRatio;


//uniform sampler2D u_DiffuseTexture;

uniform float u_VarianceMinLimit;
uniform float u_LightBleedReduction;
uniform vec2 u_Exponents = vec2(40.0, 40.0);
uniform int u_EVSMMode = 0;
uniform float u_ShadowBias;

uniform bool u_ShadowOnly;
uniform float u_ShadowDamping;


uniform bool u_NormalsOnly;
uniform int u_NormalsMode = 0;
uniform bool u_VisualizeTextureMode;


const float PI = 3.14159265359;



// Based on tutorial from learnopengl.com: https://learnopengl.com/PBR/Lighting
//		by Joey De Vries
float distributionGGX(vec3 N, vec3 H, float roughness);
float geometrySchlickGGX(float NdotV, float roughness);
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness);
vec3 fresnelSchlick(float cosTheta, vec3 F0);


vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec3 albedo, vec2 ms, float ao);

float calcShadow(vec4 fragLightSpacePos);
float chebyshev(vec2 moments, float depth);
float reduceLightBleed(float p_max, float amount);
float linstep(float minVal, float maxVal, float val);
float linstepMax(float val, float maxVal);
vec4 linstepMaxVec4(vec4 val, float maxVal);

vec3 computeNormalVec(vec4 materialContributions);
vec4 computeAlbedo(vec4 materialContributions);
vec2 computeMetallicRoughness(vec4 materialContributions);
float computeAmbientOcclusion(vec4 materialContributions);


vec2 getTexCoords(int materialIdx);

void main() {


	if (u_VisualizeTextureMode) {
		fragColor = texture(u_MaterialMap, v_TexCoords);
		return;
	}

	vec4 materialMap = texture(u_MaterialMap, v_TexCoords);
	materialMap.a = 1.0 - materialMap.a;
	vec4 materialContributions = linstepMaxVec4(materialMap, dot(vec4(1.0), materialMap));


	vec3 norm;

	if (u_NormalsMode == 0) {
		norm = computeNormalVec(materialContributions);

		norm = mix(norm, texture(u_TerrainNormalMap, u_GlobalNormalMapTiling * vec2(u_UVRatio * v_TexCoords.x, v_TexCoords.y)).rgb, u_GlobalNormalMapMixingRatio);

		norm = normalize(norm);
		norm = normalize(norm * 2.0 - 1.0);
		norm = normalize(v_TBN * norm);
	} else {
		norm = normalize(v_Normal);
	}


	if (u_NormalsOnly) {
		fragColor = vec4(norm, 1.0);
		return;
	}
	


	vec3 viewDir = normalize(u_ViewPos - v_FragPos.xyz);

	float shadow = calcShadow(v_LightSpacePos);

	vec3 result;

	if (u_ShadowOnly) {
		result = vec3(shadow);
	} else {

		vec3 albedo = vec3(computeAlbedo(materialContributions));
		vec2 metallicRoughness = computeMetallicRoughness(materialContributions);
		float ambientOcclusion = computeAmbientOcclusion(materialContributions);

		vec3 color = calcDirLight(u_DirLight, norm, viewDir, albedo, metallicRoughness, ambientOcclusion);
		result = color * min(shadow + u_ShadowDamping, 1.0);
	}



	
	if (u_CloudsCastShadows) {

		vec3 projCoords = v_PrevLightSpacePos.xyz / v_PrevLightSpacePos.w;
		projCoords = projCoords * 0.5 + vec3(0.5);

		vec4 cloudShadowVals =  texture(u_CloudShadowTexture, projCoords.xy);
		vec3 cloudShadow = vec3(1.0) - (cloudShadowVals.a * cloudShadowVals.xyz * vec3(u_CloudCastShadowAlphaMultiplier));

		//fragColor = vec4(cloudShadow, 1.0);
		//return;

		result *= cloudShadow;

	}
	

	fragColor = vec4(result, 1.0);


	
	float dist = distance(v_FragPos.xyz, u_ViewPos);
	if (u_Fog.mode == 0) {
		float t = (dist - u_Fog.minDistance) / (u_Fog.maxDistance - u_Fog.minDistance);
		fragColor = mix(fragColor, u_Fog.color, clamp(t * u_Fog.intensity, 0.0, 1.0));
	} else if (u_Fog.mode == 1) {
		float t = exp(-u_Fog.expFalloff * dist * u_Fog.expFalloff * dist);
		fragColor = mix(fragColor, u_Fog.color, clamp((1.0 - t) * u_Fog.intensity, 0.0, 1.0));
	}
	
	float gamma = 2.2;
    fragColor.rgb = pow(fragColor.rgb, vec3(1.0/gamma));


}


vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec3 albedo, vec2 ms, float ao) {
    vec3 lightDir = normalize(-light.direction);


	vec3 halfVec = normalize(lightDir + viewDir);

	vec3 radiance = light.intensity * light.color;

	//return radiance;
	
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, ms.x);
	
	float roughness = ms.y;

	float NDF = distributionGGX(normal, halfVec, roughness);
	float G = geometrySmith(normal, viewDir, lightDir, roughness);
	vec3 F = fresnelSchlick(max(dot(halfVec, viewDir), 0.0), F0);
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - ms.x;

	vec3 numerator = NDF * G * F;
	float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
	vec3 specular = numerator / max(denominator, 0.001); // check the max later

	float NdotL = max(dot(normal, lightDir), 0.0);
	vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

	//vec3 ambient = vec3(0.03) * albedo * ao;


	//vec3 color = ambient + Lo;
	//vec3 color = Lo;
	vec3 color = Lo * ao;

	// gamma correction

	vec3 matColor = color;


	if (u_UseGrungeMap) {
		matColor *= max(texture(u_GrungeMap, u_GrungeMapTiling * vec2(u_UVRatio * v_TexCoords.x, v_TexCoords.y)).xyz, vec3(u_GrungeMapMin));
	}
    
	return matColor;

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

float linstepMax(float val, float maxVal) {
	return min(val / maxVal, 1.0);
}

vec4 linstepMaxVec4(vec4 val, float maxVal) {
	return min(val / maxVal, 1.0);
}


vec2 getTexCoords(int materialIdx) {
	return vec2(u_UVRatio * v_TexCoords.x * u_Materials[materialIdx].tiling, v_TexCoords.y * u_Materials[materialIdx].tiling);
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



vec3 computeNormalVec(vec4 materialContributions) {
	return
		materialContributions.r * texture(u_Materials[0].normalMap, getTexCoords(0)).rgb 
		+ materialContributions.g * texture(u_Materials[1].normalMap, getTexCoords(1)).rgb
		+ materialContributions.b * texture(u_Materials[2].normalMap, getTexCoords(2)).rgb
		+ materialContributions.a * texture(u_Materials[3].normalMap, getTexCoords(3)).rgb;
}


vec4 computeAlbedo(vec4 materialContributions) {
	return pow(
	//return (
		materialContributions.r * texture(u_Materials[0].albedo, getTexCoords(0)) 
		+ materialContributions.g * texture(u_Materials[1].albedo, getTexCoords(1))
		+ materialContributions.b * texture(u_Materials[2].albedo, getTexCoords(2))
		+ materialContributions.a * texture(u_Materials[3].albedo, getTexCoords(3))
		, vec4(2.2)
		);
}


vec2 computeMetallicRoughness(vec4 materialContributions) {
	return
		materialContributions.r * texture(u_Materials[0].metallicRoughness, getTexCoords(0)).rg 
		+ materialContributions.g * texture(u_Materials[1].metallicRoughness, getTexCoords(1)).rg
		+ materialContributions.b * texture(u_Materials[2].metallicRoughness, getTexCoords(2)).rg
		+ materialContributions.a * texture(u_Materials[3].metallicRoughness, getTexCoords(3)).rg;
}


float computeAmbientOcclusion(vec4 materialContributions) {
	return pow(
		materialContributions.r * texture(u_Materials[0].ao, getTexCoords(0)).r 
		+ materialContributions.g * texture(u_Materials[1].ao, getTexCoords(1)).r
		+ materialContributions.b * texture(u_Materials[2].ao, getTexCoords(2)).r
		+ materialContributions.a * texture(u_Materials[3].ao, getTexCoords(3)).r
		, 2.2);
}





