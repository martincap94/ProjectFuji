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
	sampler2D diffuse;
	sampler2D specular;
	sampler2D normalMap;
	float shininess;
	float tiling;

	//float ka;
	//float kd;
	//float ks;
};

uniform float u_ka;
uniform float u_kd;
uniform float u_ks;

uniform sampler2D u_TestDiffuse;

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


uniform bool u_NormalsOnly;
uniform int u_NormalsMode = 0;
uniform bool u_VisualizeTextureMode;


vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec4 materialContributions);
vec3 calcNightAmbientLight(vec4 materialContributions, vec3 ambientColor);

float calcShadowBasic(vec4 fragLightSpacePos);
float calcShadow(vec4 fragLightSpacePos);
float chebyshev(vec2 moments, float depth);
float reduceLightBleed(float p_max, float amount);
float linstep(float minVal, float maxVal, float val);
float linstepMax(float val, float maxVal);
vec4 linstepMaxVec4(vec4 val, float maxVal);

vec2 getTexCoords(int materialIdx);

void main() {


	if (u_VisualizeTextureMode) {
		fragColor = texture(u_MaterialMap, v_TexCoords);
		return;
	}

	vec4 materialMap = texture(u_MaterialMap, v_TexCoords);
	materialMap.a = 1.0 - materialMap.a;
	vec4 materialContributions = linstepMaxVec4(materialMap, dot(vec4(1.0), materialMap));


	vec3 norm = 
		materialContributions.r * texture(u_Materials[0].normalMap, getTexCoords(0)).rgb 
		+ materialContributions.g * texture(u_Materials[1].normalMap, getTexCoords(1)).rgb
		+ materialContributions.b * texture(u_Materials[2].normalMap, getTexCoords(2)).rgb
		+ materialContributions.a * texture(u_Materials[3].normalMap, getTexCoords(3)).rgb;

	if (u_NormalsMode == 0) {
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
	//vec3 viewDir = normalize(v_FragPos.xyz - u_ViewPos);

	
	//if (u_DirLight.direction.y > 0.0) {
	//	fragColor = vec4(calcNightAmbientLight(materialMap, vec3(0.2)), 1.0);
	//} else {

		float shadow = calcShadow(v_LightSpacePos);

		vec3 result;

		if (u_ShadowOnly) {
			result = vec3(shadow);
		} else {
			vec3 color = calcDirLight(u_DirLight, norm, viewDir, materialMap);
			result = color * min(shadow + 0.2, 1.0);
		}




		if (u_CloudsCastShadows) {

			vec3 projCoords = v_PrevLightSpacePos.xyz / v_PrevLightSpacePos.w;
			projCoords = projCoords * 0.5 + vec3(0.5);

			vec3 cloudShadow = vec3(1.0);
			vec4 cloudShadowVals =  texture(u_CloudShadowTexture, projCoords.xy);

			cloudShadow -= min(cloudShadowVals.xyz + cloudShadowVals.a * u_CloudCastShadowAlphaMultiplier, 0.8); // testing out ideas
			//cloudShadow -= cloudShadowVals.xyz; // simple

			//if (cloudShadowVals.a > 0.9) {
			//	cloudShadow -= min(cloudShadowVals.xyz, 0.8);
			//}

			result *= cloudShadow;

		}

		fragColor = vec4(result, 1.0);



	//}

	float dist = distance(v_FragPos.xyz, u_ViewPos);


	if (u_Fog.mode == 0) {
		float t = (dist - u_Fog.minDistance) / (u_Fog.maxDistance - u_Fog.minDistance);
		fragColor = mix(fragColor, u_Fog.color, clamp(t * u_Fog.intensity, 0.0, 1.0));
	} else if (u_Fog.mode == 1) {
		float t = exp(-u_Fog.expFalloff * dist * u_Fog.expFalloff * dist);
		fragColor = mix(fragColor, u_Fog.color, clamp((1.0 - t) * u_Fog.intensity, 0.0, 1.0));

	}
 	
	//fragColor = mix(u_Fog.color, fragColor, min(u_Fog.minDistance / distance, 1.0));

}


vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec4 materialContributions) {
    vec3 lightDir = normalize(-light.direction);
    
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    //float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_Materials[0].shininess);
    
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 
		u_Materials[0].shininess * materialContributions.r 
		+ u_Materials[1].shininess * materialContributions.g 
		+ u_Materials[2].shininess * materialContributions.b 
		+ u_Materials[3].shininess * materialContributions.a);



	//vec3 matColor = vec3(0.5, 0.2, 0.2);
	//vec3 matColor = mix(texture(u_Materials[0].diffuse, v_TexCoords * u_Materials[0].tiling).rgb, texture(u_TestDiffuse, v_TexCoords * u_Materials[0].tiling).rgb, degrees(acos(dot(vec3(0.0, 1.0, 0.0), normal))) / 90.0);

	//vec3 matColor = texture(u_Materials[0].diffuse, v_TexCoords * u_Materials[0].tiling).rgb *
	//vec3 matColor = materialMap.r * 

	vec3 matColor = 
	materialContributions.r * texture(u_Materials[0].diffuse, getTexCoords(0)).rgb 
	+ materialContributions.g * texture(u_Materials[1].diffuse, getTexCoords(1)).rgb
	+ materialContributions.b * texture(u_Materials[2].diffuse, getTexCoords(2)).rgb
	+ materialContributions.a * texture(u_Materials[3].diffuse, getTexCoords(3)).rgb;

	if (u_UseGrungeMap) {
		matColor *= max(texture(u_GrungeMap, u_GrungeMapTiling * vec2(u_UVRatio * v_TexCoords.x, v_TexCoords.y)).xyz, vec3(u_GrungeMapMin));
	}
    // combine results
	vec3 ambient = light.color * matColor * u_ka;
    vec3 diffuse  = light.color  * diff * matColor * u_kd;
    vec3 specular = light.color * spec * matColor * u_ks;

    
    return (ambient + diffuse + specular);
}

vec3 calcNightAmbientLight(vec4 materialContributions, vec3 ambientColor) {
	vec3 matColor = materialContributions.r * texture(u_Materials[0].diffuse, getTexCoords(0)).rgb + materialContributions.g * texture(u_Materials[1].diffuse, getTexCoords(1)).rgb;

	return matColor * ambientColor;

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

float linstepMax(float val, float maxVal) {
	return min(val / maxVal, 1.0);
}

vec4 linstepMaxVec4(vec4 val, float maxVal) {
	return min(val / maxVal, 1.0);
}


vec2 getTexCoords(int materialIdx) {
	return vec2(u_UVRatio * v_TexCoords.x * u_Materials[materialIdx].tiling, v_TexCoords.y * u_Materials[materialIdx].tiling);
}
