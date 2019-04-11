#version 400 core

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec2 v_TexCoords;
in mat3 v_TBN;
in vec4 v_LightSpacePos;

out vec4 fragColor;

uniform vec3 u_ViewPos;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
	sampler2D normalMap;
    float shininess;
};

// quick testing
uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;





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



#define NR_POINT_LIGHTS 8
uniform PointLight u_PointLights[NR_POINT_LIGHTS];

uniform int u_NumActivePointLights = 0;

uniform bool u_SpotLightEnabled;

vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir);
vec3 calcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

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

	/*
	vec3 N = texture(u_Material.normalMap, v_TexCoords).rgb;
	N = normalize(N * 2.0 - 1.0);
	N = normalize(v_TBN * N); 
	*/
	vec3 N = normalize(v_Normal);

    vec3 V = normalize(u_ViewPos - v_WorldPos);

	//float shadow = calcShadow(v_LightSpacePos); // directional light only
	
	vec3 result;

	// PBR
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);

	vec3 L = normalize(-u_DirLight.direction);
	vec3 H = normalize(L + V);
	
	vec3 radiance = u_DirLight.color; // no attenuation for directional light

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

	vec3 ambient = vec3(0.03) * albedo * ao;
	vec3 color = ambient + Lo;

	// gamma correction
	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0 / 2.2));

	result = color;

	//result = N;



    
    fragColor = vec4(result, 1.0);

	/*
	float dist = distance(v_WorldPos.xyz, u_ViewPos);

	if (u_Fog.mode == 0) {
		float t = (dist - u_Fog.minDistance) / (u_Fog.maxDistance - u_Fog.minDistance);
		fragColor = mix(fragColor, u_Fog.color, clamp(t * u_Fog.intensity, 0.0, 1.0));
	}
	*/
	
    
}

vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    float diff = max(dot(normal, lightDir), 0.0);
    
	float spec = 0.0;
	vec3 halfwayDir = normalize(lightDir + viewDir);
	spec = pow(max(dot(normal, halfwayDir), 0.0), u_Material.shininess);

    
    float lightDistance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * lightDistance + light.quadratic * pow(lightDistance, 2));
    
    vec3 ambient = light.color * vec3(texture(u_Material.diffuse, v_TexCoords));

	vec3 diffuse  = light.color  * diff * vec3(texture(u_Material.diffuse, v_TexCoords));
    vec3 specular = light.color * spec * vec3(texture(u_Material.specular, v_TexCoords));
    
    
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutoff - light.outerCutoff;
    float intensity = clamp((theta - light.outerCutoff) / epsilon, 0.0, 1.0);
    
    diffuse *= intensity;
    specular *= intensity;
    
	return (ambient + diffuse + specular) * attenuation;
	
}

vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // specular shading
    float spec = 0.0;
	vec3 halfwayDir = normalize(lightDir + viewDir);
	spec = pow(max(dot(normal, halfwayDir), 0.0), u_Material.shininess);

    
    // combine results
    vec3 ambient  = 0.4 * vec3(texture(u_Material.diffuse, v_TexCoords));
    vec3 diffuse  = light.color  * diff * vec3(texture(u_Material.diffuse, v_TexCoords));
    vec3 specular = light.color * spec * vec3(texture(u_Material.specular, v_TexCoords));

	vec3 ret = vec3(0.0, 0.0, 0.0);

	ret = (ambient + diffuse + specular);
	
	return ret;
}

vec3 calcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // specular shading

	float spec = 0.0;
	vec3 halfwayDir = normalize(lightDir + viewDir);
	spec = pow(max(dot(normal, halfwayDir), 0.0), u_Material.shininess);
	
    
    
    // attenuation
    float distance    = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance +
                               light.quadratic * (distance * distance));
    // combine results
    vec3 ambient  = light.color  * vec3(texture(u_Material.diffuse, v_TexCoords));
    vec3 diffuse  = light.color  * diff * vec3(texture(u_Material.diffuse, v_TexCoords));
    vec3 specular = light.color * spec * vec3(texture(u_Material.specular, v_TexCoords));
    ambient  *= attenuation;
    diffuse  *= attenuation;
    specular *= attenuation;
    
	return (ambient + diffuse + specular);
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
































