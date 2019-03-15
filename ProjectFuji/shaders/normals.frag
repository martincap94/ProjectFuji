#version 400 core

in vec2 v_TexCoords;
in vec3 v_FragPos;
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

uniform Material u_Material;

struct Fog {
	float intensity;
	float minDistance;
	float maxDistance;
	vec4 color; // alpha could be used instead of intensity
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


void main() {
	vec4 texColor = texture(u_Material.diffuse, v_TexCoords);
	if (texColor.a < 0.9) {
		discard;
	}
	

	vec3 norm = texture(u_Material.normalMap, v_TexCoords).rgb;
	norm = normalize(norm * 2.0 - 1.0);
	norm = normalize(v_TBN * norm); 

    vec3 viewDir = normalize(u_ViewPos - v_FragPos);

    //vec3 result = calcDirLight(u_DirLight, norm, viewDir);

	float shadow = calcShadow(v_LightSpacePos); // directional light only
	vec3 result;
	if (u_ShadowOnly) {
		result = vec3(shadow);
	} else {
		result = calcDirLight(u_DirLight, norm, viewDir) * min(shadow + 0.2, 1.0);
    }
    
	/*
	int numPointLights = u_NumActivePointLights;
	if (numPointLights > NR_POINT_LIGHTS) {
		numPointLights = NR_POINT_LIGHTS;
	}
    for (int i = 0; i < numPointLights; i++) {
        result += calcPointLight(u_PointLights[i], norm, v_FragPos, viewDir);
    }
    if (u_SpotLightEnabled) {
		result += calcSpotLight(u_SpotLight, norm, v_FragPos, viewDir);
	}
	*/
	
	

	//fragColor = vec4(1.0, 0.0, 0.0, 1.0);

    
    fragColor = vec4(result, 1.0);

	float distance = length(v_FragPos.xyz - u_ViewPos);
	float t = (distance - u_Fog.minDistance) / (u_Fog.maxDistance - u_Fog.minDistance);
	fragColor = mix(fragColor, u_Fog.color, min(t, 1.0) * u_Fog.intensity);

    
    
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



































