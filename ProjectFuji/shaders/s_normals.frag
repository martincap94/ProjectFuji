#version 330 core

in vec2 texCoord;
in vec3 normal;
in vec3 fragPos;
in mat3 TBN;
in vec4 fragPosLightSpace;

out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D shadowMap;

struct Material {
    sampler2D diffuse; // Cannot instantiate material!!! -> can now only be used as uniform
    sampler2D specular;
	sampler2D normalMap;
    float shininess;

	// This approach is supposed to be very slow!
	bool useDiffuse;
	bool useSpecular;
	bool useNormal;
};

uniform Material material;

struct Fog {
	vec4 color;
	float minDistance;
};

uniform Fog fog;

struct DirLight {
    vec3 direction;
    
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform DirLight dirLight;

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutoff;
    float outerCutoff;
    
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    
    float constant;
    float linear;
    float quadratic;
};

uniform SpotLight spotLight;

struct PointLight {
    vec3 position;
    
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    
    float constant;
    float linear;
    float quadratic;
};

#define NR_POINT_LIGHTS $$NR_POINT_LIGHTS$$
uniform PointLight pointLights[NR_POINT_LIGHTS];

uniform int numActivePointLights = 0;

uniform bool blinnEnabled;
uniform bool spotlightEnabled;
uniform bool ignoreSpecular;
uniform bool ignoreNormalMap;

vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir);
vec3 calcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
float calcShadow(vec4 fragPosLightSpace);

void main() {

	//// obtain normal from normal map in range [0,1]
 //   vec3 normal = texture(normalMap, fs_in.TexCoords).rgb;
 //   // transform normal vector to range [-1,1]
 //   normal = normalize(normal * 2.0 - 1.0);   

 //   //vec3 norm = normalize(normal);
	//vec3 norm = normal;

	vec3 norm = texture(material.normalMap, texCoord).rgb;
	norm = normalize(norm * 2.0 - 1.0);
	norm = normalize(TBN * norm); 

	//fragColor = vec4(norm, 1.0);

	//vec3 norm = normalize(normal);
    vec3 viewDir = normalize(viewPos - fragPos);

    vec3 result = calcDirLight(dirLight, norm, viewDir);
    
	int numPointLights = numActivePointLights;
	if (numPointLights > NR_POINT_LIGHTS) {
		numPointLights = NR_POINT_LIGHTS;
	}
    for (int i = 0; i < numPointLights; i++) {
        result += calcPointLight(pointLights[i], norm, fragPos, viewDir);
    }
    if (spotlightEnabled) {
		result += calcSpotLight(spotLight, norm, fragPos, viewDir);
	}

	float distance = length(fragPos.xyz - viewPos);

    
    fragColor = vec4(result, 1.0);
	fragColor = mix(fog.color, fragColor, min(fog.minDistance / distance, 1.0));
    
    
}

vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    float diff = max(dot(normal, lightDir), 0.0);
    
	float spec = 0.0;
	if (blinnEnabled) {
		vec3 halfwayDir = normalize(lightDir + viewDir);
		spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
	} else {
		vec3 reflectDir = reflect(-lightDir, normal);
		spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
	}
    
    float lightDistance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * lightDistance + light.quadratic * pow(lightDistance, 2));
    
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, texCoord));
	//vec3 diffuseTexColor;
	//if (material.useDiffuse) {
	//	diffuseTexColor = vec3(texture(material.diffuse, texCoord));
	//} else {
	//	diffuseTexColor = vec3(1.0, 0.05, 0.45);
	//}
    //vec3 diffuse  = light.diffuse  * diff * diffuseTexColor;
	vec3 diffuse  = light.diffuse  * diff * vec3(texture(material.diffuse, texCoord));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, texCoord));
    
    
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutoff - light.outerCutoff;
    float intensity = clamp((theta - light.outerCutoff) / epsilon, 0.0, 1.0);
    
    diffuse *= intensity;
    specular *= intensity;
    
	if (!ignoreSpecular) {
		return (ambient + diffuse + specular) * attenuation;
	} else {
		return (ambient + diffuse) * attenuation;
	}
}

vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // specular shading
    float spec = 0.0;
	if (blinnEnabled) {
		vec3 halfwayDir = normalize(lightDir + viewDir);
		spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
	} else {
		vec3 reflectDir = reflect(-lightDir, normal);
		spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
	}
    
    // combine results
    vec3 ambient  = light.ambient  * vec3(texture(material.diffuse, texCoord));
    vec3 diffuse  = light.diffuse  * diff * vec3(texture(material.diffuse, texCoord));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, texCoord));


	float shadow = calcShadow(fragPosLightSpace);
	vec3 ret = vec3(0.0, 0.0, 0.0);

	//if (shadow == 1.0) {
	//	return vec3(1.0, 0.05, 0.42);
	//}
    if (!ignoreSpecular) {
		ret = (ambient + (1.0 - shadow) * (diffuse + specular));
	} else {
		ret = (ambient + (1.0 - shadow) *  diffuse);
	}
	return ret;
}

vec3 calcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // specular shading

	float spec = 0.0;
	if (blinnEnabled) {
		vec3 halfwayDir = normalize(lightDir + viewDir);
		spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
	} else {
		vec3 reflectDir = reflect(-lightDir, normal);
		spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
	}
    
    
    // attenuation
    float distance    = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance +
                               light.quadratic * (distance * distance));
    // combine results
    vec3 ambient  = light.ambient  * vec3(texture(material.diffuse, texCoord));
    vec3 diffuse  = light.diffuse  * diff * vec3(texture(material.diffuse, texCoord));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, texCoord));
    ambient  *= attenuation;
    diffuse  *= attenuation;
    specular *= attenuation;
    
    if (!ignoreSpecular) {
		return (ambient + diffuse + specular);
	} else {
		return (ambient + diffuse);
	}
}

float calcShadow(vec4 fragPosLightSpace) {
	
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	projCoords = projCoords * 0.5 + 0.5;
	float currentDepth = projCoords.z;
	if (currentDepth > 1.0) {
		return 0.0;
	}
	float closestDepth = texture(shadowMap, projCoords.xy).r;
	float bias = 0.0005;
	//float bias = max(0.05 * (1.0 - dot(normal, dirLight.direction)), 0.005);  
	//float shadow = (currentDepth - bias) > closestDepth ? 1.0 : 0.0;
	float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += (currentDepth - bias) > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
	return shadow;
}
























