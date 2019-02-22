#version 330 core

out vec4 fragColor;

in vec3 vFragPos;
in vec3 vNormal;

struct DirLight {
	vec3 direction;

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

uniform DirLight dirLight;

uniform vec3 vViewPos;

vec3 calcDirLight(DirLight light, vec3 normal, vec3 viewDir);



void main() {

	vec3 norm = normalize(vNormal);
	vec3 viewDir = normalize(vViewPos - vFragPos);
	
	vec3 result = calcDirLight(dirLight, norm, viewDir);

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
    vec3 diffuse  = light.diffuse  * diff * matColor;
    vec3 specular = light.specular * spec * matColor;

    
    return (ambient + diffuse + specular);
}
