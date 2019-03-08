#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

out vec2 texCoord;
out vec3 normal;
out vec3 fragPos;
out mat3 TBN;
out vec4 fragPosLightSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

void main() {
	vec3 T = normalize(vec3(model * vec4(aTangent, 0.0)));
    vec3 B = normalize(vec3(model * vec4(aBitangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(aNormal, 0.0)));
    TBN = mat3(T, B, N);
	
    texCoord = aTexCoord;
    normal = mat3(transpose(inverse(model))) * aNormal;
    fragPos = vec3(model * vec4(aPos, 1.0));
	fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
	gl_Position = projection * view * model * vec4(aPos, 1.0);

}