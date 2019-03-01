#version 400 core

layout (location = 0) out vec4 FragColor;

in vec3 v_Normal;
in vec2 v_TexCoord;
in vec4 v_Vertex;
in vec4 v_LightSpacePos;

uniform int       u_UserVariableInt;
uniform float     u_UserVariableFloat;

uniform int		  u_PCFMode; // 0 - no PCF, 1 - 2x2 HW, 2 - 3x3 basic, 3 - 3x3 gaussian, 4 - 9x9 gaussian

uniform int       u_ShadowOnly;
uniform float	  u_ShadowBias;

uniform vec4      u_LightPosition;
uniform sampler2D u_DepthMapTexture;
uniform sampler2DShadow u_ZBufferTexture;


const float pcfKernel9x9[81] = float[](
0.000814,	0.001918,	0.003538,	0.005108,	0.005774,	0.005108,	0.003538,	0.001918,	0.000814,
0.001918,	0.00452,	0.008338,	0.012038,	0.013605,	0.012038,	0.008338,	0.00452,	0.001918,
0.003538,	0.008338,	0.015378,	0.022203,	0.025094,	0.022203,	0.015378,	0.008338,	0.003538,
0.005108,	0.012038,	0.022203,	0.032057,	0.036231,	0.032057,	0.022203,	0.012038,	0.005108,
0.005774,	0.013605,	0.025094,	0.036231,	0.04095,	0.036231,	0.025094,	0.013605,	0.005774,
0.005108,	0.012038,	0.022203,	0.032057,	0.036231,	0.032057,	0.022203,	0.012038,	0.005108,
0.003538,	0.008338,	0.015378,	0.022203,	0.025094,	0.022203,	0.015378,	0.008338,	0.003538,
0.001918,	0.00452,	0.008338,	0.012038,	0.013605,	0.012038,	0.008338,	0.00452,	0.001918,
0.000814,	0.001918,	0.003538,	0.005108,	0.005774,	0.005108,	0.003538,	0.001918,	0.000814
);

const float pcfKernel3x3[9] = float[](
0.102059,	0.115349,	0.102059,
0.115349,	0.130371,	0.115349,
0.102059,	0.115349,	0.102059
);

float calcShadow(vec4 fragLightSpacePos);

void main() {
// Compute fragment diffuse color
    //vec3 N = normalize(v_Normal);
    //vec3 L = normalize(u_LightPosition.xyz - v_Vertex.xyz);
    //float NdotL = max(dot(N, L), 0.0);
    //vec4 color = texture(u_SceneTexture, v_TexCoord) * NdotL;

	vec4 color = vec4(1.0);

	float res = calcShadow(v_LightSpacePos);
	vec4 shadow = vec4(res);

	if (u_ShadowOnly == 1) {
		FragColor = shadow;
	} else {
		FragColor = color*shadow;
	}
}



float calcShadow(vec4 fragLightSpacePos) {

	switch(u_PCFMode) {
		case 0: {
			// does not use projected coordinates!
			vec2 texCoords = -fragLightSpacePos.xy / fragLightSpacePos.z * 0.5 + vec2(0.5);
			float depth = texture(u_DepthMapTexture, texCoords).r + u_ShadowBias * 10.0; // make bias larger due to different approach
			float distance = length(fragLightSpacePos.xyz);
			return (depth <= distance) ? 0.0 : 1.0;
		}
		//case 1:
		//case 2:
		//case 3:
		//case 4:
		default:
			vec3 projCoords = fragLightSpacePos.xyz / fragLightSpacePos.w;
			projCoords.z -= u_ShadowBias;
			projCoords = projCoords * 0.5 + vec3(0.5); // use projected coordinates in [0, 1]
			float currentDepth = projCoords.z;
			if (currentDepth >= 1.0) {
				return 0.0;
			}

			if (u_PCFMode == 1) {
				float visibility = texture(u_ZBufferTexture, projCoords);
				return visibility;
			} else {
				float shadow = 0.0;
				//vec2 texelSize = 1.0 / textureSize(u_DepthMapTexture, 0);
				if (u_PCFMode == 2) {
					for (int x = -1; x <= 1; x++) {
						for (int y = -1; y <= 1; y++) {
							//float texelDepth = texture(u_DepthMapTexture, projCoords.xy + vec2(x, y) * texelSize).b;
							float texelDepth = textureOffset(u_DepthMapTexture, projCoords.xy, ivec2(x, y)).b;
							shadow += (currentDepth > texelDepth) ? 0.0 : 1.0;
 						}
					}
					shadow /= 9.0;
				} else if (u_PCFMode == 3) {
					for (int x = -1; x <= 1; x++) {
						for (int y = -1; y <= 1; y++) {
							//float texelDepth = texture(u_DepthMapTexture, projCoords.xy + vec2(x, y) * texelSize).b;
							float texelDepth = textureOffset(u_DepthMapTexture, projCoords.xy, ivec2(x, y)).b;
							shadow += (currentDepth > texelDepth) ? 0.0 : pcfKernel3x3[x + 1 + (y + 1) * 3];
 						}
					}
				} else {
					for (int x = -4; x <= 4; x++) {
						for (int y = -4; y <= 4; y++) {
							//float texelDepth = texture(u_DepthMapTexture, projCoords.xy + vec2(x, y) * texelSize).b;
							float texelDepth = textureOffset(u_DepthMapTexture, projCoords.xy, ivec2(x, y)).b;
							shadow += (currentDepth > texelDepth) ? 0.0 : pcfKernel9x9[x + 4 + (y + 4) * 9];
 						}
					}
				}
				return shadow;
			}
	}
}