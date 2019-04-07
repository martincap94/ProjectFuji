#version 330 core

in vec3 v_FragPos;

out vec4 fragColor;

//uniform samplerCube skybox;


// BASED ON: https://github.com/benanders/Hosek-Wilkie/blob/master/src/shaders/frag.glsl

//uniform float u_Params[30];
uniform vec3 u_Params[10];
uniform vec3 u_SunDir;

uniform float u_SunIntensity;
uniform int u_SunExponent;

uniform float u_HorizonOffset = 0.01;

vec3 hosekWilkie(float cosTheta, float gamma, float cosGamma);

void main() {

	vec3 viewVec = normalize(v_FragPos); // view into the sky from (0, 0, 0)

	//fragColor = vec4(1.0);
	//fragColor = vec4(vec3(u_Params[0].y), 1.0);
	//return;
	vec3 sunDir = normalize(u_SunDir);
	float cosTheta = viewVec.y;

	float cosGamma = dot(viewVec, sunDir);
	float gamma = acos(cosGamma);

	vec3 Z = u_Params[9];
	vec3 R = Z * hosekWilkie(cosTheta, gamma, cosGamma);

	if (cosGamma > 0 /*&& cosGamma >= (1.0 - u_SunIntensity)*/) {
		// Only positive values of dot product, so we don't end up creating two
		// spots of light 180 degrees apart
		R = R + vec3(pow(cosGamma, u_SunExponent) * u_SunIntensity);
		//R = vec3(1.0);
	}

	fragColor = vec4(R, 1.0);
    
}




vec3 hosekWilkie(float cosTheta, float gamma, float cosGamma) {
	vec3 A = u_Params[0];
	vec3 B = u_Params[1];
	vec3 C = u_Params[2];
	vec3 D = u_Params[3];
	vec3 E = u_Params[4];
	vec3 F = u_Params[5];
	vec3 G = u_Params[6];
	vec3 H = u_Params[7];
	vec3 I = u_Params[8];
	//vec3 Z = u_Params[9];
	vec3 chi = (1.0 + cosGamma * cosGamma) / pow(1.0 + H * H - 2.0 * cosGamma * H, vec3(1.5));
    return (1.0 + A * exp(B / (cosTheta + u_HorizonOffset))) * (C + D * exp(E * gamma) + F * (cosGamma * cosGamma) + G * chi + I * sqrt(cosTheta));
}


/*
vec3 hosekWilkie(float cosTheta, float gamma, float cosGamma) {
	vec3 A = vec3(u_Params[0], u_Params[10], u_Params[20]);
	vec3 B = vec3(u_Params[1], u_Params[11], u_Params[21]);
	vec3 C = vec3(u_Params[2], u_Params[12], u_Params[22]);
	vec3 D = vec3(u_Params[3], u_Params[13], u_Params[23]);
	vec3 E = vec3(u_Params[4], u_Params[14], u_Params[24]);
	vec3 F = vec3(u_Params[5], u_Params[15], u_Params[25]);
	vec3 G = vec3(u_Params[6], u_Params[16], u_Params[26]);
	vec3 H = vec3(u_Params[7], u_Params[17], u_Params[27]);
	vec3 I = vec3(u_Params[8], u_Params[18], u_Params[28]);
	vec3 Z = vec3(u_Params[9], u_Params[19], u_Params[29]);
	vec3 chi = (1 + cosGamma * cosGamma) / pow(1 + H * H - 2 * cosGamma * H, vec3(1.5));
    return (1 + A * exp(B / (cosTheta + 0.01))) * (C + D * exp(E * gamma) + F * (cosGamma * cosGamma) + G * chi + I * sqrt(cosTheta));
}


*/



















