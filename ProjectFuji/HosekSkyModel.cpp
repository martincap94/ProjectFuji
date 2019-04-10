#include "HosekSkyModel.h"

#include <iostream>
#include "DataStructures.h"

#include "ArHosekSkyModel.c"
#include "ShaderManager.h"

//#include "ArHosekSkyModelData_RGB.h"



using namespace std;

HosekSkyModel::HosekSkyModel() {
	initBuffers();
	shader = ShaderManager::getShaderPtr("sky_hosek");
}


HosekSkyModel::~HosekSkyModel() {
}

void HosekSkyModel::draw() {
	if (!shader) {
		cerr << "No shader set in " << __FILE__ << " on " << __LINE__ << endl;
		return;
	}
	shader->use();

	shader->setFloat("u_HorizonOffset", (float)horizonOffset);
	shader->setFloat("u_SunIntensity", sunIntensity * 0.1f);
	shader->setInt("u_SunExponent", sunExponent);

	glDepthMask(GL_FALSE);

	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

	glDepthMask(GL_TRUE);


}

void HosekSkyModel::initBuffers() {

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVerticesNew), skyboxVerticesNew, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyboxIndicesNew), skyboxIndicesNew, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

}

void HosekSkyModel::update(glm::vec3 sunDir) {

	shader->use();

	elevation = -sunDir.y;
	eta = asinf(-sunDir.y);
	sunTheta = acosf(-sunDir.y);
	//cout << "ETA: " << eta << endl;
	//cout << "   degrees: " << glm::degrees(eta) << endl;
	//double elev = (double)glm::radians(eta); // doesn't make sense to use radians again (since eta is already in radians)

	/*
	double testElev = pow(eta / (MATH_PI / 2.0), (1.0 / 3.0));
	cout << "testElev  = " << testElev << endl;
	testElev = pow(1.0 - sunTheta / (MATH_PI / 2.0), (1.0 / 3.0));
	cout << "testElev2 = " << testElev << endl;
	*/


	if (!shouldUpdate(eta)) {
		return;
	}


	if (calcParamMode == 0) {
		recalculateParams(sunDir);
	} else {
		recalculateParamsHosek(sunDir);
	}


	glUniform3fv(glGetUniformLocation(shader->id, "u_Params"), 10, glm::value_ptr(params[0]));

	prevEta = eta;
	prevTurbidity = turbidity;
	prevAlbedo = albedo;
	prevCalcParamMode = calcParamMode;
	prevUseAndersonsRGBNormalization = useAndersonsRGBNormalization;

}

float HosekSkyModel::getElevationDegrees() {
	return glm::degrees(eta);
}

std::string HosekSkyModel::getCalcParamModeName() {
	return getCalcParamModeName(calcParamMode);
}

std::string HosekSkyModel::getCalcParamModeName(int mode) {
	if (mode == 0) {
		return "Ben Anderson impl.";
	} else {
		return "Hosek & Wilkie impl.";
	}
}

bool HosekSkyModel::shouldUpdate(float newEta) {
	return (newEta != prevEta 
			|| turbidity != prevTurbidity 
			|| albedo != prevAlbedo 
			|| calcParamMode != prevCalcParamMode
			|| prevUseAndersonsRGBNormalization != useAndersonsRGBNormalization);
}

glm::vec3 HosekSkyModel::getColor(float cosTheta, float gamma, float cosGamma) {

	auto A = params[0];
	auto B = params[1];
	auto C = params[2];
	auto D = params[3];
	auto E = params[4];
	auto F = params[5];
	auto G = params[6];
	auto H = params[7];
	auto I = params[8];

	glm::vec3 chi = (1 + cosGamma * cosGamma) / glm::pow(glm::vec3(1.0f) + H * H - 2.0f * cosGamma * H, glm::vec3(1.5));
	return (glm::vec3(1.0f) + A * glm::exp(B / (cosTheta + glm::vec3(horizonOffset)))) * (C + D * glm::exp(E * gamma) + F * (cosGamma * cosGamma) + G * chi + I * sqrt(cosTheta));
}

glm::vec3 HosekSkyModel::getSunColor() {
	glm::vec3 sunColor = params[9] * getColor(elevation, 0.0f, 1.0f);
	//sunColor += sunIntensity; // simplified calculation from fragment shader since cosGamma == 1.0 (pow(cosGamma, u_SunExponent) == 1.0)
	
	// testing: get multiple samples samples
	//glm::vec3 sunColor = glm::vec3(0.0f);
	//sunColor += getColor(elevation + glm::radians(1.0f), 0.0f, 1.0f);
	//sunColor += getColor(elevation + glm::radians(0.5f), 0.0f, 1.0f);
	//sunColor += getColor(elevation, 0.0, 1.0f);
	//sunColor += getColor(elevation, 1.0f, cosf(1.0f));
	//sunColor *= params[9];
	//sunColor /= 4;

	return sunColor;
}

void HosekSkyModel::recalculateParams(glm::vec3 sunDir) {


	telev = pow(eta / (MATH_PI / 2.0), (1.0 / 3.0));
	//cout << "testElev  = " << telev << endl;
	//telev = pow(1.0 - sunTheta / (MATH_PI / 2.0), (1.0 / 3.0));
	//cout << "testElev2 = " << telev << endl;

	for (int i = 0; i < 3; i++) {
		params[0][i] = calculateParam(&datasetsRGB[i][0], 9);
		params[1][i] = calculateParam(&datasetsRGB[i][1], 9);
		params[2][i] = calculateParam(&datasetsRGB[i][2], 9);
		params[3][i] = calculateParam(&datasetsRGB[i][3], 9);
		params[4][i] = calculateParam(&datasetsRGB[i][4], 9);
		params[5][i] = calculateParam(&datasetsRGB[i][5], 9);
		params[6][i] = calculateParam(&datasetsRGB[i][6], 9);

		// as in https://github.com/benanders/Hosek-Wilkie/blob/master/src/main.rs
		//params[7][i] = calculateParam(&datasetsRGB[i][7], 9);
		params[8][i] = calculateParam(&datasetsRGB[i][7], 9);
		params[7][i] = calculateParam(&datasetsRGB[i][8], 9);

		params[9][i] = calculateParam(datasetsRGBRad[i], 1);
	}

	normalizeRGBParams(sunDir);

}

double HosekSkyModel::calculateParam(double *dataset, int stride) {

	int turbidity_low = glm::clamp((int)turbidity, 1, 10);
	int turbidity_high = glm::min(turbidity_low + 1, 10);
	double turbidity_rem = turbidity - (double)turbidity_low;

	int albedo0_offset = 0;
	int albedo1_offset = stride * 6 * 10;

	double albedo0_turbidity_low = calculateBezier(dataset, albedo0_offset + stride * 6 * (turbidity_low - 1), stride);
	double albedo1_turbidity_low = calculateBezier(dataset, albedo1_offset + stride * 6 * (turbidity_low - 1), stride);
	double albedo0_turbidity_high = calculateBezier(dataset, albedo0_offset + stride * 6 * (turbidity_high - 1), stride);
	double albedo1_turbidity_high = calculateBezier(dataset, albedo1_offset + stride * 6 * (turbidity_high - 1), stride);

	double res;
	res = albedo0_turbidity_low * (1.0 - albedo) * (1.0 - turbidity_rem);
	res += albedo1_turbidity_low * albedo * (1.0 - turbidity_rem);
	res += albedo0_turbidity_high * (1.0 - albedo) * turbidity_rem;
	res += albedo1_turbidity_high * albedo * turbidity_rem;

	return res;

}

double HosekSkyModel::calculateBezier(double * dataset, int start, int stride) {
	return
		1.0 * pow(1.0 - telev, 5.0) * dataset[start + 0 * stride] +
		5.0 * pow(1.0 - telev, 4.0) * dataset[start + 1 * stride] * telev +
		10.0 * pow(1.0 - telev, 3.0) * dataset[start + 2 * stride] * pow(telev, 2.0) +
		10.0 * pow(1.0 - telev, 2.0) * dataset[start + 3 * stride] * pow(telev, 3.0) +
		5.0 * (1.0 - telev) * dataset[start + 4 * stride] * pow(telev, 4.0) +
		1.0 * dataset[start + 5 * stride] * pow(telev, 5.0);
}

void HosekSkyModel::normalizeRGBParams(glm::vec3 sunDir) {
	if (useAndersonsRGBNormalization) {
		glm::vec3 S = getColor(-sunDir.y, 0.0f, 1.0f) * params[9];
		params[9] /= glm::dot(S, glm::vec3(0.2126f, 0.7152f, 0.0722f));

		float sunAmount = fmodf(((-sunDir.y) / (MATH_PI / 2.0f)), 4.0f);
		if (sunAmount > 2.0f) {
			sunAmount = 0.0f;
		}
		if (sunAmount > 1.0f) {
			sunAmount = 2.0f - sunAmount;
		} else if (sunAmount < -1.0f) {
			sunAmount = -2.0f - sunAmount;
		}
		float normalizedSunY = 0.6f + 0.45f * sunAmount;
		params[9] *= normalizedSunY;
	}
}

void HosekSkyModel::recalculateParamsHosek(glm::vec3 sunDir) {

	double elev = eta;

	skymodel_state = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo, glm::radians(elev));

	glm::vec3 tmp(0.0f);
	for (int j = 0; j < 9; j++) {
		for (int i = 0; i < 3; i++) {
			tmp[i] = skymodel_state->configs[i][j];
		}
		params[j] = tmp;
	}
	for (int i = 0; i < 3; i++) {
		tmp[i] = skymodel_state->radiances[i];
	}
	params[9] = tmp;

	normalizeRGBParams(sunDir);


	/*
	glm::vec3 test;
	double thetaTest = glm::radians(50.0);
	double gammaTest = glm::radians(30.0);
	test.x = arhosek_tristim_skymodel_radiance(skymodel_state, glm::cos(thetaTest), glm::cos(gammaTest), 0);
	test.y = arhosek_tristim_skymodel_radiance(skymodel_state, thetaTest, gammaTest, 1);
	test.z = arhosek_tristim_skymodel_radiance(skymodel_state, thetaTest, gammaTest, 2);
	//test *= glm::vec3(0.2126f, 0.7152f, 0.0722f);
	cout << "TEST:   " << test.x << ", " << test.y << ", " << test.z << endl;
	test = getColor(glm::cos(thetaTest), gammaTest, glm::cos(gammaTest));
	cout << "my own: " << test.x << ", " << test.y << ", " << test.z << endl;
	*/


	arhosekskymodelstate_free(skymodel_state);
}
