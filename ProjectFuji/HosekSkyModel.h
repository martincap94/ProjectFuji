///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       HosekSkyModel.h
* \author     Martin Cap
*
*	Describes HosekSkyModel class that is used to generate and feed atmosphere visualization
*	on GPU as well as CPU using Hosek-Wilkie's sky model with the provided data.
*	Hosek-Wilkie's model is desribed here: https://cgg.mff.cuni.cz/projects/SkylightModelling/ 
*	This is a C++ reimplementation of Ben Anderson's Rust implementation of the sky model that is
*	available here: https://github.com/benanders/Hosek-Wilkie
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>

#include "ShaderProgram.h"

#include "ArHosekSkyModel.h"
#include "DirectionalLight.h"

// BASED ON: https://github.com/benanders/Hosek-Wilkie BY BEN ANDERSON

class HosekSkyModel {
public:

	double turbidity = 4.0;
	double albedo = 0.5;

	int liveRecalc = 1;

	int calcParamMode = 0;
	int useAndersonsRGBNormalization = 1;

	float elevation;
	double eta = 0.0; // angle "below" sun (between sun and xz plane)
	double sunTheta = 0.0; // angle "above" sun (between sun and y plane)
	

	// shader uniforms
	double horizonOffset = 0.01;
	float sunIntensity = 2.5f;
	int sunExponent = 512;

	HosekSkyModel(DirectionalLight *dirLight);
	~HosekSkyModel();

	void draw(const glm::mat4 &viewMatrix);
	void initBuffers();

	void update();

	glm::vec3 getColor(float cosTheta, float gamma, float cosGamma);
	glm::vec3 getSunColor();

	float getElevationDegrees();

	// UI helpers
	std::string getCalcParamModeName();
	std::string getCalcParamModeName(int mode);


private:

	glm::vec3 params[10];

	ArHosekSkyModelState *skymodel_state;

	double prevEta = 0.0;
	double prevTurbidity = turbidity;
	double prevAlbedo = albedo;
	int prevCalcParamMode = calcParamMode;
	int prevUseAndersonsRGBNormalization = useAndersonsRGBNormalization;

	double telev = 0.0;

	ShaderProgram *shader = nullptr;

	DirectionalLight *dirLight = nullptr;

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;

	bool shouldUpdate(float newEta);


	void recalculateParams(glm::vec3 sunDir);

	// stride only different for radiosity dataset, otherwise always 9
	double calculateParam(double *dataset, int stride);
	double calculateBezier(double *dataset, int start, int stride);

	void normalizeRGBParams(glm::vec3 sunDir);

	// uses implementation provided by Hosek
	void recalculateParamsHosek(glm::vec3 sunDir);


};

