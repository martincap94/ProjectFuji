#pragma once

#include <glad\glad.h>

#include "ShaderProgram.h"

#include "ArHosekSkyModel.h"
//#include "DirectionalLight.h"

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

	HosekSkyModel();
	~HosekSkyModel();

	void draw();
	void initBuffers();

	//void update(DirectionalLight *sun);
	void update(glm::vec3 sunDir);

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

