#pragma once

#include <glad\glad.h>

#include "ShaderProgram.h"

#include "ArHosekSkyModel.h"
//#include "DirectionalLight.h"


class HosekSkyModel {
public:

	double turbidity = 4.0;
	double albedo = 0.5;
	//float sunElevation = 4.0f;

	int liveRecalc = 1;

	double horizonOffset = 0.01;
	double eta = 0.0; // angle "below" sun (between sun and xz plane)
	double sunTheta = 0.0; // angle "above" sun (between sun and y plane)
	

	HosekSkyModel();
	~HosekSkyModel();

	void draw();
	void initBuffers();

	//void update(DirectionalLight *sun);
	void update(glm::vec3 sunDir);


	float getElevationDegrees();

private:

	glm::vec3 params[10];

	ArHosekSkyModelState *skymodel_state;

	double prevEta = 0.0;
	double prevTurbidity = turbidity;
	double prevAlbedo = albedo;

	double elevation = 0.0;

	ShaderProgram *shader = nullptr;

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;

	bool shouldUpdate(float newEta);

	glm::vec3 getColor(float cosTheta, float gamma, float cosGamma);

	void recalculateParams(glm::vec3 sunDir);

	// stride only different for radiosity dataset, otherwise always 9
	double calculateParam(double *dataset, int stride);
	double calculateBezier(double *dataset, int start, int stride);

	// uses implementation provided by Hosek
	void recalculateParamsHosek();


};

