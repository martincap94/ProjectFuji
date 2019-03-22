#pragma once

#include <glad\glad.h>

#include "ShaderProgram.h"

#include "ArHosekSkyModel.h"
//#include "DirectionalLight.h"


class HosekSkyModel {
public:

	float turbidity = 4.0f;
	float albedo = 0.5f;
	//float sunElevation = 4.0f;

	int liveRecalc = 1;

	float horizonOffset = 0.01f;
	float eta = 0.0f;
	

	HosekSkyModel();
	~HosekSkyModel();

	void draw();
	void initBuffers();

	//void update(DirectionalLight *sun);
	void update(glm::vec3 sunDir);


	float getElevationDegrees();

private:

	ArHosekSkyModelState *skymodel_state;

	float prevEta = 0.0f;
	float prevTurbidity = turbidity;
	float prevAlbedo = albedo;

	ShaderProgram *shader = nullptr;

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;

	bool shouldUpdate(float newEta);

};

