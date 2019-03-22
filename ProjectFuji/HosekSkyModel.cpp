#include "HosekSkyModel.h"

#include <iostream>
#include "DataStructures.h"

#include "ArHosekSkyModel.c"
#include "ShaderManager.h"


using namespace std;

HosekSkyModel::HosekSkyModel() {
	initBuffers();
	shader = ShaderManager::getShaderPtr("sky_hosek");
	shader->use();

	vector<glm::vec3> params;

	//double elev = pow(4.0 / (MATH_PI / 2.0), 1.0 / 3.0);
	double elev = 0.1;

	skymodel_state = arhosek_rgb_skymodelstate_alloc_init(2.0, 0.9, glm::radians(elev));

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 9; j++) {
			cout << skymodel_state->configs[i][j] << ", ";
		}
		cout << "radiance: " << skymodel_state->radiances[i];
		cout << endl;
	}

	glm::vec3 tmp(0.0f);
	for (int j = 0; j < 9; j++) {
		for (int i = 0; i < 3; i++) {
			tmp[i] = skymodel_state->configs[i][j];
		}
		params.push_back(tmp);
	}
	for (int i = 0; i < 3; i++) {
		tmp[i] = skymodel_state->radiances[i];
	}
	params.push_back(tmp);

	glUniform3fv(glGetUniformLocation(shader->id, "u_Params"), 10, glm::value_ptr(params[0]));

	for (int i = 0; i < 3; i++) {
		cout << arhosek_tristim_skymodel_radiance(skymodel_state, 40.0, 25.0, i) << endl;
	}

	arhosekskymodelstate_free(skymodel_state);

	//ArHosekSkyModelState *skymodel_state[3];
	//double albedo[3];

	//for (int i = 0; i < 3; i++) {
	//	//skymodel_state[i] = arhosekskymodelstate_alloc_init(1.0, 0.1, 0.1);

	//	skymodel_state[i] = arhosek_rgb_skymodelstate_alloc_init(4.0, 0.1, 4.0);
	//}
	//for (int i = 0; i < 11; i++) {
	//	for (int j = 0; j < 9; j++) {
	//		cout << skymodel_state[0]->configs[i][j] << ", ";
	//	}
	//	cout << "radiance: " << skymodel_state[0]->radiances[i];
	//	cout << endl;
	//}



	//double center_wavelengths[3] = { 640.0, 532.0, 473.0 };
	////double center_wavelengths[3] = { 575.0, 535.0, 445.0 };

	//double skydome_result[3];
	//for (int i = 0; i < 3; i++) {
	//	skydome_result[i] = arhosekskymodel_radiance(skymodel_state[i], 30.0, 25.0, center_wavelengths[i]);
	//	cout << skydome_result[i] << ", ";
	//}
	//cout << endl;
	//return 0;
}


HosekSkyModel::~HosekSkyModel() {
}

void HosekSkyModel::draw() {
	if (!shader) {
		cerr << "No shader set in " << __FILE__ << " on " << __LINE__ << endl;
		return;
	}
	shader->use();

	shader->setFloat("u_HorizonOffset", horizonOffset);

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

	vector<glm::vec3> params;

	eta = asinf(-sunDir.y);
	//cout << "ETA: " << eta << endl;
	//cout << "   degrees: " << glm::degrees(eta) << endl;

	if (!shouldUpdate(eta)) {
		return;
	}

	//double elev = pow(4.0 / (MATH_PI / 2.0), 1.0 / 3.0);
	double elev = (double)glm::radians(eta); // this doesn't make sense...
	//cout << "ELEV: " << elev << endl;

	skymodel_state = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo, elev);

	glm::vec3 tmp(0.0f);
	for (int j = 0; j < 9; j++) {
		for (int i = 0; i < 3; i++) {
			tmp[i] = skymodel_state->configs[i][j];
		}
		params.push_back(tmp);
	}
	for (int i = 0; i < 3; i++) {
		tmp[i] = skymodel_state->radiances[i];
	}
	params.push_back(tmp);

	glUniform3fv(glGetUniformLocation(shader->id, "u_Params"), 10, glm::value_ptr(params[0]));

	arhosekskymodelstate_free(skymodel_state);

	prevEta = eta;
	prevTurbidity = turbidity;
	prevAlbedo = albedo;
}

float HosekSkyModel::getElevationDegrees() {
	return glm::degrees(eta);
}

bool HosekSkyModel::shouldUpdate(float newEta) {
	return (newEta != prevEta || turbidity != prevTurbidity || albedo != prevAlbedo);
}
