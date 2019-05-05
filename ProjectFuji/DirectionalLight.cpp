#include "DirectionalLight.h"

#include "DataStructures.h"
#include "ShaderManager.h"
#include "ShaderProgram.h"
#include "Model.h"

#include <glm/gtc/matrix_transform.hpp>
#include <vector>

DirectionalLight::DirectionalLight() {

	shader = ShaderManager::getShaderPtr("singleColorModel");

	sunModel = new Model("models/unitsphere.fbx");
	sunModel->transform.scale = glm::vec3(100.0f);

}


DirectionalLight::~DirectionalLight() {
	if (sunModel) {
		delete sunModel;
	}
}

glm::mat4 DirectionalLight::getViewMatrix() {
	return glm::lookAt(position, focusPoint, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 DirectionalLight::getProjectionMatrix() {
	return glm::ortho(pLeft, pRight, pBottom, pTop, pNear, pFar);
}

void DirectionalLight::setProjectionMatrix(float left, float right, float bottom, float top) {
	pLeft = left;
	pRight = right;
	pBottom = bottom;
	pTop = top;
}

void DirectionalLight::setProjectionMatrix(float left, float right, float bottom, float top, float nearPlane, float farPlane) {
	setProjectionMatrix(left, right, bottom, top);
	pNear = nearPlane;
	pFar = farPlane;

}

glm::vec3 DirectionalLight::getDirection() {
	return glm::normalize(focusPoint - position);
}

void DirectionalLight::circularMotionStep(float deltaTime) {

	float rot = radius * cos(glm::radians(theta));

	if (rotationAxis == Y_AXIS) {
		position.x = rot;
		position.z = 0.0f;
	} else if (rotationAxis = Z_AXIS) {
		position.z = rot;
		position.x = 0.0f;
	}

	//position.z = radius * cos(glm::radians(theta)); // rotate around z
	position.y = radius * sin(glm::radians(theta));
	//position.z = focusPoint.z;
	//position.x = 0.0f;
	position += focusPoint; // offset by circle center (focus point)

	theta += circularMotionSpeed * deltaTime;
	if (skipNightTime) {
		if (theta >= 180.0f) {
			theta = 0.0f;
		}
	} else {
		if (theta >= 360.0f) {
			theta = 0.0f;
		}
	}

}

void DirectionalLight::draw() {
	shader->use();
	shader->setColor(glm::vec3(1.0f));

	glm::mat4 model(1.0f);
	model = glm::translate(model, position);
	shader->setModelMatrix(model);

	sunModel->transform.position = position;
	sunModel->transform.updateModelMatrix();
	sunModel->drawGeometry(shader);
}


void DirectionalLight::initBuffers() {
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), &cubeVertices[0], GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	/*vector<glm::vec3> vertices;

	vertices.push_back(glm::vec3(pLeft, pBottom, pNear));	
	vertices.push_back(glm::vec3(pRight, pBottom, pNear));

	vertices.push_back(glm::vec3(pRight, pBottom, pNear));
	vertices.push_back(glm::vec3(pRight, pTop, pNear));

	vertices.push_back(glm::vec3(pRight, pTop, pNear));
	vertices.push_back(glm::vec3(pLeft, pTop, pNear));

	vertices.push_back(glm::vec3(pLeft, pTop, pNear));
	vertices.push_back(glm::vec3(pLeft, pBottom, pNear));



	vertices.push_back(glm::vec3(pLeft, pBottom, pFar));
	vertices.push_back(glm::vec3(pRight, pBottom, pFar));

	vertices.push_back(glm::vec3(pRight, pBottom, pFar));
	vertices.push_back(glm::vec3(pRight, pTop, pFar));

	vertices.push_back(glm::vec3(pRight, pTop, pFar));
	vertices.push_back(glm::vec3(pLeft, pTop, pFar));

	vertices.push_back(glm::vec3(pLeft, pTop, pFar));
	vertices.push_back(glm::vec3(pLeft, pBottom, pFar));



	vertices.push_back(glm::vec3(pLeft, pBottom, pNear));
	vertices.push_back(glm::vec3(pLeft, pBottom, pFar));

	vertices.push_back(glm::vec3(pLeft, pTop, pNear));
	vertices.push_back(glm::vec3(pLeft, pTop, pFar));

	vertices.push_back(glm::vec3(pRight, pBottom, pNear));
	vertices.push_back(glm::vec3(pRight, pBottom, pFar));

	vertices.push_back(glm::vec3(pRight, pTop, pNear));
	vertices.push_back(glm::vec3(pRight, pTop, pFar));





	glGenVertexArrays(1, &projVAO);
	glGenBuffers(1, &projVBO);

	glBindVertexArray(projVAO);

	glBindBuffer(GL_ARRAY_BUFFER, projVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);*/

}
