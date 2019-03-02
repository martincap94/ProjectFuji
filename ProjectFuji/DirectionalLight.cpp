#include "DirectionalLight.h"

#include <glm/gtc/matrix_transform.hpp>

DirectionalLight::DirectionalLight() {

	// setup some default projection matrix
	//projectionMatrix = glm::ortho(-100.0f, 100.0f, -100.0f, 100.0f, 1.0f, 1000.0f);
	//projectionMatrix = glm::perspective(glm::radians(90.0f), 1.0f, 1.0f, 1000.0f);


}


DirectionalLight::~DirectionalLight() {
}

glm::mat4 DirectionalLight::getViewMatrix() {
	//glm::mat4 lightView = glm::lookAt(dirLight.transform.position, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	//return glm::lookAt(-direction, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	

	return glm::lookAt(position, focusPoint, glm::vec3(0.0f, 1.0f, 0.0f));

	//return glm::lookAt(position, position + direction, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 DirectionalLight::getProjectionMatrix() {
	return glm::ortho(pLeft, pRight, pBottom, pTop, pNear, pFar);
	//return projectionMatrix;
}

void DirectionalLight::setProjectionMatrix(float left, float right, float bottom, float top) {
	//projectionMatrix = glm::ortho(left, right, bottom, top, near, far);
	pLeft = left;
	pRight = right;
	pBottom = bottom;
	pTop = top;
}

void DirectionalLight::setProjectionMatrix(float left, float right, float bottom, float top, float near, float far) {
	setProjectionMatrix(left, right, bottom, top);
	pNear = near;
	pFar = far;

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
	if (theta >= 360.0f) {
		theta = 0.0f;
	}

}
