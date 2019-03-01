#include "DirectionalLight.h"

#include <glm/gtc/matrix_transform.hpp>

DirectionalLight::DirectionalLight() {

	// setup some default projection matrix
	//projectionMatrix = glm::ortho(-100.0f, 100.0f, -100.0f, 100.0f, 1.0f, 1000.0f);
	projectionMatrix = glm::perspective(glm::radians(90.0f), 1.0f, 1.0f, 1000.0f);

}


DirectionalLight::~DirectionalLight() {
}

glm::mat4 DirectionalLight::getViewMatrix() {
	//glm::mat4 lightView = glm::lookAt(dirLight.transform.position, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	//return glm::lookAt(-direction, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	return glm::lookAt(position, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 DirectionalLight::getProjectionMatrix() {
	return projectionMatrix;
}
