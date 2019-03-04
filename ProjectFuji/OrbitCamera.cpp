#include "OrbitCamera.h"

#include <iostream>
#include "glm/gtx/string_cast.hpp"

OrbitCamera::OrbitCamera() {
}

OrbitCamera::OrbitCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch, glm::vec3 focusPoint, float radius) 
	: Camera(position, up, yaw, pitch), focusPoint(focusPoint), initFocusPoint(focusPoint), radius(radius) {
	updateCameraVectors();
}



OrbitCamera::~OrbitCamera() {
}


void OrbitCamera::processKeyboardMovement(eCameraMovementDirection direction, double deltaTime) {
	float velocity = (float)((double)movementSpeed * deltaTime);

	if (direction == FORWARD) {
		position += front * velocity;
		focusPoint += front * velocity;

	}
	if (direction == BACKWARD) {
		position -= front * velocity;
		focusPoint -= front * velocity;

	}
	if (direction == LEFT) {
		position -= right * velocity;
		focusPoint -= right * velocity;

	}
	if (direction == RIGHT) {
		position += right * velocity;
		focusPoint += right * velocity;

	}
	if (direction == UP) {
		position += up * velocity;
		focusPoint += up * velocity;

	}
	if (direction == DOWN) {
		position -= up * velocity;
		focusPoint -= up * velocity;

	}
	if (direction == ROTATE_LEFT) {
		yaw -= velocity;
		updateCameraVectors();
	}
	if (direction == ROTATE_RIGHT) {
		yaw += velocity;
		updateCameraVectors();
	}
}

void OrbitCamera::processMouseScroll(double yoffset) {
	//this->radius += yoffset;
	//updateCameraVectors();
}


void OrbitCamera::setView(eCameraView camView) {
	switch (camView) {
		case VIEW_FRONT:
			position = glm::vec3(latticeWidth / 2.0f, latticeHeight / 2.0f, latticeDepth * 2.0f);
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(1.0f, 0.0f, 0.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		case VIEW_SIDE:
			position = glm::vec3(latticeWidth * 2.0f, latticeHeight / 2.0f, latticeDepth / 2.0f);
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(0.0f, 0.0f, -1.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		case VIEW_TOP:
			position = glm::vec3(latticeWidth / 2.0f, latticeHeight * 2.0f, latticeDepth / 2.0f);
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(1.0f, 0.0f, 0.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		default:
			break;
	}
}

void OrbitCamera::printInfo() {
	cout << "Camera position: " << glm::to_string(position) << endl;
}

void OrbitCamera::updateCameraVectors() {
	float x = radius * sin(glm::radians(pitch)) * cos(glm::radians(yaw));
	float y = radius * cos(glm::radians(pitch));
	float z = radius * sin(glm::radians(pitch)) * sin(glm::radians(yaw));

	position = focusPoint + glm::vec3(x, y, z);

	front = glm::normalize(focusPoint - position);
	right = glm::normalize(glm::cross(front, WORLD_UP));
	up = glm::normalize(glm::cross(right, front));
}
