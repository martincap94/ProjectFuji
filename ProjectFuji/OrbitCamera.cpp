#include "OrbitCamera.h"

#include <iostream>
#include "glm/gtx/string_cast.hpp"

using namespace std;

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

void OrbitCamera::processKeyboardMovement(int glfwKey, double deltaTime) {
	float velocity = movementSpeed * (float)deltaTime;
	float rotationVelocity = rotationSpeed * (float)deltaTime;


	//if (glfwKey == ) {
	//	position += front * velocity;
	//	focusPoint += front * velocity;

	//}
	//if (direction == BACKWARD) {
	//	position -= front * velocity;
	//	focusPoint -= front * velocity;

	//}
	if (glfwKey == GLFW_KEY_A) {
		position -= right * velocity;
		focusPoint -= right * velocity;

	}
	if (glfwKey == GLFW_KEY_D) {
		position += right * velocity;
		focusPoint += right * velocity;

	}
	if (glfwKey == GLFW_KEY_W) {
		position += up * velocity;
		focusPoint += up * velocity;

	}
	if (glfwKey == GLFW_KEY_S) {
		position -= up * velocity;
		focusPoint -= up * velocity;

	}
	if (glfwKey == GLFW_KEY_E) {
		yaw -= rotationVelocity;
		updateCameraVectors();
	}
	if (glfwKey == GLFW_KEY_Q) {
		yaw += rotationVelocity;
		updateCameraVectors();
	}
}

void OrbitCamera::processMouseScroll(double yoffset) {
	//this->radius += yoffset;
	//updateCameraVectors();
}

void OrbitCamera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
}


void OrbitCamera::setView(eCameraView camView) {
	switch (camView) {
		case VIEW_FRONT:
			position = initFocusPoint + glm::vec3(0.0f, 0.0f, 1.0f) * radius;
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(1.0f, 0.0f, 0.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		case VIEW_SIDE:
			position = initFocusPoint + glm::vec3(1.0f, 0.0f, 0.0f) * radius;
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(0.0f, 0.0f, -1.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		case VIEW_TOP:
			position = initFocusPoint + glm::vec3(0.0f, 1.0f, 0.0f) * radius;
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(1.0f, 0.0f, 0.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		default:
			break;
	}
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
