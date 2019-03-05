#include "Camera2D.h"

#include <iostream>
#include "glm/gtx/string_cast.hpp"

Camera2D::Camera2D() {
}

Camera2D::Camera2D(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Camera(position, up, yaw, pitch) {
	updateCameraVectors();
}



Camera2D::~Camera2D() {
}


void Camera2D::processKeyboardMovement(eCameraMovementDirection direction, double deltaTime) {
	float velocity = (float)((double)movementSpeed * deltaTime);

	if (direction == FORWARD) {
		position += front * velocity;
	}
	if (direction == BACKWARD) {
		position -= front * velocity;
	}
	if (direction == LEFT) {
		position -= right * velocity;
	}
	if (direction == RIGHT) {
		position += right * velocity;
	}
	if (direction == UP) {
		position += up * velocity;
	}
	if (direction == DOWN) {
		position -= up * velocity;
	}
	//if (direction == ROTATE_LEFT) {
	//	yaw -= velocity;
	//	updateCameraVectors();
	//}
	//if (direction == ROTATE_RIGHT) {
	//	yaw += velocity;
	//	updateCameraVectors();
	//}
	/*if (direction == ROTATE_LEFT) {
		yaw -= velocity;
		updateCameraVectors();
	}
	if (direction == ROTATE_RIGHT) {
		yaw += velocity;
		updateCameraVectors();
	}*/
}

void Camera2D::processKeyboardMovement(int glfwKey, double deltaTime) {
	
	float velocity = (float)((double)movementSpeed * deltaTime);

	if (glfwKey == GLFW_KEY_S) {
		position += up * velocity;
	}
	if (glfwKey == GLFW_KEY_W) {
		position -= up * velocity;
	}
	if (glfwKey == GLFW_KEY_A) {
		position -= right * velocity;
	}
	if (glfwKey == GLFW_KEY_D) {
		position += right * velocity;
	}

}


void Camera2D::processMouseScroll(double yoffset) {
	/*if (Zoom >= 1.0f && Zoom <= 45.0f) {
		Zoom -= yoffset;
	}
	if (Zoom <= 1.0f) {
		Zoom = 1.0f;
	}
	if (Zoom >= 45.0f) {
		Zoom = 45.0f;
	}*/
}

void Camera2D::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
}

void Camera2D::printInfo() {
	cout << "Camera position: " << glm::to_string(position) << endl;
}

void Camera2D::updateCameraVectors() {
	glm::vec3 tmp;
	tmp.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	tmp.y = sin(glm::radians(pitch));
	tmp.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	front = glm::normalize(tmp);
	right = glm::normalize(glm::cross(front, WORLD_UP));
	up = glm::normalize(glm::cross(right, front));
}
