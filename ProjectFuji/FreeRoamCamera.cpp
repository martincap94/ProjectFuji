#include "FreeRoamCamera.h"

#include <iostream>


using namespace std;


FreeRoamCamera::FreeRoamCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
	: Camera(position, up, yaw, pitch) {
	updateCameraVectors();
}




FreeRoamCamera::~FreeRoamCamera() {
}

void FreeRoamCamera::processKeyboardMovement(eCameraMovementDirection direction, double deltaTime) {
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
	if (direction == ROTATE_LEFT) {
		yaw -= velocity;
		updateCameraVectors();
	}
	if (direction == ROTATE_RIGHT) {
		yaw += velocity;
		updateCameraVectors();
	}
}

void FreeRoamCamera::processKeyboardMovement(int glfwKey, double deltaTime) {
	float velocity = (float)((double)movementSpeed * deltaTime);

	if (glfwKey == GLFW_KEY_W) {
		position += front * velocity;
	}
	if (glfwKey == GLFW_KEY_S) {
		position -= front * velocity;
	}
	if (glfwKey == GLFW_KEY_A) {
		position -= right * velocity;
	}
	if (glfwKey == GLFW_KEY_D) {
		position += right * velocity;
	}
	if (glfwKey == GLFW_KEY_E) {
		position += up * velocity;
	}
	if (glfwKey == GLFW_KEY_Q) {
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
	if (walking && heightMap) {
		updateCameraVectors();
	}

}

void FreeRoamCamera::snapToGround() {
	position.y = heightMap->getHeight(position.x, position.z) + playerHeight;
}

void FreeRoamCamera::snapToGround(HeightMap * heightMap) {
	position.y = heightMap->getHeight(position.x, position.z) + playerHeight;
}

//void FreeRoamCamera::changeRotation(float yaw, float pitch) {
//	this->yaw = yaw;
//	this->pitch = pitch;
//}

void FreeRoamCamera::processMouseScroll(double yoffset) {
	// scrolling up is more demanding, multiply it
	if (yoffset > 0.0) {
		yoffset *= 1.5;
	}
	float fineGrainControlMultiplier = glm::clamp(movementSpeed / 10.0f, 0.1f, 500.0f);
	movementSpeed += fineGrainControlMultiplier * (float)yoffset;
	if (movementSpeed <= 0.1f) {
		movementSpeed = 0.1f;
	}
}

void FreeRoamCamera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {

	xoffset *= mouseSensitivity;
	yoffset *= mouseSensitivity;

	yaw += xoffset;
	pitch += yoffset;

	//cout << "yaw = " << yaw << ", pitch = " << pitch << endl;

	if (constrainPitch) {
		if (pitch > 90.0f) {
			pitch = 90.0f;
		} else if (pitch < -90.0f) {
			pitch = -90.0f;
		}
	}
	updateCameraVectors();

}

void FreeRoamCamera::updateCameraVectors() {

	if (walking && heightMap) {
		position.y = heightMap->getHeight(position.x, position.z) + playerHeight;
	}

	glm::vec3 tmpFront;
	tmpFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	tmpFront.y = sin(glm::radians(pitch));
	tmpFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	front = glm::normalize(tmpFront);
	right = glm::normalize(glm::cross(tmpFront, WORLD_UP));
	up = glm::normalize(glm::cross(right, tmpFront));
}


