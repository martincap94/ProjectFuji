#include "Camera.h"

#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

using namespace std;

Camera::Camera() {}

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
	: position(position), up(up), yaw(yaw), pitch(pitch) {
}

Camera::~Camera() {}

glm::mat4 Camera::getViewMatrix() {
	return glm::lookAt(position, position + front, up);
}

//void Camera::processMouseScroll(double yoffset) {
//}
//
//void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
//	cout << "Process mouse movement in Camera" << endl;
//}

void Camera::setView(eCameraView camView) {}

void Camera::printInfo() {
	cout << "Camera position: " << glm::to_string(position) << endl;
}