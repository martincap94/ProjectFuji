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

void Camera::setView(eCameraView camView) {}

void Camera::printInfo() {
	cout << "Camera position: " << glm::to_string(position) << endl;
}

void Camera::setLatticeDimensions(int latticeWidth, int latticeHeight, int latticeDepth) {
	this->latticeWidth = latticeWidth;
	this->latticeHeight = latticeHeight;
	this->latticeDepth = latticeDepth;
}
