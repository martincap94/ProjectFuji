#include "Transform.h"
#include <sstream>
#include <glm/gtc/matrix_transform.hpp>


Transform::Transform() : Transform(glm::vec3(), glm::vec3(), glm::vec3(1.0f)) {
}

Transform::Transform(glm::vec3 position) : Transform(position, glm::vec3(), glm::vec3(1.0f)) {
}


Transform::Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale) : position(position), rotation(rotation), scale(scale) {
}

Transform::~Transform() {
}

glm::mat4 Transform::getModelMatrix() const {
	glm::mat4 model(1.0f);
	model = glm::translate(model, position);

	// Only rotate around global y axis, full rotation to be implemented using quaternions!
	model = glm::rotate(model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));

	model = glm::scale(model, scale);

	return model;
}

std::string Transform::toString() {
	std::ostringstream oss;
	oss << "Position: " << position.x << " " << position.y << " " << position.z << std::endl;
	oss << "Rotation: " << rotation.x << " " << rotation.y << " " << rotation.z << std::endl;
	oss << "Scale:    " << scale.x    << " " << scale.y	   << " " << scale.z    << std::endl;
	return oss.str();
}

void Transform::print() {
	std::cout << toString();
}
