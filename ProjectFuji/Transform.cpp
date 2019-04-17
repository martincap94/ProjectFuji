#include "Transform.h"

#include "Actor.h"

#include <sstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>


Transform::Transform() : Transform(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f)) {
}

Transform::Transform(glm::vec3 position) : Transform(position, glm::vec3(0.0f), glm::vec3(1.0f)) {
}


Transform::Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale) : position(position), rotation(rotation), scale(scale) {
	prevPosition = position;
	prevRotation = rotation;
	prevScale = scale;
}

Transform::~Transform() {
}

void Transform::updateModelMatrix() {
	// It would make sense to store these results and not calculate them again if no changes to the model position/scale/rotation were made

	glm::mat4 model(1.0f);

	if (owner != nullptr && owner->parent != nullptr) {
		model = model * owner->parent->transform.getModelMatrix();
	}


	glm::quat rotQ = glm::quat(glm::radians(rotation));

	// Beware of GLM's multiplication order!
	model = glm::translate(model, position);
	model = model * glm::toMat4(rotQ);
	model = glm::scale(model, scale);

	modelMatrix = model;
}

glm::mat4 Transform::getModelMatrix() {
	return modelMatrix;
}


void Transform::setOwner(Actor *owner) {
	this->owner = owner;
}

void Transform::unparent(bool keepWorldPosition) {
	if (keepWorldPosition) {
		if (owner->parent != nullptr) {
			position += owner->parent->transform.position;
			rotation += owner->parent->transform.rotation;
			scale *= owner->parent->transform.scale;
		}
	}
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
