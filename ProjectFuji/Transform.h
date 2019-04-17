#pragma once

#include <glm/glm.hpp>

#include <iostream>

class Actor;

class Transform {

public:

	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);	




	Transform();
	Transform(glm::vec3 position);
	Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale);
	~Transform();

	void updateModelMatrix();
	glm::mat4 getModelMatrix();

	void setOwner(Actor *owner);
	void unparent(bool keepWorldPosition = true);

	std::string toString();
	void print();

private:

	glm::vec3 prevPosition;
	glm::vec3 prevRotation;
	glm::vec3 prevScale;

	glm::mat4 modelMatrix;



	Actor *owner = nullptr;

};

