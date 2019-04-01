#pragma once

#include <glm/glm.hpp>
#include <iostream>

class Transform {

public:

	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);	

	Transform();
	Transform(glm::vec3 position);
	Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale);
	~Transform();

	glm::mat4 getModelMatrix() const;
	

	std::string toString();
	void print();

};

