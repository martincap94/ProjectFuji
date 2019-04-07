#pragma once

#include <glm\glm.hpp>


class Particle {
public:

	glm::vec3 position;
	glm::vec3 velocity;
	//float convectiveTemperature = 0.0f;
	float pressure; // pressure at its current height (can be computed from position.z) -> TODO, remove references and remove this member variable
	int profileIndex;

	Particle();
	~Particle();

	void updatePressureVal();

	float getPressureVal();

};


