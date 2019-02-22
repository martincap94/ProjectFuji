#pragma once

#include <glm\glm.hpp>


class Particle {
public:

	glm::vec3 position;
	glm::vec3 velocity;
	float convectiveTemperature;
	float pressure; // pressure at its current height (can be computed from position.z)
	int profileIndex;

	Particle();
	~Particle();

	void updatePressureVal();

	float getPressureVal();

};


