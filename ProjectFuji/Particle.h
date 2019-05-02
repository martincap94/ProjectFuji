///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Particle.h
* \author     Martin Cap
*
*	Describes a utility Particle class that is only used on CPU. This class is not the basis
*	of the simulator. Particle data are generally stored in VBOs and CUDA global memory instead
*	(and are therefore separated into arrays that contain individual particle properties)!
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>


class Particle {
public:

	glm::vec3 position;
	glm::vec3 velocity;
	float pressure;
	int profileIndex;

	Particle();
	~Particle();

	void updatePressureVal();

	float getPressureVal();

};


