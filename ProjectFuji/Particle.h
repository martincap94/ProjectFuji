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

//! Simple particle description for CPU usage only.
/*!
	This Particle class is not actually used in any simulation, it is there just as a
	helper class when preparing data on the CPU.
*/
class Particle {
public:

	glm::vec3 position;		//!< Position of the particle (in world space)
	glm::vec3 velocity;		//!< Velocity of the particle
	float pressure;			//!< Pressure at the particle's altitude [hPa]
	int profileIndex;		//!< Index of the convective temperature profile the particle belongs to

	//! Updates the pressure value [hPa] for this particle based on its current altitude [m].
	void updatePressureVal();

	//! Returns the pressure value for this particle's altitude.
	/*!
		\return		Pressure value [hPa] of this particle's altitude.
	*/
	float getPressureVal();

};


