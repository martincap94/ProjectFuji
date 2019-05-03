///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Transform.h
* \author     Martin Cap
*
*	Simple Transform class that stores object's position. Should be part of any object that is 
*	situated in the 3D scene.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm/glm.hpp>

#include <iostream>

class Actor;

//! Describes basic transform properties of actors.
/*!
	Describes basic properties of objects such as position, rotation and their scale.
	Uses GLM library to produce model matrices.
	Supports parenting through owning actors.
*/
class Transform {

public:

	glm::vec3 position = glm::vec3(0.0f);		//!< Position of the actor
	glm::vec3 rotation = glm::vec3(0.0f);		//!< Rotation of the actor
	glm::vec3 scale = glm::vec3(1.0f);			//!< Scale of the actor



	//! Constructs the transform with default values (position in origin, no rotation and unit scale).
	Transform();

	//! Constructs the transform with given position and default rotation and scale.
	Transform(glm::vec3 position);

	//! Constructs the transform with given parameters.
	Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale);

	//! Default destructor.
	~Transform();

	//! Recalculates the model matrix based on current position, rotation and scale.
	void updateModelMatrix();

	//! Returns the currently saved model matrix.
	glm::mat4 getSavedModelMatrix();

	//! Recalculates the model matrix and returns it.
	glm::mat4 getModelMatrix();

	//! Sets the owner of this transform.
	void setOwner(Actor *owner);

	//! Unparents the transform from its parent transform.
	/*!
		\param[in] keepWorldPosition	If true, we update the transform so the unparented actor remains at the same place.
	*/
	void unparent(bool keepWorldPosition = true);

	//! Returns string description of the transform.
	std::string toString();

	//! Prints description of this transform to the console.
	void print();

private:

	glm::vec3 prevPosition;	//!< --- NOT USED --- Previous position of the transform
	glm::vec3 prevRotation;	//!< --- NOT USED --- Previous rotation of the transform
	glm::vec3 prevScale;	//!< --- NOT USED --- Previous scale of the transform

	glm::mat4 modelMatrix;	//!< Saved model matrix

	Actor *owner = nullptr; //!< The owning actor

};

