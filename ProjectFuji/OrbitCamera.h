///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       OrbitCamera.h
* \author     Martin Cap
*
*	Camera class that is used for orbiting a fixed focus point. Subclass of Camera.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

#include "Camera2D.h"
#include "Camera.h"

//! Camera that is used in the 3D viewport to orbit the scene.
/*!
	Camera that is used in 3D simulation that orbits around a given focus point.
	It supports setting front, side and top orthogonal views.
*/
class OrbitCamera : public Camera {
public:

	glm::vec3 focusPoint;		//!< Point at which the camera focuses (orients toward)
	glm::vec3 initFocusPoint;	//!< Initial focus point for resetting the camera


	float radius = 100.0f;		//!< Radius at which the camera orbits around the focus point

	//! Default constructor.
	OrbitCamera();

	//! Constructs orbit camera with given position, up vector, yaw and pitch angles, and focus point.
	/*!
		Creates a new camera with given position, up vector, yaw and pitch angles.
		\param[in] position		Camera position.
		\param[in] up			Initial up vector.
		\param[in] yaw			Initial yaw.
		\param[in] pitch		Initial pitch.
		\param[in] focusPoint	Focus point of the camera towards which it orients itself.
		\param[in] radius		Radius of the camera with which it will orbit around the focus point.
	*/
	OrbitCamera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f, glm::vec3 focusPoint = glm::vec3(0.0f), float radius = 100.0f);

	//! Default destructor.
	~OrbitCamera();


	virtual void processKeyboardMovement(eCameraMovementDirection direction, double deltaTime);
	virtual void processKeyboardMovement(int glfwKey, double deltaTime);

	//! Set the vie of this camera when orthogonal projection is used.
	virtual void setView(eCameraView camView);
	virtual void processMouseScroll(double yoffset);
	virtual void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = false);

private:

	virtual void updateCameraVectors();

};

