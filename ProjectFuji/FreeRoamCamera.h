///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       FreeRoamCamera.h
* \author     Martin Cap
* \brief      Declaration of the FreeRoamCamera class.
*
*	Describes the FreeRoamCamera class, a subclass of abstract Camera class.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Camera.h"
#include "HeightMap.h"

#include <glm\glm.hpp>

class FreeRoamCamera : public Camera {
public:

	float mouseSensitivity = 0.1f;

	int walking = 0;
	float playerHeight = 1.8f;
	HeightMap *heightMap = nullptr;


	/// Constructs orbit camera with given position, up vector, yaw and pitch angles, and focus point.
	/**
	Creates a new camera with given position, up vector, yaw and pitch angles.
	\param[in] position		Camera position.
	\param[in] up			Initial up vector.
	\param[in] yaw			Initial yaw.
	\param[in] pitch		Initial pitch.
	\param[in] focusPoint	Focus point of the camera towards which it orients itself.
	\param[in] radius		Radius of the camera with which it will orbit around the focus point.
	*/
	FreeRoamCamera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);
	~FreeRoamCamera();

	virtual void processKeyboardMovement(eCameraMovementDirection direction, double deltaTime);
	virtual void processKeyboardMovement(int glfwKey, double deltaTime);

	void snapToGround();
	void snapToGround(HeightMap *heightMap);

	//void changeRotation(float yaw, float pitch);
	virtual void processMouseScroll(double yoffset);

	virtual void processMouseMovement(float xoffset, float yoffset, bool constrainPitch);

protected:

	virtual void updateCameraVectors();



};

