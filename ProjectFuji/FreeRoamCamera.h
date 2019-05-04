///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       FreeRoamCamera.h
* \author     Martin Cap
*
*	Describes the FreeRoamCamera class, a subclass of abstract Camera class.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Camera.h"
#include "HeightMap.h"

#include <glm\glm.hpp>

//! Camera used for free movement in the 3D scene.
/*!
	Rotation of the camera is done by mouse movement.
	For movement, W, S, A, D keys are used. Q and E are also used for upward/downward motion.
*/
class FreeRoamCamera : public Camera {
public:

	float mouseSensitivity = 0.1f;		//!< Sensitivity of the mouse movement

	int walking = 0;					//!< Whether the player is walking (camera is attached to ground)
	float playerHeight = 1.8f;			//!< Height of the player if walking
	HeightMap *heightMap = nullptr;		//!< Heightmap used for ground snapping


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
	FreeRoamCamera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);

	//! Default destructor.
	~FreeRoamCamera();

	virtual void processKeyboardMovement(eCameraMovementDirection direction, double deltaTime);
	virtual void processKeyboardMovement(int glfwKey, double deltaTime);

	//! Snaps the camera to ground using the member heightmap.
	void snapToGround();

	//! Snaps the camera to ground using the given heightmap.
	/*!
		\param[in] heightmap	Terrain heightmap to which we want to snap the camera.
	*/
	void snapToGround(HeightMap *heightMap);

	virtual void processMouseScroll(double yoffset);
	virtual void processMouseMovement(float xoffset, float yoffset, bool constrainPitch);

protected:

	virtual void updateCameraVectors();



};

