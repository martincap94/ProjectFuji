///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Camera2D.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Camera class for 2D LBM visualization.
*
*  Camera class that is used when the 2D view is used (e.g. diagram view).
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"
#include "Camera.h"

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

/// Camera that is used for LBM 2D simulation.
/**
	Simple camera that is used to view the LBM 2D simulation scene.
*/
class Camera2D : public Camera {
public:

	/// Default constructor.
	Camera2D();

	/// Camera2D constructor.
	/**
		Construct the Camera2D instance with given position, up vector and yaw and pitch angles.
		\param[in] position		Camera position.
		\param[in] up			Initial up vector.
		\param[in] yaw			Initial yaw.
		\param[in] pitch		Initial pitch.
	*/
	Camera2D(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);
	
	/// Default destructor.
	~Camera2D();

	// note: doxygen descriptions should be inherited from base class

	//virtual glm::mat4 getViewMatrix();
	virtual void processKeyboardMovement(eCameraMovementDirection direction, double deltaTime);
	virtual void processKeyboardMovement(int glfwKey, double deltaTime);

	virtual void processMouseScroll(double yoffset);
	virtual void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = false);
	virtual void printInfo();

private:

	virtual void updateCameraVectors();

};

