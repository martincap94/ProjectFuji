///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Camera.h
* \author     Martin Cap
*
*	Abstract camera class that provides basic functionality interface for more specific camera types.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

#include <GLFW\glfw3.h>

#include "Config.h"

const glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f, 0.0f); //!< World up vector for the application (positive y axis).

//! Abstract Camera class.
/*!
	Abstract Camera class that provides basic functionality interface for other camera types.
*/
class Camera {

public:

	//! Enumeration of possible camera movement directions for keyboard controls.
	enum eCameraMovementDirection {
		FORWARD,		//!< Move forward
		BACKWARD,		//!< Move backwards
		LEFT,			//!< Move to the left
		RIGHT,			//!< Move to the right
		UP,				//!< Move up
		DOWN,			//!< Move down
		ROTATE_LEFT,	//!< Rotate to the left (for orbit cameras)
		ROTATE_RIGHT	//!< Rotate to the right (for orbit cameras)
	};

	//! Enumeration of possible camera views that can be selected. These pertain to OrbitCameras only.
	enum eCameraView {
		VIEW_FRONT,		//!< View the scene from the front
		VIEW_SIDE,		//!< View the scene from the side
		VIEW_TOP		//!< View the scene from the top
	};



	glm::vec3 position;		//!< Current position
	glm::vec3 front;		//!< Front vector
	glm::vec3 up;			//!< Up vector
	glm::vec3 right;		//!< Right vector

	float yaw;				//!< Yaw angle
	float pitch;			//!< Pitch angle
	float roll;				//!< Roll angle


	float movementSpeed = DEFAULT_CAMERA_SPEED;	//!< Movement speed of the camera


	//! Default camera constructor
	Camera();

	//! Camera constructor
	/*!
		Creates a new camera with given position, up vector, yaw and pitch angles.
		\param[in] position		Camera position.
		\param[in] up			Initial up vector.
		\param[in] yaw			Initial yaw.
		\param[in] pitch		Initial pitch.
	*/
	Camera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);

	//! Default camera destructor
	~Camera();

	//! Returns the view matrix that was computed for the current camera orientation and position
	virtual glm::mat4 getViewMatrix();

	//! Processes keyboard input and moves the camera accordingly.
	/*!
		Processes keyboard input and moves the camera accordingly. The movement depends on the camera type (2D or 3D).
		\param[in] direction		eCameraMovementDirection value that describes the direction of the movement.
		\param[in] deltaTime		Delta time of the rendered frame for frame dependent movement (and possible smoothing).
	*/
	virtual void processKeyboardMovement(eCameraMovementDirection direction, double deltaTime) = 0;

	//! Processes keyboard input and moves the camera accordingly.
	/*!
		Processes keyboard input and moves the camera accordingly. The movement depends on the camera type (2D or 3D).
		\param[in] glfwKey			GLFW key code of the key that is to be processed.
	*/
	virtual void processKeyboardMovement(int glfwKey, double deltaTime) = 0;

	//! Processes mouse scroll.
	virtual void processMouseScroll(double yoffset) = 0;

	//! Processes mouse movement.
	virtual void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) = 0;

	//! Sets the view of the camera (based on the eCameraView enum).
	/*!
		Sets the view of the camera - used only in 3D at the moment.
		\param[in] camView			The camera view to be used.
	*/
	virtual void setView(eCameraView camView);

	//! Prints information about the camera position.
	virtual void printInfo();


protected:

	//! Computes the camera orientation based on the camera angles (yaw, pitch, roll) or using other metrics.
	virtual void updateCameraVectors() = 0;

};

