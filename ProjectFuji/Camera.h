///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Camera.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      The abstract camera class.
*
*  Abstract camera class that provides basic functionality descriptions for the application.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>
#include "Config.h"

const glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f, 0.0f); ///< 

/// Abstract camera class.
/**
	This class defines the basic functionality of the camera.
*/
class Camera {

public:

	/// Enumeration of possible camera movement directions for keyboard controls.
	enum eCameraMovementDirection {
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT,
		UP,
		DOWN,
		ROTATE_LEFT,
		ROTATE_RIGHT
	};

	/// Enumeration of possible camera views that can be selected.
	enum eCameraView {
		VIEW_FRONT,
		VIEW_SIDE,
		VIEW_TOP
	};



	glm::vec3 position;		///< Current position
	glm::vec3 front;		///< Front vector
	glm::vec3 up;			///< Up vector
	glm::vec3 right;		///< Right vector

	float yaw;				///< Yaw angle
	float pitch;			///< Pitch angle
	float roll;				///< Roll angle

	int latticeWidth;		///< Width of the lattice for computations
	int latticeHeight;		///< Height of the lattice for computations
	int latticeDepth;		///< Depth of the lattice for computations

	float movementSpeed = DEFAULT_CAMERA_SPEED;	///< Movement speed of the camera


	/// Default camera constructor
	Camera();

	/// Camera constructor
	/**
		Creates a new camera with given position, up vector, yaw and pitch angles.
		\param[in] position		Camera position.
		\param[in] up			Initial up vector.
		\param[in] yaw			Initial yaw.
		\param[in] pitch		Initial pitch.
	*/
	Camera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);

	/// Default camera destructor
	~Camera();

	/// Returns the view matrix that was computed for the current camera orientation and position
	virtual glm::mat4 getViewMatrix() = 0;

	/// Processes keyboard input and moves the camera accordingly.
	/**
		Processes keyboard input and moves the camera accordingly. The movement depends on the camera type (2D or 3D).
		\param[in] direction		eCameraMovementDirection value that describes the direction of the movement.
		\param[in] deltaTime		Delta time of the rendered frame for frame dependent movement (and possible smoothing).
	*/
	virtual void processKeyboardMovement(eCameraMovementDirection direction, double deltaTime) = 0;

	/// Process mouse scroll - unused at the moment!
	virtual void processMouseScroll(double yoffset) = 0;

	/// Sets the view of the camera (based on the eCameraView enum).
	/**
		Sets the view of the camera - used only in 3D at the moment.
		\param[in] camView			The camera view to be used.
	*/
	virtual void setView(eCameraView camView);

	/// Prints information about the camera position.
	virtual void printInfo();

	/// Setter for lattice dimensions.
	void setLatticeDimensions(int latticeWidth, int latticeHeight, int latticeDepth);

protected:

	/// Computes the camera orientation based on the camera angles (yaw, pitch, roll) or on other metrics.
	virtual void updateCameraVectors() = 0;

};

