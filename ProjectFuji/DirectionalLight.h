///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       DirectionalLight.h
* \author     Martin Cap
* \date       2018/12/23
*
*  Basic directional light representation that is used to light the scene.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>
#include <glad\glad.h>


class ShaderProgram;
class Model;

//! DirectionalLight for simple scene lighting.
/**
	DirectionalLight that lights the scene.
	At the moment we use simple Blinn-Phong lighting, hence needing ambient, diffuse and specular values.
*/
class DirectionalLight {
public:

	//! Axis of rotation for the simplified sun simulation.
	enum eRotationAxis {
		Y_AXIS,		//!< Rotate around y axis
		Z_AXIS		//!< Rotate around z axis
	};

	ShaderProgram *shader;			//!< Default shader used for DirectionalLight visualization

	glm::mat4 projectionMatrix;		//!< Projection matrix of the light (its view of the scene)

	float pLeft = -30000.0f;		//!< Left value of the orthogonal projection
	float pRight = 30000.0f;		//!< Right value of the orthogonal projection
	float pBottom = -30000.0f;		//!< Bottom value of the orthogonal projection
	float pTop = 30000.0f;			//!< Top value of the orthogonal projection
	float pNear = 10.0f;			//!< Near value of the orthogonal projection
	float pFar = 600000.0f;			//!< Far value of the orthogonal projection

	// for circular motion -> overwrites position and direction (focus point is the center of the circular motion)
	float theta = 0.0f;						//!< Angle that keeps track of sun rotation [0, 360]
											//!< Note: this is not elevation or angle from zenith!
	float radius = 500000.0f;				//!< Radius of rotation around the focus point
	float circularMotionSpeed = 20.0f;		//!< Speed of the circular motion around the focus point
	eRotationAxis rotationAxis = Z_AXIS;	//!< Selected rotation axis of the circular motion
	int skipNightTime = 1;					//!< Whether to skip night time in the simplified sun rotation simulation

	glm::vec3 position;		//!< Position of the light from which we compute the direction
	//glm::vec3 direction;	//!< Direction of the light
	glm::vec3 focusPoint = glm::vec3(0.0f);		//!< Focus point of the light (used for computing projection matrix)

	glm::vec3 color = glm::vec3(1.0f);	//!< Color of the light
	float intensity = 8.0f;				//!< Intensity of the light used in PBR shaders
	
	//! Default constructor.
	DirectionalLight();
	//! Default destructor.
	~DirectionalLight();

	//! Returns the view matrix of this directional light (lookAt).
	glm::mat4 getViewMatrix();
	//! Returns the projection matrix of this directional light.
	glm::mat4 getProjectionMatrix();

	//! Sets the projection matrix parameters.
	void setProjectionMatrix(float left, float right, float bottom, float top);
	//! Sets the projection matrix parameters.
	void setProjectionMatrix(float left, float right, float bottom, float top, float nearPlane, float farPlane);

	//! Returns direction of the light (from light position to focus point).
	glm::vec3 getDirection();

	//! Do one step (one frame) of the circular motion.
	void circularMotionStep(float deltaTime = 1.0f);

	//! Draw a visualization mesh of the directional light.
	void draw();
	//! Initialize OpenGL buffers for the visualization mesh of the light.
	void initBuffers();

private:

	Model *sunModel = nullptr;	//!< Visualization model of the sun


	GLuint VAO;
	GLuint VBO;

	//GLuint projVAO;
	//GLuint projVBO;


};

