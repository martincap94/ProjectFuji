///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       DirectionalLight.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Directional light object for scene lighting.
*
*  Basic directional light representation that is used to light the scene.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>
#include <glad\glad.h>

#include "ShaderProgram.h"

/// Direction light for simple scene lighting.
/**
	Directional light that lights the scene.
	At the moment we use simple Blinn-Phong lighting, hence needing ambient, diffuse and specular values.
*/
class DirectionalLight {
public:

	enum eRotationAxis {
		Y_AXIS,
		Z_AXIS
	};

	ShaderProgram *shader;

	glm::mat4 projectionMatrix;
	//glm::mat4 viewMatrix;

	float pLeft = -100.0f;
	float pRight = 100.0f;
	float pBottom = -100.0f;
	float pTop = 100.0f;
	float pNear = 1.0f;
	float pFar = 1000.0f;

	// for circular motion -> overwrites position and direction (focus point is the center of the circular motion)
	float theta = 0.0f;
	float radius = 250.0f;
	float circularMotionSpeed = 20.0f;
	eRotationAxis rotationAxis = Z_AXIS;

	glm::vec3 position;
	glm::vec3 direction;	///< Direction of the light
	glm::vec3 focusPoint = glm::vec3(0.0f);
	glm::vec3 up;

	glm::vec3 ambient;		///< Ambient value
	glm::vec3 diffuse;		///< Diffuse value
	glm::vec3 specular;		///< Specular value
	
	DirectionalLight();		///< Default constructor.
	~DirectionalLight();	///< Default destructor.

	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();

	void setProjectionMatrix(float left, float right, float bottom, float top);
	void setProjectionMatrix(float left, float right, float bottom, float top, float nearPlane, float farPlane);


	void circularMotionStep(float deltaTime = 1.0f);


	void draw();
	void init();
	void initBuffers();

private:

	GLuint VAO;
	GLuint VBO;

	GLuint projVAO;
	GLuint projVBO;


};

