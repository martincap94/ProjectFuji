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

/// Direction light for simple scene lighting.
/**
	Directional light that lights the scene.
	At the moment we use simple Blinn-Phong lighting, hence needing ambient, diffuse and specular values.
*/
class DirectionalLight {
public:

	glm::vec3 direction;	///< Direction of the light

	glm::vec3 ambient;		///< Ambient value
	glm::vec3 diffuse;		///< Diffuse value
	glm::vec3 specular;		///< Specular value
	
	DirectionalLight();		///< Default constructor.
	~DirectionalLight();	///< Default destructor.

};

