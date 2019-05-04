///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Skybox.h
* \author     Martin Cap
*
*	Simple Skybox class that generates simple geometry and cubemap to be used as a skybox.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>
#include "ShaderProgram.h"

//! Simple skybox using a cubemap.
class Skybox {
public:

	//! Loads the shader and prepares OpenGL buffers by calling setupSkybox().
	/*!
		\see setupSkybox()
	*/
	Skybox();
	//! Default destructor.
	~Skybox();

	//! Draws the skybox using the given view matrix.
	/*!
		\param[in] viewMatrix	View matrix to be used when drawing the cubemap.
	*/
	void draw(const glm::mat4 &viewMatrix);

private:

	ShaderProgram *shader = nullptr;	//!< Shader used for drawing the skybox

	//! Preset cubemap face textures.
	const std::vector<std::string> faces {
		"skybox/right.jpg",
		"skybox/left.jpg",
		"skybox/top.jpg",
		"skybox/bottom.jpg",
		"skybox/back.jpg",
		"skybox/front.jpg"
	};

	unsigned int VAO;				//!< VAO of the skybox
	unsigned int VBO;				//!< VBO of the skybox
	unsigned int EBO;				//!< EBO of the skybox
	unsigned int skyboxTexture;		//!< Cubemap texture

	//! Loads the cubemap texture and prepares OpenGL buffers.
	/*!
		Uses indexed skybox, therefore EBO is used.
	*/
	void setupSkybox();

};

