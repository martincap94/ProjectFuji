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

class Skybox {
public:
	Skybox();
	~Skybox();

	void draw(const glm::mat4 &viewMatrix);

private:

	ShaderProgram *shader = nullptr;

	const std::vector<std::string> faces {
		"skybox/right.jpg",
		"skybox/left.jpg",
		"skybox/top.jpg",
		"skybox/bottom.jpg",
		"skybox/back.jpg",
		"skybox/front.jpg"
	};

	unsigned int VAO;
	unsigned int VBO;
	unsigned int EBO;
	unsigned int skyboxTexture;


	void setupSkybox();

};

