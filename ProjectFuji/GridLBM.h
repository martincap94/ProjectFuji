///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       GridLBM.h
* \author     Martin Cap
*
*	Describes the GridLBM class that is used to draw a bounding volume and a grid of the LBM 
*	simulation area.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>
#include <vector>


class LBM3D_1D_indices;
class ShaderProgram;

class GridLBM {
public:

	GridLBM(LBM3D_1D_indices *owner, glm::vec3 boxColor = glm::vec3(0.9f, 0.9f, 0.2f), glm::vec3 stepSize = glm::vec3(1.0f));
	~GridLBM();

	void draw();
	void draw(glm::mat4 modelMatrix);

private:

	glm::vec3 stepSize;
	glm::vec3 boxColor = glm::vec3(0.9f, 0.9f, 0.2f);

	LBM3D_1D_indices *lbm = nullptr;

	ShaderProgram *shader = nullptr;
	GLuint boxVAO;
	GLuint boxVBO;

	std::vector<glm::vec3> gridVertices;


};

