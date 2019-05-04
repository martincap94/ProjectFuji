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

//! Helper class for drawing a simple bounding box (and possibly grid) of the LBM simulation area.
class GridLBM {
public:

	//! Loads the needed shader and prepares VBO for drawing.
	/*!
		\param[in] owner		The owning LBM simulator.
		\param[in] boxColor		Color of the bounding box that is to be drawn.
		\param[in] stepSize		--- NOT USED --- Size of step between grid lines.
	*/
	GridLBM(LBM3D_1D_indices *owner, glm::vec3 boxColor = glm::vec3(0.9f, 0.9f, 0.2f), glm::vec3 stepSize = glm::vec3(1.0f));

	//! Default destructor.
	~GridLBM();

	//! Draws the grid using owner's model matrix.
	void draw();

	//! Draws the grid using the given modelMatrix.
	/*!
		\param[in] modelMatrix	Model matrix to transform the grid/bounding box.
	*/
	void draw(glm::mat4 modelMatrix);

private:

	glm::vec3 stepSize;	//!< --- NOT USED --- Length between individual grid lines
	glm::vec3 boxColor = glm::vec3(0.9f, 0.9f, 0.2f);	//!< Color of the drawn bounding box.

	LBM3D_1D_indices *lbm = nullptr;	//!< LBM simulator owner for which we visualize its simulation area

	ShaderProgram *shader = nullptr;	//!< Shader used for drawing this grid
	GLuint boxVAO;	//!< VAO for the bounding box of LBM
	GLuint boxVBO;	//!< VBO for the bounding box of LBM

	std::vector<glm::vec3> gridVertices;	//!< List of vertices that create this grid/bounding box


};

