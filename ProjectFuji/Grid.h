///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Grid.h
* \author     Martin Cap
*	
*	--- DEPRECATED ---
*	Abstract class describing the grid to be used in LBM visualization. Provides polymorphism when
*	we use either LBM 2D or 3D (with Grid2D and Grid3D subclasses).
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ShaderProgram.h"

#include <vector>
#include <glad\glad.h>

/// Abstract grid class.
/**
	Grid that is used for the LBM visualization.
*/
class Grid {
public:

	/// Default constructor. Does nothing.
	Grid();
	/// Default destructor. Does nothing.
	~Grid();

	/// Draw the grid using the provided shader program.
	virtual void draw(ShaderProgram &shader) = 0;

protected:

	GLuint VAO;		///< VAO for this grid
	GLuint VBO;		///< VBO for this grid

	vector<glm::vec3> gridVertices;		///< Grid vertices


};

