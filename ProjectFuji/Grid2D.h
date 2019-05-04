///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Grid2D.h
* \author     Martin Cap
*
*	--- DEPRECATED ---
*  Grid that is used for LBM 2D visualization. Implements the Grid abstract class.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"

#include <glm\glm.hpp>

#include <glad/glad.h>
#include <vector>
#include "Grid.h"
#include "ShaderProgram.h"

//! Grid that is used for LBM 2D visualization.
class Grid2D : public Grid {

public:

	//! Constructor for the grid.
	/*!
		Constructs the grid with the provided width and height.
		Optionally the step in x and y axes can be set.
		\param[in] width	Desired width of the grid.
		\param[in] height	Desired height of the grid.
		\param[in] stepX	Size of the step in the x axis.
		\param[in] stepY	Size of the step in the y axis.
	*/
	Grid2D(int width, int height, int stepX = 1, int stepY = 1);

	//! Default destructor.
	~Grid2D();

	//! Draws the grid with using the provided shader program.
	virtual void draw(ShaderProgram &shader);
};

