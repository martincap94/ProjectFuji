///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       GeneralGrid.h
* \author     Martin Cap
* \brief      GeneralGrid class declaration.
* 
*	Describes the GeneralGrid class that is used to draw main axes and helper groud grid.
*	
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>

#include "Config.h"
#include "ShaderProgram.h"

/// General grid that shows main axes and x/y grid for easier orientation.
/**
	General grid that shows main axes and x/y grid for easier orientation.
	The step size of the grid can be configured as well as its range.
	The x, y and z axes are standardly colored red, green and blue.
*/
class GeneralGrid {
public:

	/// Default constructor.
	GeneralGrid();

	/// Constructs general grid with given range and step size.
	/**
		Constructs general grid with given range and step size.
		\param[in] range		Range of the x/y grid.
		\param[in] stepSize		Size of the spacing between lines in the x/y grid.
	*/
	GeneralGrid(float range, float stepSize);

	/// Default destructor.
	~GeneralGrid();

	void draw();

private:

	ShaderProgram *unlitColorShader;

	GLuint VAO;		///< VAO for the grid
	GLuint VBO;		///< VBO for the grid

	int numLines;	///< Number of lines to draw for the x/y grid

	float range;		///< Range of the x/y grid
	float stepSize;	///< Number of steps (spacing) between individual lines of the x/y grid


};

