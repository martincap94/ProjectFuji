///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Grid3D.h
* \author     Martin Cap
*
*	--- DEPRECATED ---
*  Grid that is used when LBM 3D is drawn. It creates a 3D grid of dark grey lines with the given step
*  sizes. It also provides the option to draw points where the lattice nodes lie.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"

#include <glm\glm.hpp>

#include <glad\glad.h>
#include <vector>
#include "Grid.h"

#include "ShaderProgram.h"

/// Grid that is used for LBM 3D visualization. Draw 3D grid of lines and lattice nodes. Furthermore, draws the bounding box of the scene.
class Grid3D : public Grid {
public:
	/// Constructs the 3D grid with the given attributes.
	/**
		Constructs the 3D grid with the specified width, height and depth. Optionally, the step sizes
		in each axis can be set.
		\param[in] width	Width of the grid.
		\param[in] height	Height of the grid.
		\param[in] depth	Depth of the grid.
		\param[in] stepX	Step size on the x axis.
		\param[in] stepY	Step size on the y axis.
		\param[in] stepZ	Step size on the z axis.
	*/
	Grid3D(int width, int height, int depth, int stepX = 1, int stepY = 1, int stepZ = 1);
	
	/// Default destructor.
	~Grid3D();

	/// Draws the grid using the given shader program.
	virtual void draw(ShaderProgram &shader);

private:

	GLuint pointsVAO;	///< VAO of the lattice points
	GLuint pointsVBO;	///< VBO of the lattice points

	GLuint boxVAO;		///< VAO of the bounding box for the whole scene
	GLuint boxVBO;		///< VBO of the bounding box for the whole scene


};

