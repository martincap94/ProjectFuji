///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       LatticeCollider.h
* \author     Martin Cap
* \date       2018/12/23
* \brief	  Description of LatticeCollider class.
*
*  Describes the LatticeCollider that is used in LBM 2D. It represents obstacles in the scene
*  around which the particles should flow.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"
#include <string>
#include "ShaderProgram.h"

/// Lattice collider that is used as an obstacle in LBM 2D simulation.
/**
	Lattice collider is used as an obstacle in LBM 2D. It can be drawn when rendering the scene.
*/
class LatticeCollider {
public:

	int width;			///< Width of the scene
	int height;			///< Height of the scene
	int maxIntensity;	///< Maximum intensity of the image color values, usually 255 since we use .ppm format

	bool *area;			///< Area of the collider

	/// Constructs the collider from the given file.
	/**
		Constructs the collider from the given file.
		\param[in] filename		Name of the file to be used.
	*/
	LatticeCollider(string filename);

	/// Destructs the collider by deleting the area array.
	~LatticeCollider();

	/// Draws the collider using the provided shader program.
	void draw(ShaderProgram &shader);

private:

	int numPoints = 0;	///< Number of points of the collider to be drawn (visualized as set of points) 
	GLuint VAO;
	GLuint VBO;

};

