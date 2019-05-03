///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       TerrainPicker.h
* \author     Martin Cap
*
*	Helper TerrainPicker class that draws terrain using special shader. Using pixel read back
*	it can determine world position of the given screen pixel. This gives us the ability to
*	pick any point on the terrain with pixel perfect precision.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>

#include <glm\glm.hpp>

#include "VariableManager.h"
#include "ShaderProgram.h"
#include "HeightMap.h"

//! Helper class that gives us the option to pick any point on the terrain using the mouse cursor.
/*!
	To obtain correct results, the terrain must be first drawn.
*/
class TerrainPicker {
public:

	//! Initializes the TerrainPicker framebuffers and sets its shader.
	/*!
		\param[in] vars		VariableManager to be used by the TerrainPicker.
	*/
	TerrainPicker(VariableManager *vars);

	//! Default destructor.
	~TerrainPicker();

	//! Draws the terrain using the custom shader.
	void drawTerrain();

	//! Draws the given terrain using the custom shader.
	/*!
		\param[in] heightMap	The terrain to be drawn.
	*/
	void drawTerrain(HeightMap *heightMap);

	//! Initializes the helper framebuffers that are used when drawing the terrain.
	void initFramebuffer();

	//! Returns the world position of a screen pixel with the given coordinates.
	/*!
		\param[in] x					Screen x coordinate of the pixel.
		\param[in] y					Screen y coordinate of the pixel.
		\param[out] outTerrainHit		Whether the terrain has been hit.
		\param[in] invertY				Whether to invert the screen coordinate y.
		\return							The world position of the screen pixel.
	*/
	glm::vec3 getPixelData(int x, int y, bool &outTerrainHit, bool invertY = true);

	//! Returns the world position of a screen pixel with the given coordinates.
	/*!
		\param[in] screenPos			Screen x, y coordinates of the pixel.
		\param[out] outTerrainHit		Whether the terrain has been hit.
		\param[in] invertY				Whether to invert the screen coordinate y.
		\return							The world position of the screen pixel.
	*/
	glm::vec3 getPixelData(glm::ivec2 screenPos, bool &outTerrainHit, bool invertY = true);

private:

	ShaderProgram *shader = nullptr;	//!< Custom shader used to draw the terrain

	VariableManager *vars = nullptr;	//!< Pointer to the used VariableManager

	GLuint framebuffer;		//!< ID of the main framebuffer
	GLuint texture;			//!< Color (float!) attachment of the main framebuffer
	GLuint depthTexture;	//!< Depth attachment of the main framebuffer


};

