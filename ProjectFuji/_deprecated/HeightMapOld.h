///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       HeightMap.h
* \author     Martin Cap
* \date       2018/12/23
*
*  Describes the height map class that is used for terrain creation and rendering in 3D.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
//#include <glad\glad.h>

#include "ShaderProgram.h"

#include "Config.h"
#include "Texture.h"
#include "Material.h"

class VariableManager;

/// Height map that describes the terrain of 3D scenes.
/**
	Height map that describes the terrain of 3D scenes.
	Loads and renders the height map. The height map is loaded from .ppm ASCII file's red channel only!
	It is also used as an obstacle in the LBM 3D simulation when the y coordinate of the height map is compared
	to the lattice node y coordinate.
*/
class HeightMap {
public:

	int width;		///< Width of the height map
	int height;		///< Height of the height map - IMPORTANT - in the scene, the height of the map is described by the depth of the scene!!!

	int maxIntensity;	///< Maximum intensity of the height map - at the moment it will always be set to 255 due to .ppm usage
	
	float **data;		///< The height map data array		

	
	Material materials[MAX_TERRAIN_MATERIALS];


	Texture *diffuseTexture;
	Texture *normalMap;
	Texture *testDiffuse;

	Texture *secondDiffuseTexture;
	Texture *secondNormalMap;

	Texture *terrainNormalMap;
	Texture *materialMap;

	VariableManager *vars;


	ShaderProgram *shader;		///< Shader reference (that is used to render the height map terrain)

	/// Default constructor.
	HeightMap();

	/// Loads and constructs the height map from the given file and height.
	/**
		Constructs the height map from the given filename and height. Sets the shader that is used for rendering.
		\param[in] filename			Filename of the height map to be loaded.
		\param[in] latticeHeight	Height of the lattice for scaling (and normalization of height values).
	*/
	HeightMap(string filename, int latticeHeight);

	/// Deletes the height map data.
	~HeightMap();

	void initMaterials();
	void initBuffers();
	void initBuffersOld();

	float getHeight(float x, float z);


	/// Draws the height map.
	void draw();
	void draw(ShaderProgram *shader);
	void drawGeometry(ShaderProgram *shader);

private:

	GLuint VAO;		///< VAO of the height map
	GLuint VBO;		///< VBO of the height map


	int numPoints = 0;	///< Number of vertices; helper value for rendering

	glm::vec3 computeNormal(int x, int z);

};
