///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       HeightMap.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      The height map that is used for terrain creation in 3D.
*
*  Describes the height map class that is used for terrain creation and rendering in 3D.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <glad\glad.h>

#include "ShaderProgram.h"

#include "Config.h"
#include "Texture.h"
#include "Material.h"
#include "PBRMaterial.h"
#include "CDFSampler.h"
#include "CDFSamplerMultiChannel.h"

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


	int downSample = 10;

	int maxIntensity;	///< Maximum intensity of the height map - at the moment it will always be set to 255 due to .ppm usage
	
	float *data = nullptr;		///< The height map data array	

	int showNormalsOnly = 0;
	int normalsShaderMode = 0;

	glm::vec2 terrainHeightRange;
	float texelWorldSize;

	float globalNormalMapMixingRatio;
	float globalNormalMapTiling = 10.0f;

	int useGrungeMap = 0;
	Texture *grungeMap = nullptr;
	float grungeMapMin = 0.0f;
	float grungeMapTiling = 8.0f;

	float ambientIntensity = 0.05f;
	float diffuseIntensity = 0.9f;
	float specularIntensity = 0.75f;


	int activeMaterialCount = 0;


	VariableManager *vars;
	std::string heightMapFilename = "";


	Material materials[MAX_TERRAIN_MATERIALS];
	PBRMaterial pbrMaterials[MAX_TERRAIN_MATERIALS - 1];

	Texture *terrainNormalMap;
	Texture *materialMap;
	CDFSamplerMultiChannel *materialMapSampler = nullptr;

	int visualizeTextureMode = 0;
	Texture *visTexture = nullptr;


	ShaderProgram *shader;		///< Shader reference (that is used to render the height map terrain)
	//ShaderProgram *wireframeShader;


	HeightMap(VariableManager *vars);


	/// Deletes the height map data.
	~HeightMap();

	void smoothHeights();

	void loadHeightMapData();
	void loadHeightMapData(std::string filename);

	void initMaterials();
	void initBuffers();
	void createAndUploadMesh();

	void initBuffersOld();

	float getHeight(float x, float z, bool worldPosition = true);
	float getWorldWidth();
	float getWorldDepth();

	glm::vec3 getSampleWorldPosition(glm::ivec2 sampleCoords);
	glm::vec3 getWorldPositionSample(CDFSampler *sampler);
	glm::vec3 getWorldPositionMultiChannelSample(CDFSamplerMultiChannel *sampler, int channel);
	glm::vec3 getWorldPositionMaterialMapSample(int materialIdx);

	/// Draws the height map.
	void draw();
	void draw(ShaderProgram *shader);
	void drawGeometry(ShaderProgram *shader);

private:

	const int materialMapTextureUnit = 12;
	const int normalMapTextureUnit = 13;
	const int grungeMapTextureUnit = 14;

	GLuint VAO;		///< VAO of the height map
	GLuint VBO;		///< VBO of the height map


	int numPoints = 0;	///< Number of vertices; helper value for rendering

	glm::vec3 computeNormal(int x, int z);

};

