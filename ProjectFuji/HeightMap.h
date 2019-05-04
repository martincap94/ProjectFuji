///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       HeightMap.h
* \author     Martin Cap
* \date       2018/12/23
*
*  Describes the HeightMap class that is used for terrain creation and rendering in 3D.
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
#include "PerlinNoiseSampler.h"

#include <nuklear.h>


class VariableManager;

//! Height map that describes the terrain of 3D scenes.
/*!
	Height map that describes the terrain of 3D scenes.
	Note that a more appropriate name would be Terrain for this class, this will be refactored in the future.
	Loads and renders the height map/terrain. We use 16-bit grayscale (png) images to generate the terrain. 
	It is also used as an obstacle in the LBM 3D simulation when the y coordinate of the height map is compared
	to the lattice node y coordinate.
*/
class HeightMap {
public:

	int width = 1024;		//!< Width of the height map
	int height = 1024;		//!< Height of the height map - IMPORTANT - in the scene, the height of the map is described by the depth of the scene!!!


	int maxIntensity;	//!< Maximum intensity of the height map - at the moment it will always be set to 255 due to .ppm usage
	
	float *data = nullptr;			//!< The height map data array	

	int showNormalsOnly = 0;		//!< Whether to show normal map on the terrain only (visualize normals as texture)
	int normalsShaderMode = 0;		//!< Whether to show texture modified normals or just normals from the VBO

	glm::vec2 terrainHeightRange;	//!< Range between which the terrain vertices may lie (minimum and maximum height of the terrain)
	float texelWorldSize;			//!< Size of one texel in the world unit size (e.g. 10 means that one texel is 10 meters)

	float globalNormalMapMixingRatio;		//!< How much the global normal map changes normals of the terrain
	float globalNormalMapTiling = 10.0f;	//!< How much the global normal map is tiled

	int useGrungeMap = 0;				//!< Whether to use grunge (color) map to break repeating patterns
	Texture *grungeMap = nullptr;		//!< Texture of the grunge map
	float grungeMapMin = 0.0f;			//!< Minimum darkening from the grunge map
	float grungeMapTiling = 8.0f;		//!< How much is the grunge map tiled across the terrain

	float ambientIntensity = 0.05f;		//!< Phong only - intensity of ambient color
	float diffuseIntensity = 0.9f;		//!< Phong only - intensity of diffuse color
	float specularIntensity = 0.75f;	//!< Phong only - intensity of specular color
										//!< Note that in physically realistic model, diffuseIntensity + specularIntensity <= 1.0

	int activeMaterialCount = 0;	//!< Number of active materials used when drawing the terrain


	VariableManager *vars;					//!< VariableManager used by this terrain
	std::string heightMapFilename = "";		//!< Filename of the current heightmap
	PerlinNoiseSampler perlinSampler;		//!< Sampler used to generate random terrains

	//! Possible ways of generating the terrain.
	enum eDataGenerationMode {
		HEIGHT_MAP = 0,		//!< Use a heightmap
		RANDOM_PERLIN,		//!< Generate random terrain using perlin noise
		_NUM_MODES			//!< Number of terrain generation modes
	};

	int dataGenerationMode = 0;	//!< Selected terrain data generation mode
	int terrainSeed = 0;		//!< Seed used when generating random terrain


	Material materials[MAX_TERRAIN_MATERIALS];	//!< List of Phong materials used by the terrain
	PBRMaterial pbrMaterials[MAX_TERRAIN_PBR_MATERIALS];	//!< List of PBR materials used by the terrain

	Texture *terrainNormalMap;		//!< Normal map for the terrain
	Texture *materialMap;			//!< Material (index) map for the terrain
	CDFSamplerMultiChannel *materialMapSampler = nullptr;	//!< Material (index) map sampler

	int visualizeTextureMode = 0;		//!< Whether the terrain should be drawn with a given visualization texture
	Texture *visTexture = nullptr;		//!< Texture to be shown on the terrain when visualizeTextureMode is active

	int visible = 1;		//!< Whether the terrain is visible in the 3D viewport


	ShaderProgram *shader;	//!< Shader that is used for terrain rendering

	//! Loads and initializes the HeightMap / terrain from the configuration scene file.
	/*!
		\param[in] vars		VariableManager which contains the scene filename.
		\see initBuffers()
		\see loadHeightMapData()
		\see createAndUploadMesh()
		\see initMaterials()
	*/
	HeightMap(VariableManager *vars);


	//! Deletes the height map data.
	~HeightMap();

	//! Applies 3x3 blur kernel to the heightmap.
	/*!
		This was used for smoothing the heightmap when there was a problem with loading 16-bit per channel
		images. Nowadays this is not needed.
	*/
	void smoothHeights();

	//! Loads and uploads heightmap/terrain data to OpenGL buffers.
	/*!
		Uses the member dataGenerationMode.
		\see loadHeightMapData()
		\see createAndUploadMesh()
	*/
	void loadAndUpload();

	//! Loads and uploads heightmap/terrain data to OpenGL buffers.
	/*!
		\param[in] dataGenerationMode		Mode of generating the heightmap data (from file or randomly generated)
		\see loadHeightMapData()
		\see createAndUploadMesh()
	*/
	void loadAndUpload(int dataGenerationMode);

	//! Loads the heightmap data from a file specified by the member variable heightMapFilename.
	void loadHeightMapData();

	//! Loads the heightmap data from a file.
	/*!
		\param[in] filename		Filename of texture to be loaded as heightmap.
	*/
	void loadHeightMapData(std::string filename);

	//! Generates heightmap data using the perlin noise sampler.
	/*!
		\return True if generation succeeded, false otherwise.
	*/
	bool generateRandomHeightData();

	//! Initializes the material set for terrain rendering.
	/*!
		Supports both Phong and PBR pipeline.
	*/
	void initMaterials();

	//! Initializes the OpenGL buffers for drawing the terrain.
	void initBuffers();

	//! Creates terrain triangular mesh from heightmap data.
	/*!
		Normals are computed using computeNormal() function.
		Tangents and bitangents are also generated.
		Uses a naive approach of generating all triangles.
		Should be updated to generate triangle strips or indexed triangles preferably.
	*/
	void createAndUploadMesh();

	//! --- DEPRECATED --- Old way of generating the terrain mesh from heightmap data.
	/*!
		\deprecated Naive approach that does not generate tangents and bitangents.
	*/
	void initBuffersOld();

	//! Returns a height of the x and z positions (either in world space or in terrain texel space).
	/*!
		If x and z are out of terrain bounds, we clamp the height value to terrain edges.
		This is similar to texture edge clamping.

		\param[in] x				Coordinate x for which we want to get height.
		\param[in] z				Coordinate z for which we want to get height.
		\param[in] worldPosition	Whether x and z are world position coordinates or direct indices to heightmap data.
		\return						World space height (y) value.
	*/
	float getHeight(float x, float z, bool worldPosition = true);

	//! Returns the terrain's width in world units.
	float getWorldWidth();

	//! Returns the terrain's depth in world units.
	float getWorldDepth();

	//! Returns a (random) world position from a given texture coordinate sample.
	/*!
		The point of this function is to provide x and y indices to the texel.
		These indices are then moved by random values in [0, 1] (i.e. moved inside the texel).
		These randomly offset values are then converted to world coordinates and height of their position
		is generated.

		\param[in] sampleCoords		Texture coordinates of the sample.
		\return						Random world position that is inside the texel.
	*/
	glm::vec3 getSampleWorldPosition(glm::ivec2 sampleCoords);

	//! Returns a world position sample on the terrain using the CDF sampler probability texture.
	/*!
		\param[in] sampler		CDFSampler whose probability texture is used.
		\return					World position terrain sample.
	*/
	glm::vec3 getWorldPositionSample(CDFSampler *sampler);

	//! Returns a world position sample for the given CDFSamplerMultiChannel channel.
	/*!
		\param[in] sampler		CDFSamplerMultiChannel whose probability texture channel is used.
		\param[in] channel		Channel for which we want to sample.
		\return					World position terrain sample.
	*/
	glm::vec3 getWorldPositionMultiChannelSample(CDFSamplerMultiChannel *sampler, int channel);

	//! Returns a random world position sample for the given material using the material map as a CDFSamplerMultiChannel.
	/*!
		\param[in] materialIdx		Index of the material for which we want to generate the sample.
		\return						World position terrain sample for the given material.
	*/
	glm::vec3 getWorldPositionMaterialMapSample(int materialIdx);

	//! Returns a random world position sample with uniform probability distribution.
	/*!
		\return		Random world position sample with uniform probability distribution.
	*/
	glm::vec3 getRandomWorldPosition();

	//! Returns string for the current data generation mode.
	/*!
		\return				String representing the data generation mode.
	*/
	const char *getDataGenerationModeString();

	//! Returns string for the given data generation mode.
	/*!
		\param[in] mode		Data generation mode.
		\return				String representing the data generation mode.
	*/
	const char *getDataGenerationModeString(int mode);

	//! Constructs the perlin sampler user interface tab.
	/*!
		\param[in] ctx		Nuklear context for which the tab is generated.
	*/
	void constructPerlinGeneratorUITab(struct nk_context *ctx);

	//! Draws the terrain using its default shader.
	void draw();

	//! Draws the terrain using the given shader.
	void draw(ShaderProgram *shader);

	//! Draws geometry of the terrain using the given shader.
	void drawGeometry(ShaderProgram *shader);

private:

	const int materialMapTextureUnit = 12;		//!< Texture unit used by the material map
	const int normalMapTextureUnit = 13;		//!< Texture unit used by the global normal map
	const int grungeMapTextureUnit = 14;		//!< Texture unit used by the color grunge map

	GLuint VAO;		//!< VAO of the terrain's mesh
	GLuint VBO;		//!< VBO of the terrain's mesh


	int numPoints = 0;	//!< Number of vertices of the terrain

	//! Computes normal from four adjacent heightmap data values.
	/*!
		Uses central differences approximation of partial derivatives to compute tangents.
		Cross product of the approximated tangents is the normal vector.

		\param[in] x	Texel x coordinate.
		\param[in] z	Texel z coordinate.
		\return			Normalized normal vector.
	*/
	glm::vec3 computeNormal(int x, int z);

};

