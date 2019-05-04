///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Model.h
* \author     Martin Cap
*
*	Describes a general Model class that is used to load and draw models in our engine.
*	Based on Joey de Vries's tutorials: https://learnopengl.com/Model-Loading/Model
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Actor.h"

#include <vector>
#include <glm\glm.hpp>

#include "ShaderProgram.h"
#include "Mesh.h"
#include "Texture.h"
#include "Material.h"
#include "PBRMaterial.h"
#include "Transform.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "HeightMap.h"

#include <nuklear.h>

//! Simple model that is an Actor in the game world.
/*!
	Based on Joey de Vries's tutorials: https://learnopengl.com/Model-Loading/Model
*/
class Model : public Actor {
public:

	ShaderProgram *shader = nullptr;		//!< Default shader used by the model
	Material *material = nullptr;			//!< Blinn-Phong Material used by the model
	PBRMaterial *pbrMaterial = nullptr;		//!< PBR material used by the model
	//Transform transform;

	//! Constructs a model from the given file.
	Model(const char *path);

	//! Constructs a model from the given file and sets its material and shader.
	Model(const char *path, Material *material, ShaderProgram *shader = nullptr);

	//! Default destructor.
	~Model();

	// Inherited doxygen docs from Actor

	virtual bool draw();
	virtual bool draw(ShaderProgram *shader);
	virtual bool drawGeometry(ShaderProgram *shader);
	virtual bool drawShadows(ShaderProgram *shader);
	virtual bool drawWireframe(ShaderProgram *shader);

	//! Makes the model instanced using the provided array of transforms.
	void makeInstanced(std::vector<Transform> &instanceTransforms);

	//! --- DEPRECATED --- Old way of creating instanced models.
	void makeInstancedOld(HeightMap *heightMap, int numInstances, glm::vec2 scaleModifier = glm::vec2(1.0f), float maxY = 1000.0f, int maxYTests = 0, glm::vec2 position = glm::vec2(0.0f), glm::vec2 areaSize = glm::vec2(0.0f));

	//! Makes the model instanced with random uniform distribution on the provided terrain/heightmap.
	/*!
		\param[in] heightMap		The heightmap to be used for determining instance positions.
		\param[in] numInstances		Number of instances to be generated.
		\param[in] scaleModifier	Range of scales the instances can have.
		\param[in] position			Position of the area in which the instances can be put.
		\param[in] areaSize			Size of the area where the instances can be put (if zeros, uses whole terrain).
	*/
	void makeInstanced(HeightMap *heightMap, int numInstances, glm::vec2 scaleModifier = glm::vec2(1.0f), glm::vec2 position = glm::vec2(0.0f), glm::vec2 areaSize = glm::vec2(0.0f));

	//! Makes the model instanced using the terrain/heightmap's material map sampler.
	/*!
		\param[in] heightMap		The heightmap to be used for determining instance positions.
		\param[in] numInstances		Number of instances to be generated.
		\param[in] materialIdx		Index of the terrain's material that will be used for CDF sampling.
		\param[in] scaleModifier	Range of scales the instances can have.
	*/
	void makeInstancedMaterialMap(HeightMap *heightMap, int numInstances, int materialIdx, glm::vec2 scaleModifier = glm::vec2(1.0f));

	//! Constructs user interface tab for the model.
	virtual void constructUserInterfaceTab(struct nk_context *ctx, HeightMap *hm = nullptr);


protected:

	bool instanced = false;	//!< Whether the model is intanced or not.
	int numInstances = 0;	//!< Number of instances of the model.
	glm::vec2 savedInstanceScaleModifier = glm::vec2(1.0f);	//!< Saved instance scale modifier (if we want to refresh instances for new terrain)
	glm::vec2 savedInstanceAreaSize = glm::vec2(0.0f);		//!< Saved instance area size (if we want to refresh instances for new terrain)

	std::vector<Mesh> meshes;	//!< List of all meshes this model is composed of

	std::string directory;		//!< Directory from which the model was loaded.

	//! Use the material of this model.
	void useMaterial();

	//! Use the material of this model with the given shader.
	void useMaterial(ShaderProgram *shader);

	//! Load the model from the given file ussing Assimp library.
	void loadModel(std::string path);

	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName);

};

