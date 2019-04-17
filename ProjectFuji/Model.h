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


class Model : public Actor {
public:

	ShaderProgram *shader = nullptr;
	Material *material = nullptr;
	PBRMaterial *pbrMaterial = nullptr;
	//Transform transform;


	Model(const char *path);
	Model(const char *path, Material *material, ShaderProgram *shader = nullptr);

	~Model();

	virtual bool draw();
	virtual bool draw(ShaderProgram *shader);
	virtual bool drawGeometry(ShaderProgram *shader);
	virtual bool drawShadows(ShaderProgram *shader);
	virtual bool drawWireframe(ShaderProgram *shader);

	void makeInstanced(std::vector<Transform> &instanceTransforms);
	void makeInstancedOld(HeightMap *heightMap, int numInstances, glm::vec2 scaleModifier = glm::vec2(1.0f), float maxY = 1000.0f, int maxYTests = 0, glm::vec2 position = glm::vec2(0.0f), glm::vec2 areaSize = glm::vec2(0.0f));

	void makeInstanced(HeightMap *heightMap, int numInstances, glm::vec2 scaleModifier = glm::vec2(1.0f), glm::vec2 position = glm::vec2(0.0f), glm::vec2 areaSize = glm::vec2(0.0f));
	void makeInstancedMaterialMap(HeightMap *heightMap, int numInstances, int materialIdx, glm::vec2 scaleModifier = glm::vec2(1.0f));

	virtual void constructUserInterfaceTab(struct nk_context *ctx, HeightMap *hm = nullptr);


protected:

	bool instanced = false;
	int numInstances = 0;
	glm::vec2 savedInstanceScaleModifier = glm::vec2(1.0f);
	glm::vec2 savedInstanceAreaSize = glm::vec2(0.0f);

	std::vector<Mesh> meshes;

	string directory;

	void useMaterial();
	void useMaterial(ShaderProgram *shader);

	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);

};

