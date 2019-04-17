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
	virtual bool drawWireframe(ShaderProgram *shader);

	void makeInstanced(std::vector<Transform> &instanceTransforms);
	void makeInstanced(HeightMap *heightMap, int numInstances, glm::vec2 scaleModifier = glm::vec2(1.0f), float maxY = 1000.0f, int maxYTests = 0, glm::vec2 position = glm::vec2(0.0f), glm::vec2 areaSize = glm::vec2(0.0f));
	void makeInstancedMaterialMap(HeightMap *heightMap, int numInstances, int materialIdx, glm::vec2 scaleModifier = glm::vec2(1.0f));

protected:

	bool instanced = false;

	std::vector<Mesh> meshes;

	string directory;

	void useMaterial();
	void useMaterial(ShaderProgram *shader);

	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);

};

