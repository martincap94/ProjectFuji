#pragma once

#include <vector>
#include <glm\glm.hpp>

#include "ShaderProgram.h"
#include "Mesh.h"
#include "Texture.h"
#include "Material.h"
#include "Transform.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "HeightMap.h"

class Model {
public:

	ShaderProgram *shader;
	Material *material;
	Transform transform;

	Model(const char *path);
	Model(const char *path, Material *material, ShaderProgram *shader = nullptr);

	~Model();

	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void drawGeometry(ShaderProgram *shader);
	virtual void drawWireframe(ShaderProgram *shader);

	void makeInstanced(std::vector<Transform> &instanceTransforms);
	void makeInstanced(HeightMap *heightMap, int numInstances, glm::vec2 scaleModifier = glm::vec2(0.8, 1.2), float maxY = 1000.0f, int maxYTests = 0);

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

