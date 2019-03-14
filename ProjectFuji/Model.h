#pragma once

#include <vector>

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

	void makeInstanced(std::vector<Transform> &instanceTransforms);
	void makeInstanced(HeightMap *heightMap, int numInstances);

protected:

	bool instanced = false;

	std::vector<Mesh> meshes;

	string directory;

	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);

};

