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

	void makeInstanced(std::vector<Transform> &instanceTransforms);

protected:

	bool instanced = false;
	int numInstances = 0;


	std::vector<Mesh> meshes;

	string directory;

	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);

};

