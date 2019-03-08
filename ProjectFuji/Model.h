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

	void draw();
	void draw(ShaderProgram &shader);

private:

	std::vector<Mesh> meshes;

	string directory;

	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);

};

