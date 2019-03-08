#include "Model.h"

#include <iostream>

using namespace std;

Model::Model(const char *path) {
	loadModel(path);
}

Model::Model(const char * path, Material * material, ShaderProgram *shader) : material(material), shader(shader) {
	loadModel(path);
}


Model::~Model() {
}

void Model::draw() {
	if (!shader) {
		cerr << "No shader attributed to the model" << endl;
		return;
	}
	shader->use();
	shader->setModelMatrix(transform.getModelMatrix());

	material->diffuseTexture->use(0);
	material->specularMap->use(1);
	material->normalMap->use(2);

	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].draw(*shader);
	}
}

void Model::draw(ShaderProgram & shader) {

	shader.use();
	shader.setModelMatrix(transform.getModelMatrix());

	material->diffuseTexture->useTexture();
	material->specularMap->useTexture();
	material->normalMap->useTexture();

	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].draw(shader);
	}
}

void Model::loadModel(string path) {
	Assimp::Importer import;
	const aiScene *scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_CalcTangentSpace);

	if (!scene || scene->mFlags * AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		cerr << "ASSIMP ERROR: " << import.GetErrorString() << endl;
		return;
	}

	directory = path.substr(0, path.find_last_of('/'));
	processNode(scene->mRootNode, scene);

}

void Model::processNode(aiNode * node, const aiScene * scene) {
	for (int i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene));
	}
	for (int i = 0; i < node->mNumChildren; i++) {
		processNode(node->mChildren[i], scene);
	}
}

Mesh Model::processMesh(aiMesh * mesh, const aiScene * scene) {

	vector<MeshVertex> vertices;
	vector<GLuint> indices;
	vector<Texture> textures;

	for (int i = 0; i < mesh->mNumVertices; i++) {
		MeshVertex vertex;
		glm::vec3 tmp;

		tmp.x = mesh->mVertices[i].x;
		tmp.y = mesh->mVertices[i].y;
		tmp.z = mesh->mVertices[i].z;
		vertex.position = tmp;

		tmp.x = mesh->mNormals[i].x;
		tmp.y = mesh->mNormals[i].y;
		tmp.z = mesh->mNormals[i].z;
		vertex.normal = tmp;

		if (mesh->mTextureCoords[0]) {
			glm::vec2 ttmp;
			ttmp.x = mesh->mTextureCoords[0][i].x;
			ttmp.y = mesh->mTextureCoords[0][i].y;
			vertex.texCoords = ttmp;
		} else {
			vertex.texCoords = glm::vec2(0.0f);
		}
		tmp.x = mesh->mTangents[i].x;
		tmp.y = mesh->mTangents[i].y;
		tmp.z = mesh->mTangents[i].z;
		vertex.tangent = tmp;

		tmp.x = mesh->mBitangents[i].x;
		tmp.y = mesh->mBitangents[i].y;
		tmp.z = mesh->mBitangents[i].z;
		vertex.bitangent = tmp;

		vertices.push_back(vertex);
	}

	for (int i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (int j = 0; j < face.mNumIndices; j++) {
			indices.push_back(face.mIndices[j]);
		}
	}


	return Mesh(vertices, indices, textures);
}

vector<Texture> Model::loadMaterialTextures(aiMaterial * mat, aiTextureType type, string typeName) {
	return vector<Texture>();
}
