#include "Model.h"

#include <iostream>
#include "Utils.h"

using namespace std;

Model::Model(const char *path) : Actor(path) {
	loadModel(path);
}

Model::Model(const char * path, Material * material, ShaderProgram *shader) : Actor(path), material(material), shader(shader) {
	loadModel(path);
}


Model::~Model() {
}

bool Model::draw() {
	if (!shouldDraw()) {
		return false;
	}

	if (!shader) {
		//cerr << "No shader attributed to the model" << endl;
		return false;
	}
	shader->use();
	shader->setModelMatrix(transform.getSavedModelMatrix());
	useMaterial();

	shader->setInt("u_DepthMapTexture", TEXTURE_UNIT_DEPTH_MAP);


	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].draw(shader);
	}
	return Actor::draw();

}

bool Model::draw(ShaderProgram *shader) {
	if (!shouldDraw()) {
		return false;
	}

	shader->use();
	shader->setModelMatrix(transform.getSavedModelMatrix());
	useMaterial(shader);

	shader->setInt("u_DepthMapTexture", TEXTURE_UNIT_DEPTH_MAP);
	//glBindTextureUnit(3, evsm)


	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].draw(shader);
	}
	return Actor::draw(shader);

}

bool Model::drawGeometry(ShaderProgram * shader) {
	if (!shouldDraw()) {
		return false;
	}


	shader->use();
	shader->setModelMatrix(transform.getSavedModelMatrix());
	shader->setBool("u_IsInstanced", instanced);
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].draw(shader);
	}
	return Actor::drawGeometry(shader);

}

bool Model::drawShadows(ShaderProgram * shader) {
	if (!(shouldDraw() && castShadows)) {
		return false;
	}
	shader->use();
	shader->setModelMatrix(transform.getSavedModelMatrix());
	shader->setBool("u_IsInstanced", instanced);
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].draw(shader);
	}
	return Actor::drawShadows(shader);

}

bool Model::drawWireframe(ShaderProgram * shader) {
	if (!shouldDraw()) {
		return false;
	}


	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	drawGeometry(shader);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	return Actor::drawWireframe(shader);

}

void Model::makeInstanced(std::vector<Transform> &instanceTransforms) {
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].makeInstanced(instanceTransforms);
	}
	this->numInstances = numInstances;
	instanced = true;
}

void Model::makeInstancedOld(HeightMap *heightMap, int numInstances, glm::vec2 scaleModifier, float maxY, int maxYTests, glm::vec2 position, glm::vec2 areaSize) {

	if (areaSize.x == 0.0f || areaSize.y == 0.0f) {
		areaSize.x = heightMap->getWorldWidth();
		areaSize.y = heightMap->getWorldDepth();
	}
	float areaHalfWidth = areaSize.x / 2.0f;
	float areaHalfDepth = areaSize.y / 2.0f;

	std::vector<Transform> instanceTransforms;
	for (int i = 0; i < numInstances; i++) {
		float instanceScaleModifier = getRandFloat(scaleModifier.x, scaleModifier.y);
		
		float xPos;
		float zPos;
		float yPos;
		int counter = 0;

		do {
			xPos = getRandFloat(position.x - areaHalfWidth, position.x + areaHalfWidth);
			zPos = getRandFloat(position.y - areaHalfDepth, position.y + areaHalfDepth);
			yPos = heightMap->getHeight(xPos, zPos);
			counter++;
		} while (yPos > maxY && counter < maxYTests);

		Transform t(glm::vec3(xPos, yPos, zPos), glm::vec3(0.0f, getRandFloat(0.0f, 90.0f), 0.0f), glm::vec3(instanceScaleModifier));
		instanceTransforms.push_back(t);
	}
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].makeInstanced(instanceTransforms);
	}
	savedInstanceAreaSize = areaSize;
	savedInstanceScaleModifier = scaleModifier;
	this->numInstances = numInstances;
	instanced = true;

}

void Model::makeInstanced(HeightMap * heightMap, int numInstances, glm::vec2 scaleModifier, glm::vec2 position, glm::vec2 areaSize) {
	transform.position = glm::vec3(position.x, 0.0f, position.y);

	if (areaSize.x == 0.0f || areaSize.y == 0.0f) {
		areaSize.x = heightMap->getWorldWidth();
		areaSize.y = heightMap->getWorldDepth();
	}
	float areaHalfWidth = areaSize.x / 2.0f;
	float areaHalfDepth = areaSize.y / 2.0f;

	std::vector<Transform> instanceTransforms;
	for (int i = 0; i < numInstances; i++) {
		float instanceScaleModifier = getRandFloat(scaleModifier.x, scaleModifier.y);

		float xOff = getRandFloat(-areaHalfWidth, areaHalfWidth);
		float zOff = getRandFloat(-areaHalfDepth, areaHalfDepth);
		float xPos = transform.position.x + xOff;
		float zPos = transform.position.z + zOff;
		float yPos = heightMap->getHeight(xPos, zPos);

		Transform t(glm::vec3(xOff, yPos, zOff), glm::vec3(0.0f, getRandFloat(0.0f, 90.0f), 0.0f), glm::vec3(instanceScaleModifier));
		instanceTransforms.push_back(t);
	}
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].makeInstanced(instanceTransforms);
	}
	savedInstanceAreaSize = areaSize;
	savedInstanceScaleModifier = scaleModifier;
	this->numInstances = numInstances;
	instanced = true;

}

void Model::makeInstancedMaterialMap(HeightMap * heightMap, int numInstances, int materialIdx, glm::vec2 scaleModifier) {
	std::vector<Transform> instanceTransforms;
	for (int i = 0; i < numInstances; i++) {
		float instanceScaleModifier = getRandFloat(scaleModifier.x, scaleModifier.y);

		glm::vec3 pos = heightMap->getWorldPositionMaterialMapSample(materialIdx);

		int counter = 0;

		Transform t(pos, glm::vec3(0.0f, getRandFloat(0.0f, 90.0f), 0.0f), glm::vec3(instanceScaleModifier));
		instanceTransforms.push_back(t);
	}
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].makeInstanced(instanceTransforms);
	}
	savedInstanceScaleModifier = scaleModifier;
	this->numInstances = numInstances;
	instanced = true;
}

void Model::constructUserInterfaceTab(struct nk_context *ctx, HeightMap *hm) {
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_checkbox_label(ctx, "Cast Shadows", &castShadows);
	if (instanced) {

		if (hm != nullptr) {
			if (nk_button_label(ctx, "Refresh Instances")) {
				makeInstanced(hm, this->numInstances, savedInstanceScaleModifier, glm::vec2(transform.position.x, transform.position.z), savedInstanceAreaSize);
			}
		}
	}
}



void Model::useMaterial() {
	useMaterial(shader);
}

void Model::useMaterial(ShaderProgram *shader) {

	if (shader->matType == ShaderProgram::eMaterialType::PHONG) {
		material->use(shader);
	} else if (shader->matType == ShaderProgram::eMaterialType::PBR) {
		pbrMaterial->use(shader);
	}

	//shader->setInt("u_Material.diffuse", 0);
	//shader->setInt("u_Material.specular", 1);
	//shader->setInt("u_Material.normalMap", 2);
	//shader->setFloat("u_Material.shininess", material->shininess);

	//material->diffuseTexture->use(0);
	//material->specularMap->use(1);
	//material->normalMap->use(2);
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
	for (unsigned int i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene));
	}
	for (unsigned int i = 0; i < node->mNumChildren; i++) {
		processNode(node->mChildren[i], scene);
	}
}

Mesh Model::processMesh(aiMesh * mesh, const aiScene * scene) {

	vector<MeshVertex> vertices;
	vector<GLuint> indices;
	vector<Texture> textures;

	for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
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

	for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j++) {
			indices.push_back(face.mIndices[j]);
		}
	}


	return Mesh(vertices, indices, textures);
}

vector<Texture> Model::loadMaterialTextures(aiMaterial * mat, aiTextureType type, string typeName) {
	return vector<Texture>();
}
