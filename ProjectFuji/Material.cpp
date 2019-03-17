#include "Material.h"

#include <sstream>

namespace {
	enum ePrepMatUniformsIndices {
		DIFFUSE = 0,
		SPECULAR,
		NORMAL_MAP,
		SHININESS,
		TILING
	};

	// very simply prepare uniform names instead of building them all the time - it would be nicer to precompute them as some kind of static global variable, but this will suffice for now
	const string preparedMaterialUniforms[4][5] = {
		{ "u_Materials[0].diffuse", "u_Materials[0].specular", "u_Materials[0].normalMap", "u_Materials[0].shininess", "u_Materials[0].tiling" },
		{ "u_Materials[1].diffuse", "u_Materials[1].specular", "u_Materials[1].normalMap", "u_Materials[1].shininess", "u_Materials[1].tiling" },
		{ "u_Materials[2].diffuse", "u_Materials[2].specular", "u_Materials[2].normalMap", "u_Materials[2].shininess", "u_Materials[2].tiling" },
		{ "u_Materials[3].diffuse", "u_Materials[3].specular", "u_Materials[3].normalMap", "u_Materials[3].shininess", "u_Materials[3].tiling" }
	};
}


Material::Material() : Material(nullptr, nullptr, nullptr, 32.0f, 1.0f) {
}

Material::Material(Texture *diffuseTexture, Texture *specularMap, Texture *normalMap, float shininess, float textureTiling) 
 : diffuseTexture(diffuseTexture), specularMap(specularMap), normalMap(normalMap), shininess(shininess), textureTiling(textureTiling) {
	
}

Material::Material(Texture &diffuseTexture, Texture &specularMap, Texture &normalMap, float shininess, float textureTiling)
 : diffuseTexture(&diffuseTexture), specularMap(&specularMap), normalMap(&normalMap), shininess(shininess), textureTiling(textureTiling) {
}

Material::Material(std::vector<Texture*> textures, float shininess, float textureTiling)
 : Material(textures[0], textures[1], textures[2], shininess, textureTiling){
	
}




Material::~Material() {
}

void Material::set(Texture * diffuseTexture, Texture * specularMap, Texture * normalMap, float shininess, float textureTiling) {
	this->diffuseTexture = diffuseTexture;
	this->specularMap = specularMap;
	this->normalMap = normalMap;
	this->shininess = shininess;
	this->textureTiling = textureTiling;
}


// basic usage of material
void Material::use(ShaderProgram &shader) {
	use(&shader);
}

void Material::use(ShaderProgram *shader) {
	shader->setFloat("u_Material.shininess", shininess);
	shader->setFloat("u_Material.tiling", textureTiling);
	if (diffuseTexture) {
		diffuseTexture->use(0);
	}
	if (specularMap) {
		specularMap->use(1);
	}
	if (normalMap) {
		normalMap->use(2);
	}
}


void Material::setTextureUniforms(ShaderProgram * shader) {
	shader->setInt("u_Material.diffuse", 0);
	shader->setInt("u_Material.specular", 1);
	shader->setInt("u_Material.normalMap", 2);
}

void Material::useMultiple(ShaderProgram * shader, int materialIdx) {
	shader->setFloat(preparedMaterialUniforms[materialIdx][SHININESS], shininess);
	shader->setFloat(preparedMaterialUniforms[materialIdx][TILING], textureTiling);
	if (diffuseTexture) {
		diffuseTexture->use(0 + 3 * materialIdx);
	}
	if (specularMap) {
		specularMap->use(1 + 3 * materialIdx);
	}
	if (normalMap) {
		normalMap->use(2 + 3 * materialIdx);
	}
}

void Material::setTextureUniformsMultiple(ShaderProgram * shader, int materialIdx) {
	//stringstream ss;
	//ss << "u_Materials[" << materialIdx << "].";
	//string baseStr = ss.str();
	shader->setInt(preparedMaterialUniforms[materialIdx][DIFFUSE], 0 + 3 * materialIdx);
	shader->setInt(preparedMaterialUniforms[materialIdx][SPECULAR], 1 + 3 * materialIdx);
	shader->setInt(preparedMaterialUniforms[materialIdx][NORMAL_MAP], 2 + 3 * materialIdx);
}