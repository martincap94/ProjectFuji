#include "PBRMaterial.h"

using namespace std;

namespace {
	enum ePrepMatUniformsIndices {
		ALBEDO = 0,
		METALLIC_ROUGHNESS,
		NORMAL_MAP,
		AMBIENT_OCCLUSION,
		TILING
	};

	// very simply prepare uniform names instead of building them all the time - it would be nicer to precompute them as some kind of static global variable, but this will suffice for now
	const string preparedMaterialUniforms[4][5] = {
		{ "u_Materials[0].albedo", "u_Materials[0].metallicRoughness", "u_Materials[0].normalMap", "u_Materials[0].ao", "u_Materials[0].tiling" },
		{ "u_Materials[1].albedo", "u_Materials[1].metallicRoughness", "u_Materials[1].normalMap", "u_Materials[1].ao", "u_Materials[1].tiling" },
		{ "u_Materials[2].albedo", "u_Materials[2].metallicRoughness", "u_Materials[2].normalMap", "u_Materials[2].ao", "u_Materials[2].tiling" },
		{ "u_Materials[3].albedo", "u_Materials[3].metallicRoughness", "u_Materials[3].normalMap", "u_Materials[3].ao", "u_Materials[3].tiling" }
	};
}



PBRMaterial::PBRMaterial() {
}

PBRMaterial::PBRMaterial(Texture *albedo, Texture *metallicRoughness, Texture *normalMap, Texture *ao, float textureTiling) :
albedo(albedo), metallicRoughness(metallicRoughness), normalMap(normalMap), ao(ao) {}


PBRMaterial::~PBRMaterial() {
}

void PBRMaterial::use(ShaderProgram *shader) {
	setTextureUniforms(shader);
	if (albedo) {
		albedo->use(0);
	}
	if (metallicRoughness) {
		metallicRoughness->use(1);
	}
	if (normalMap) {
		normalMap->use(2);
	}
	if (ao) {
		ao->use(3);
	}
}

void PBRMaterial::useMultiple(ShaderProgram *shader, int materialIdx) {
	setTextureUniformsMultiple(shader, materialIdx);
	if (albedo) {
		albedo->use(0 + 4 * materialIdx);
	}
	if (metallicRoughness) {
		metallicRoughness->use(1 + 4 * materialIdx);
	}
	if (normalMap) {
		normalMap->use(2 + 4 * materialIdx);
	}
	if (ao) {
		ao->use(3 + 4 * materialIdx);
	}
}

void PBRMaterial::setTextureUniforms(ShaderProgram *shader) {
	shader->use();
	shader->setFloat("u_Material.tiling", textureTiling);
	shader->setInt("u_Material.albedo", 0);
	shader->setInt("u_Material.metallicRoughness", 1);
	shader->setInt("u_Material.normalMap", 2);
	shader->setInt("u_Material.ao", 3);
}

void PBRMaterial::setTextureUniformsMultiple(ShaderProgram *shader, int materialIdx) {
	shader->use();
	shader->setFloat(preparedMaterialUniforms[materialIdx][TILING], textureTiling);
	shader->setInt(preparedMaterialUniforms[materialIdx][ALBEDO], 0 + 4 * materialIdx);
	shader->setInt(preparedMaterialUniforms[materialIdx][METALLIC_ROUGHNESS], 1 + 4 * materialIdx);
	shader->setInt(preparedMaterialUniforms[materialIdx][NORMAL_MAP], 2 + 4 * materialIdx);
	shader->setInt(preparedMaterialUniforms[materialIdx][AMBIENT_OCCLUSION], 3 + 4 * materialIdx);

}
