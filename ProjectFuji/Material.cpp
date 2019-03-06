#include "Material.h"



Material::Material() {
}

Material::Material(Texture *diffuseTexture, Texture *specularMap, Texture *normalMap, float shininess) 
 : diffuseTexture(diffuseTexture), specularMap(specularMap), normalMap(normalMap), shininess(shininess) {
	
}

Material::Material(Texture &diffuseTexture, Texture &specularMap, Texture &normalMap, float shininess)
 : diffuseTexture(&diffuseTexture), specularMap(&specularMap), normalMap(&normalMap), shininess(shininess) {
}

Material::Material(std::vector<Texture*> textures, float shininess) 
 : Material(textures[0], textures[1], textures[2], shininess){
	
}




Material::~Material() {
}

// Uniforms are supposedly slow for switching off sampler2D params -> use separate shaders?
void Material::use(ShaderProgram &shader) {
	shader.setFloat("material.shininess", shininess);
	if (diffuseTexture != nullptr) {
		diffuseTexture->use(0);
		shader.setBool("material.useDiffuse", true);
	} else {
		shader.setBool("material.useDiffuse", false);
	}
	if (specularMap != nullptr) {
		specularMap->use(1);
		shader.setBool("material.useSpecular", true);
	} else {
		shader.setBool("material.useSpecular", false);
	}
	if (normalMap != nullptr) {
		normalMap->use(2);
		shader.setBool("material.useNormal", true);
	} else {
		shader.setBool("material.useNormal", false);
	}
}
