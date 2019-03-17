#pragma once

#include "Texture.h"
#include "ShaderProgram.h"
#include <vector>
#include <string>

class Material {
public:

	Texture *diffuseTexture = nullptr;
	Texture *specularMap = nullptr;
	Texture *normalMap = nullptr;
	float shininess;
	float textureTiling;

	Material();
	Material(Texture *diffuseTexture, Texture *specularMap, Texture *normalMap, float shininess, float textureTiling = 1.0f);
	Material(Texture &diffuseTexture, Texture &specularMap, Texture &normalMap, float shininess, float textureTiling = 1.0f);
	Material(std::vector<Texture *> textures, float shininess, float textureTiling = 1.0f);
	~Material();

	void set(Texture *diffuseTexture, Texture *specularMap, Texture *normalMap, float shininess, float textureTiling = 1.0f);
	void use(ShaderProgram &shader);
	void use(ShaderProgram *shader);
	void setTextureUniforms(ShaderProgram *shader);

	void useMultiple(ShaderProgram *shader, int materialIdx);
	void setTextureUniformsMultiple(ShaderProgram *shader, int materialIdx);

	std::string tryGetTextureFilename(Texture::eTextureMaterialType texType);


};

