#pragma once

#include "Texture.h"
#include "ShaderProgram.h"
#include <vector>

class Material {
public:

	Texture *diffuseTexture = nullptr;
	Texture *specularMap = nullptr;
	Texture *normalMap = nullptr;
	float shininess;

	Material();
	Material(Texture *diffuseTexture, Texture *specularMap, Texture *normalMap, float shininess);
	Material(Texture &diffuseTexture, Texture &specularMap, Texture &normalMap, float shininess);
	Material(std::vector<Texture *> textures, float shininess);
	~Material();

	void use(ShaderProgram &shader);
};

