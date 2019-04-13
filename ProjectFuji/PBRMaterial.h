#pragma once

#include "Texture.h"
#include "ShaderProgram.h"
#include <vector>
#include <string>

class PBRMaterial {
public:

	Texture *albedo = nullptr;
	Texture *metallicSmoothness = nullptr;
	Texture *normalMap = nullptr;
	Texture *ao = nullptr;
	float textureTiling = 1.0f;

	PBRMaterial();
	PBRMaterial(Texture *albedo, Texture *metallicSmoothness, Texture *normalMap, Texture *ao, float textureTiling = 1.0f);

	~PBRMaterial();

	void use(ShaderProgram *shader);
	void useMultiple(ShaderProgram *shader, int materialIdx);

	void setTextureUniforms(ShaderProgram *shader);
	void setTextureUniformsMultiple(ShaderProgram *shader, int materialIdx);


};

