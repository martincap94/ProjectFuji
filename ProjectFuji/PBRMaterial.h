///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       PBRMaterial.h
* \author     Martin Cap
*
*	Describes the PBRMaterial class that is used in the PBR pipeline.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Texture.h"
#include "ShaderProgram.h"
#include <vector>
#include <string>

class PBRMaterial {
public:

	Texture *albedo = nullptr;
	Texture *metallicRoughness = nullptr;
	Texture *normalMap = nullptr;
	Texture *ao = nullptr;
	float textureTiling = 1.0f;

	PBRMaterial();
	PBRMaterial(Texture *albedo, Texture *metallicRoughness, Texture *normalMap, Texture *ao, float textureTiling = 1.0f);

	~PBRMaterial();

	void use(ShaderProgram *shader);
	void useMultiple(ShaderProgram *shader, int materialIdx);

	void setTextureUniforms(ShaderProgram *shader);
	void setTextureUniformsMultiple(ShaderProgram *shader, int materialIdx);


};

