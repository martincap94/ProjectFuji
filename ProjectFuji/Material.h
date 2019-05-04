///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Material.h
* \author     Martin Cap
*
*	Describes the Material class that is used in the Blinn-Phong pipeline.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Texture.h"
#include "ShaderProgram.h"
#include <vector>
#include <string>

//! Blinn-Phong based material representation.
class Material {
public:

	Texture *diffuseTexture = nullptr;	//!< Diffuse (color) texture
	Texture *specularMap = nullptr;		//!< Specular map
	Texture *normalMap = nullptr;		//!< Normal map
	float shininess;					//!< The shininess factor (exponential)
	float textureTiling;				//!< Tiling of the material (applies to all texture)


	//! Default constructor that sets textures to nullptrs.
	Material();

	//! Constructs the material using the provided textures and parameters.
	/*!
		\param[in] diffuseTexture	The diffuse texture of the material.
		\param[in] specularMap		The specular map of the material.
		\param[in] normalMap		The normal map of the material.
		\param[in] shininess		Specular shininess factor.
		\param[in] textureTiling	Tiling of all the material textures.
	*/
	Material(Texture *diffuseTexture, Texture *specularMap, Texture *normalMap, float shininess, float textureTiling = 1.0f);

	//! Constructs the material using the provided textures and parameters.
	/*!
		\param[in] diffuseTexture	The diffuse texture of the material.
		\param[in] specularMap		The specular map of the material.
		\param[in] normalMap		The normal map of the material.
		\param[in] shininess		Specular shininess factor.
		\param[in] textureTiling	Tiling of all the material textures.
	*/
	Material(Texture &diffuseTexture, Texture &specularMap, Texture &normalMap, float shininess, float textureTiling = 1.0f);

	//! Constructs the material using the provided vector of textures and parameters.
	/*!
		\param[in] textures			Vector of 3 textures in this order: diffuse, specular, normalMap.
		\param[in] shininess		Specular shininess factor.
		\param[in] textureTiling	Tiling of all the material textures.
	*/
	Material(std::vector<Texture *> textures, float shininess, float textureTiling = 1.0f);

	//! Default destructor.
	~Material();

	//! Sets material textures and properties.
	/*!
		\param[in] diffuseTexture	The diffuse texture of the material.
		\param[in] specularMap		The specular map of the material.
		\param[in] normalMap		The normal map of the material.
		\param[in] shininess		Specular shininess factor.
		\param[in] textureTiling	Tiling of all the material textures.
	*/
	void set(Texture *diffuseTexture, Texture *specularMap, Texture *normalMap, float shininess, float textureTiling = 1.0f);

	//! Use the material by setting the needed uniforms including texture uniforms.
	/*!
		\param[in] shader		Shader with which the material will be drawn.
	*/
	void use(ShaderProgram &shader);

	//! Use the material by setting the needed uniforms including texture uniforms.
	/*!
		\param[in] shader		Shader with which the material will be drawn.
	*/
	void use(ShaderProgram *shader);

	//! Sets texture uniforms for this material.
	/*!
		\param[in] shader		Shader with which the material will be drawn.
	*/
	void setTextureUniforms(ShaderProgram *shader);

	//! Use this material in a multiple-material per object workflow.
	/*!
		Used for terrain only at this moment.
		The material index determines texture unit uniform settings.
		\param[in] shader		Shader with which the material will be drawn
		\param[in] materialIdx	Index of the material in the multi-material workflow.
	*/
	void useMultiple(ShaderProgram *shader, int materialIdx);

	//! Sets texture uniforms for this material in multi-material workflow.
	/*!
		Used for terrain only at this moment.
		The material index determines texture unit uniform settings.
		\param[in] shader		Shader with which the material will be drawn
		\param[in] materialIdx	Index of the material in the multi-material workflow.
	*/
	void setTextureUniformsMultiple(ShaderProgram *shader, int materialIdx);

	//! Returns material texture filename for the specified texture type.
	/*!
		\param[in] texType		Type of the texture (diffuse, specular, normal map).
		\return					Filename of the texture if it exists, "NONE" if it is a nullptr.
	*/
	std::string tryGetTextureFilename(Texture::eTextureMaterialType texType);


};

