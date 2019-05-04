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

//! Simple class describing a PBR material.
class PBRMaterial {
public:

	Texture *albedo = nullptr;				//!< Albedo (base color) of the material
	Texture *metallicRoughness = nullptr;	//!< Metallic texture (RGB) with roughness in alpha channel
	Texture *normalMap = nullptr;			//!< Normal map of the material
	Texture *ao = nullptr;					//!< Ambient occlussion texture
	float textureTiling = 1.0f;				//!< Tiling of all textures

	//! Default constructor.
	PBRMaterial();

	//! Constructs the PBR material from the given textures.
	/*!
		\param[in] albedo				Albedo texture to be used.
		\param[in] metallicRoughness	Metallic (RGB) roughness (A) texture.
		\param[in] normalMap			Normal map.
		\param[in] ao					Ambient occlussion texture.
		\param[in] textureTiling		How much are all the textures tiled.
	*/
	PBRMaterial(Texture *albedo, Texture *metallicRoughness, Texture *normalMap, Texture *ao, float textureTiling = 1.0f);

	//! Default destructor.
	~PBRMaterial();

	//! Use the material by setting the needed uniforms including texture uniforms.
	/*!
		\param[in] shader		Shader with which the material will be drawn.
	*/
	void use(ShaderProgram *shader);

	//! Use this material in a multiple-material per object workflow.
	/*!
		Used for terrain only at this moment.
		The material index determines texture unit uniform settings.
		\param[in] shader		Shader with which the material will be drawn
		\param[in] materialIdx	Index of the material in the multi-material workflow.
	*/
	void useMultiple(ShaderProgram *shader, int materialIdx);

	//! Sets texture uniforms for this material.
	/*!
		\param[in] shader		Shader with which the material will be drawn.
	*/
	void setTextureUniforms(ShaderProgram *shader);


	//! Sets texture uniforms for this material in multi-material workflow.
	/*!
		Used for terrain only at this moment.
		The material index determines texture unit uniform settings.
		\param[in] shader		Shader with which the material will be drawn
		\param[in] materialIdx	Index of the material in the multi-material workflow.
	*/
	void setTextureUniformsMultiple(ShaderProgram *shader, int materialIdx);


};

