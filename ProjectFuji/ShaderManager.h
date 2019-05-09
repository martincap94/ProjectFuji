///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       ShaderManager.h
* \author     Martin Cap
*
*	Namespace that provides utility functions for shader management across the whole application.
*	Must be initialized and torn down (destroyed) before and after use, respectively!
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <glad\glad.h>
#include <glm\glm.hpp>

#include "ShaderProgram.h"
#include "VariableManager.h"

//! Namespace that acts as a singleton and provides easily accessible management of all shaders for the application.
/*!
	The namespace must be initialized and torn down before and after use, respectively!
	It provides global shader updates that make the process easier, but do not use these excessively since these are
	not very performant.
*/
namespace ShaderManager {


	namespace {
		//! Add a shader to the manager with the given properties.
		/*!
			This is a clunky old function that determines whether the geometry shader is used if the string is empty.

			\param[in] sName		Name given to the shader (its descriptor).
			\param[in] vertShader	Path to the vertex shader file.
			\param[in] fragShader	Path to the fragment shader file.
			\param[in] geomShader	Path to the geometry shader file, if empty, omitted in creation.
			\param[in] lightingType	Lighting type of the materials to be associated with this shader.
			\param[in] matType		Shading type of the materials to be associated with this shader.
		*/
		void addShader(std::string sName, std::string vertShader, std::string fragShader, std::string geomShader = "", ShaderProgram::eLightingType lightingType = ShaderProgram::eLightingType::UNLIT, ShaderProgram::eMaterialType matType = ShaderProgram::eMaterialType::NONE);

		//! Internal shader creation helper.
		/*!
			\param[in] sPtr		Pointer to an already created shader.
			\param[in] sName	Name given to the shader.
			\param[in] sId		OpenGL ID of the shader.
		*/
		void addShader(ShaderProgram *sPtr, std::string sName, GLuint sId);

		//! Initializes all shaders and sets their initial uniform values.
		void initShaders();
	}


	//! Initializes the ShaderManager for future use.
	/*!
		Loads the shaders and initializes them.
		\param[in] vars		VariableManager to be used by this ShaderManager.
		\see loadShaders()
		\see initShaders()
	*/
	bool init(VariableManager *vars = nullptr);

	//! Tears down the manager by freeing all allocated heap memory.
	bool tearDown();

	//! Loads all hardcoded shaders.
	void loadShaders();

	//! Returns a pointer to shader with the given name.
	/*!
		\param[in] shaderName	Name of the shader.
		\return					Pointer to the shader if it exists, nullptr otherwise.
	*/
	ShaderProgram *getShaderPtr(std::string shaderName);

	//! Returns a pointer to shader with the given OpenGL ID.
	/*!
		\param[in] shaderId		OpenGL ID of the shader.
		\return					Pointer to the shader if it exists, nullptr otherwise.
	*/
	ShaderProgram *getShaderPtr(GLuint shaderId);

	//! Returns an OpenGL ID of the shader with the given name.
	/*!
		\param[in] shaderName	Name of the shader.
		\return					OpenGL ID of the shader if it exists, 0 otherwise.
	*/
	GLuint getShaderId(std::string shaderName);

	//! Returns a name of the shader with the given OpenGL ID.
	/*!
		\param[in] shaderId		OpenGL ID of the shader.
		\return					Name of the shader if it exists, nullptr otherwise.
	*/
	std::string getShaderName(GLuint shaderId);

	//! Updates the projection, view and model matrix uniforms of all shaders.
	void updatePVMMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix, glm::mat4 modelMatrix);

	//! Updates projection and view matrix uniforms of all shaders.
	void updatePVMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix);

	//! Updates projection matrix uniforms of all shaders.
	void updateProjectionMatrixUniforms(glm::mat4 projectionMatrix);

	//! Updates view matrix uniforms of all shaders.
	void updateViewMatrixUniforms(glm::mat4 viewMatrix);

	//! Updates model matrix uniforms of all shaders.
	void updateModelMatrixUniforms(glm::mat4 modelMatrix);

	//! Updates directional light uniforms of all lit shaders.
	void updateDirectionalLightUniforms(DirectionalLight &dirLight);

	//! Updates view position uniforms of all shaders.
	void updateViewPositionUniforms(glm::vec3 viewPos);

	//! Updates fog uniforms of all shaders.
	void updateFogUniforms();


}
