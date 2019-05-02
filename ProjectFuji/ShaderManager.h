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

namespace ShaderManager {


	namespace {
		void addShader(std::string sName, std::string vertShader, std::string fragShader, std::string geomShader = "", ShaderProgram::eLightingType lightingType = ShaderProgram::eLightingType::UNLIT, ShaderProgram::eMaterialType matType = ShaderProgram::eMaterialType::NONE);

		void addShader(ShaderProgram *sPtr, std::string sName, GLuint sId);

		void initShaders();
	}


	bool init(VariableManager *vars = nullptr);
	bool tearDown();

	void loadShaders();

	ShaderProgram *getShaderPtr(std::string shaderName);
	ShaderProgram *getShaderPtr(GLuint shaderId);

	GLuint getShaderId(std::string shaderName);
	std::string getShaderName(GLuint shaderId);

	void updatePVMMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix, glm::mat4 modelMatrix);
	void updatePVMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix);
	void updateProjectionMatrixUniforms(glm::mat4 projectionMatrix);
	void updateViewMatrixUniforms(glm::mat4 viewMatrix);
	void updateModelMatrixUniforms(glm::mat4 modelMatrix);

	void updateDirectionalLightUniforms(DirectionalLight &dirLight);

	void updateViewPositionUniforms(glm::vec3 viewPos);

	void updateFogUniforms();


}
