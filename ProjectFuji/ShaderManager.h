#pragma once

#include <string>
#include <glad\glad.h>
#include <glm\glm.hpp>

class ShaderProgram;

namespace ShaderManager {


	namespace {
		void addShader(std::string sName, std::string vertShader, std::string fragShader);
		void addShader(ShaderProgram *sPtr, std::string sName, GLuint sId);
	}


	bool init();
	bool tearDown();

	ShaderProgram *getShaderPtr(std::string shaderName);
	ShaderProgram *getShaderPtr(GLuint shaderId);

	GLuint getShaderId(std::string shaderName);
	std::string getShaderName(GLuint shaderId);

	void updatePVMMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix, glm::mat4 modelMatrix);
	void updatePVMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix);
	void updateProjectionMatrixUniforms(glm::mat4 projectionMatrix);
	void updateViewMatrixUniforms(glm::mat4 viewMatrix);
	void updateModelMatrixUniforms(glm::mat4 modelMatrix);


}
