///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       ShaderProgram.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines ShaderProgram class for shader program creation and usage.
*
*  Simple ShaderProgram class that helps us with creating, linking and using shader programs.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////


#pragma once

#include <glad\glad.h>
#include <GLFW\glfw3.h>

#include <string>
#include "Config.h"

#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>


/// Simple shader program representation which supports basic operations.
/**
	Simple shader program representation which supports basic creation (only vertex and fragment shaders at this moment).
	It holds the identification number of its shader program and provides basic uniform setting functions.
*/
class ShaderProgram {
public:

	GLuint id;	///< ID of the program

	/// Default constructor.
	ShaderProgram();

	/// Creates and link vertex and fragment shaders.
	/**
		Creates and links vertex and fragment shaders into a shader program.
		Vertex and fragment shaders are read from files.
		\param[in] vsPath	Path to the vertex shader.
		\param[in] fsPath	Path to the fragment shader.
	*/
	ShaderProgram(const GLchar *vsPath, const GLchar *fsPath);

	/// Default destructor.
	~ShaderProgram();

	/// Set 4x4 float matrix uniform.
	void setMat4fv(const string &name, glm::mat4 value) const;

	/// Set vec3 uniform with 3 float values.
	void setVec3(const std::string &name, float x, float y, float z) const {
		glUniform3f(glGetUniformLocation(id, name.c_str()), x, y, z);
	}

	/// Set vec3 uniform with glm::vec3.
	void setVec3(const std::string &name, glm::vec3 value) const {
		glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
	}

	/// Set vec4 unfirom with glm::vec4.
	void setVec4(const std::string &name, glm::vec4 value) const {
		glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
	}
};

