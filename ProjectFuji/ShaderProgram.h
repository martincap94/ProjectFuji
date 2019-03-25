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

#include <string>
#include "Config.h"

#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>

#include "DirectionalLight.h"


/// Simple shader program representation which supports basic operations.
/**
	Simple shader program representation which supports basic creation (only vertex and fragment shaders at this moment).
	It holds the identification number of its shader program and provides basic uniform setting functions.
*/
class ShaderProgram {
public:

	enum eMaterialType {
		NONE,
		COLOR,
		PHONG,
		PBR
	};

	enum eLightingType {
		LIT,
		UNLIT
	};

	GLuint id;	///< ID of the program

	eMaterialType matType = NONE;
	eLightingType lightingType = UNLIT;

	/// Default constructor.
	ShaderProgram();

	/// Creates and link vertex and fragment shaders.
	/**
		Creates and links vertex and fragment shaders into a shader program.
		MeshVertex and fragment shaders are read from files.
		\param[in] vsPath	Path to the vertex shader.
		\param[in] fsPath	Path to the fragment shader.
	*/
	ShaderProgram(const GLchar *vsPath, const GLchar *fsPath);

	/// Default destructor.
	~ShaderProgram();

	void use();

	void setBool(const std::string &name, bool value) const;

	void setInt(const std::string &name, int value) const;

	void setFloat(const std::string &name, float value) const;
	/// Set 4x4 float matrix uniform.
	void setMat4fv(const string &name, glm::mat4 value) const;

	/// Set vec3 uniform with 3 float values.
	void setVec3(const std::string &name, float x, float y, float z) const;

	/// Set vec3 uniform with glm::vec3.
	void setVec3(const std::string &name, glm::vec3 value) const;

	/// Set vec4 unfirom with glm::vec4.
	void setVec4(const std::string &name, glm::vec4 value) const;

	void setProjectionMatrix(glm::mat4 projectionMatrix, string uniformName = "u_Projection");

	void setViewMatrix(glm::mat4 viewMatrix, string uniformName = "u_View");

	void setModelMatrix(glm::mat4 modelMatrix, string uniformName = "u_Model");

	void setupMaterialUniforms(bool useShader = true);

	void setFogProperties(float fogIntensity, float fogMinDistance, float fogMaxDistance, glm::vec4 fogColor, int fogMode = 0, float fogExpFalloff = 0.1f, bool useShader = true);


	//void setPointLightAttributes(int lightNum, PointLight &pointLight);

	void updateDirectionalLightUniforms(DirectionalLight &dirLight);

	//void setSpotlightAttributes(Spotlight &spotlight, Camera &camera, bool spotlightOn);


	// FROM PGR2 FRAMEWORK BY AMBROZ
	/*
	GLuint CreateShaderFromSource(GLenum shader_type, const char* source) {
		if (source == NULL)
			return 0;

		switch (shader_type) {
			case GL_VERTEX_SHADER: fprintf(stderr, "vertex shader creation ... "); break;
			case GL_FRAGMENT_SHADER: fprintf(stderr, "fragment shader creation ... "); break;
			case GL_GEOMETRY_SHADER: fprintf(stderr, "geometry shader creation ... "); break;
			case GL_TESS_CONTROL_SHADER: fprintf(stderr, "tesselation control shader creation ... "); break;
			case GL_TESS_EVALUATION_SHADER: fprintf(stderr, "tesselation evaluation shader creation ... "); break;
			default: return 0;
		}

		GLuint shader_id = glCreateShader(shader_type);
		if (shader_id == 0)
			return 0;

		glShaderSource(shader_id, 1, &source, NULL);
		glCompileShader(shader_id);
		if (CheckShaderCompileStatus(shader_id) != GL_TRUE) {
			fprintf(stderr, "failed.\n");
			CheckShaderInfoLog(shader_id);
			glDeleteShader(shader_id);
			return 0;
		} else {
			fprintf(stderr, "successfull.\n");
			return shader_id;
		}
	}


	//-----------------------------------------------------------------------------
	// Name: CreateShaderFromFile()
	// Desc: 
	//-----------------------------------------------------------------------------
	GLuint CreateShaderFromFile(GLenum shader_type, const char* file_name, const char* preprocessor = NULL) {
		char* buffer = Tools::ReadFile(file_name);
		if (buffer == NULL) {
			fprintf(stderr, "Shader creation failed, input file is empty or missing!\n");
			return 0;
		}

		GLuint shader_id = 0;
		if (preprocessor) {
			std::string temp = buffer;
			std::size_t insertIdx = temp.find("\n", temp.find("#version"));
			temp.insert((insertIdx != std::string::npos) ? insertIdx : 0, std::string("\n") + preprocessor + "\n\n");
			shader_id = CreateShaderFromSource(shader_type, temp.c_str());
		} else
			shader_id = CreateShaderFromSource(shader_type, buffer);

		delete[] buffer;
		return shader_id;
	}

	bool CreateShaderProgramFromFile(GLuint& programId, const char* vs, const char* tc,
									 const char* te, const char* gs, const char* fs, const char* preprocessor = NULL, const char *shaderDir = NULL) {
		GLenum shader_types[5] = {
			vs ? GL_VERTEX_SHADER : GL_NONE,
			tc ? GL_TESS_CONTROL_SHADER : GL_NONE,
			te ? GL_TESS_EVALUATION_SHADER : GL_NONE,
			gs ? GL_GEOMETRY_SHADER : GL_NONE,
			fs ? GL_FRAGMENT_SHADER : GL_NONE,
		};
		const char* source_file_names[5] = {
			vs, tc, te, gs, fs
		};

		// Create shader program object
		GLuint pr_id = glCreateProgram();
		for (int i = 0; i < 5; i++) {
			if (source_file_names[i]) {

				std::string tmpFilename = std::string(source_file_names[i]);
				if (shaderDir) {
					tmpFilename = std::string(shaderDir) + "/" + std::string(source_file_names[i]);
				}
				const char *shaderFile = tmpFilename.c_str();
				GLuint shader_id = CreateShaderFromFile(shader_types[i], shaderFile, preprocessor);
				if (shader_id == 0) {
					glDeleteProgram(pr_id);
					return false;
				}
				glAttachShader(pr_id, shader_id);
				glDeleteShader(shader_id);
			}
		}
		glLinkProgram(pr_id);
		if (!CheckProgramLinkStatus(pr_id)) {
			CheckProgramInfoLog(pr_id);
			fprintf(stderr, "Program linking failed!\n");
			glDeleteProgram(pr_id);
			return false;
		}

		// Remove program from OpenGL and update internal list
		glDeleteProgram(programId);
		_updateProgramList(programId, pr_id);
		programId = pr_id;

		return true;
	}
	*/
};

