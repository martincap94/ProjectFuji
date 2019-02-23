///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Texture.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines Texture class for simple texture loading and usage.
*
*  Texture class that provides basic texture functionality using stb_image header file for loading.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>
#include "ShaderProgram.h"

/// Simple texture class.
/**
	Texture class that provides basic functionality.
	Uses stb_image header file for loading texture files.
*/
class Texture {
public:

	unsigned int id;			///< Texture id (for OpenGL)
	unsigned int textureUnit;	///< Texture unit we want to use the texture in
	int width;					///< Width of the texture image
	int height;					///< Height of the texture image
	int numChannels;			///< Number of channels of the texture image

	/// Default constructor.
	Texture();

	/// Constructs Texture instance and loads the texture right away.
	/**
		Constructs Texture instance and loads the texture right away.
		Function loadTexture is called inside the constructor.
		\param[in] path				Path to the texture file.
		\param[in] textureUnit		Texture unit which should be used when the texture is used.
		\param[in] clampEdges		Whether to use GL_CLAMP_TO_EDGE or GL_REPEAT.
	*/
	Texture(const char *path, unsigned int textureUnit, bool clampEdges = false);
	~Texture();


	/// Loads the texture.
	/**
		Loads the texture using stb_image header library.
		\param[in] path				Path to the texture file.
		\param[in] clampEdges		Whether to use GL_CLAMP_TO_EDGE or GL_REPEAT.
	*/
	bool loadTexture(const char *path, bool clampEdges = false);

	/// Activates and binds the texture to the textureUnit.
	void useTexture();

	/// Activates and binds the texture to the specified textureUnit.
	/**
		Binds the texture to the specified textureUnit.
		\param[in] textureUnit		Texture unit that should be used.
	*/
	void use(unsigned int textureUnit);

	/// Sets wrap options.
	/**
		Sets wrap options. Only GL_REPEAT AND GL_CLAMP_TO_EDGE are accepted (this is from old framework
		it should be later updated for much more general usage).
		\param[in] wrapS	Wrap on the S axis.
		\param[in] wrapT	Wrap on the T axis.
	*/
	void setWrapOptions(unsigned int wrapS, unsigned int wrapT);

};

// Inspired by PGR2 framework by David Ambroz and Petr Felkel
void display2DTexture(GLuint textureId, GLuint shaderId, GLint x, GLint y, GLsizei width, GLsizei height);


// Taken from PGR2 framework by David Ambroz and Petr Felkel
inline void Show2DTexture(GLuint tex_id, GLint x, GLint y, GLsizei width, GLsizei height) {
	if (glIsTexture(tex_id) == GL_FALSE) {
		return;
	}

	static const GLenum SHADER_TYPES[] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	// Vertex shader
	static const char* vertex_shader =
		"#version 330 core\n\
                layout (location = 0) in vec4 a_Vertex;\n\
                out vec2 v_TexCoord;\n\
                void main(void) {\n\
                  v_TexCoord  = a_Vertex.zw;\n\
                  gl_Position = vec4(a_Vertex.xy, 0.0, 1.0f);\n\
                }";
	// Fragment shader GL_RGBA
	static const char* fragment_shader_rgba8 =
		"#version 330 core\n\
                layout (location = 0) out vec4 FragColor;\n\
                in vec2 v_TexCoord;\n\
                uniform sampler2D u_Texture;\n\
                void main(void) {\n\
                  FragColor = vec4(texture(u_Texture, v_TexCoord).rgb, 1.0);\n\
                }";
	// Fragment shader GL_R32UI
	static const char* fragment_shader_r32ui =
		"#version 330 core\n\
                layout (location = 0) out vec4 FragColor;\n\
                in vec2 v_TexCoord;\n\
                uniform usampler2D u_Texture;\n\
                void main(void) {\n\
                   uint color = texture(u_Texture, v_TexCoord).r;\n\
                   FragColor = vec4(float(color) * 0.0625, 0.0, 0.0, 1.0);\n\
                }";
	static GLuint s_program_ids[2] = { 0 };

	glm::vec4 vp;
	glGetFloatv(GL_VIEWPORT, &vp.x);
	const GLfloat normalized_coords_with_tex_coords[] = {
		(x - vp.x) / (vp.z - vp.x)*2.0f - 1.0f,          (y - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 0.0f, 0.0f,
		(x + width - vp.x) / (vp.z - vp.x)*2.0f - 1.0f,          (y - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 1.0f, 0.0f,
		(x + width - vp.x) / (vp.z - vp.x)*2.0f - 1.0f, (y + height - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 1.0f, 1.0f,
		(x - vp.x) / (vp.z - vp.x)*2.0f - 1.0f, (y + height - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 0.0f, 1.0f,
	};

	// Setup texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_id);
	GLint tex_format = GL_RGBA8;
	GLint tex_comp_mode = GL_NONE;
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &tex_format);
	glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, &tex_comp_mode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);   // disable compare mode

																		// Compile shaders
	const char* sources[2] = {
		vertex_shader, (tex_format == GL_R32UI) ? fragment_shader_r32ui : fragment_shader_rgba8
	};
	int index = (tex_format == GL_R32UI) ? 1 : 0;
	if (s_program_ids[index] == 0) {
		ShaderProgram *s = new ShaderProgram(sources[0], sources[1]);
		s_program_ids[index] = s->id;
	}
	/*if (s_program_ids[index] == 0) {
		if (!Shader::CreateShaderProgramFromSource(s_program_ids[index], 2, SHADER_TYPES, sources)) {
			fprintf(stderr, "Show2DTexture: Unable to compile shader program.");
			return;
		}
	}*/
	GLint current_program_id = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &current_program_id);
	GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);

	// Render textured screen quad
	glDisable(GL_DEPTH_TEST);
	glUseProgram(s_program_ids[index]);
	glUniform1i(glGetUniformLocation(s_program_ids[index], "u_Texture"), 0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, normalized_coords_with_tex_coords);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	// glDisableVertexAttribArray(0);
	glUseProgram(current_program_id);
	if (depth_test_enabled)
		glEnable(GL_DEPTH_TEST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, tex_comp_mode);   // set original compare mode
}