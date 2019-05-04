///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       TextRenderer.h
* \date       2019/01/18
*
*	Defines the TextRenderer class to be used to draw text as textures.
*	Uses FreeType library to load TrueType fonts.
*   Taken from: https://learnopengl.com/In-Practice/Text-Rendering
*		author: Joey de Vries
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////


#pragma once

using namespace std;

#include <glad\glad.h>
#include <glm\glm.hpp>

#include <map>
#include "ShaderProgram.h"

//! TextRenderer class that offers text rendering capabilities.
/*!
	Based on Joey de Vries's tutorials: https://learnopengl.com/In-Practice/Text-Rendering
*/
class TextRenderer {
public:

	//! Single character attributes of TrueType font.
	struct Character {
		GLuint     textureID;  //!< ID handle of the glyph texture
		glm::ivec2 size;       //!< Size of glyph
		glm::ivec2 bearing;    //!< Offset from baseline to left/top of glyph
		GLuint     advance;    //!< Offset to advance to next glyph
	};

	map<GLchar, Character> characters;	//!< Preloaded character textures

	GLuint VAO, VBO;

	//! Constructs the text renderer, initializes FreeType and loads arial.
	TextRenderer();

	//! Default destructor.
	~TextRenderer();

	//! Renders the given text in specified location.
	/**
		Renders the given text (string) in specified location (x and y coordinates).
		\param[in] s		ShaderProgram to be used for text rendering.
		\param[in] text		String of text to be rendered.
		\param[in] x		x coordinate of the text.
		\param[in] y		y coordinate of the text.
		\param[in] scale	Scale factor for the text.
		\param[in] color	Color of the text to be set in the shader ("textColor" uniform).
	*/
	void renderText(string text, GLfloat x, GLfloat y, GLfloat scale = 0.0004f, glm::vec3 color = glm::vec3(0.0f));

private:

	ShaderProgram *shader = nullptr;	//!< Shader used when rendering text

};

