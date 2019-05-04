///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       OverlayTexture.h
* \author     Martin Cap
*
*	Describes a utility OverlayTexture class that is used to draw overlay textures on the screen.
*	These textures are mainly used for debugging purposes.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <glad\glad.h>

#include "Texture.h"
#include "ShaderManager.h"
#include "VariableManager.h"

//! Texture that can be drawn onto the screen for easier debugging.
/*!
	These are mainly managed by the TextureManager namespace (singleton).
*/
class OverlayTexture {
public:

	int active = 1;		//!< Whether the overlay texture is active (and should therefore be drawn)

	Texture *texture;	//!< Pointer to the Texture object that is to be drawn
	int texId = -1;		//!< OpenGL ID of the texture to be drawn

	int showAlphaChannel = 1;	//!< Whether the alpha channel should be shown/used

	//! Initializes the OverlayTexture by setting its shader and creating necessary buffers.
	/*!
		Does not set any position or size parameters!
		\param[in] vars		VariableManager to be used.
		\param[in] texture	Texture to be drawn as the overlay texture.
	*/
	OverlayTexture(VariableManager *vars, Texture *texture = nullptr);

	//! Initializes the overlay texture by setting its shader and creating necessary buffers.
	/*!
		\param[in] x			Screen x position.
		\param[in] y			Screen y position.
		\param[in] width		Screen width of the overlay texture.
		\param[in] height		Screen height of the overlay texture.
		\param[in] vars		VariableManager to be used.
		\param[in] texture	Texture to be drawn as the overlay texture.
	*/
	OverlayTexture(int x, int y, int width, int height, VariableManager *vars, Texture *texture = nullptr);

	//! Default constructor.
	OverlayTexture();

	//! Default destructor.
	~OverlayTexture();

	//! Draws the overlay texture using the member texture or member texture ID if either of them is set.
	void draw();

	//! Draws the overlay texture using the given Texture reference.
	/*!
		\param[in] tex			Reference to texture that is to be drawn.
	*/
	void draw(Texture &tex);

	//! Draws the overlay texture using the provided OpenGL texture ID.
	/*!
		\param[in] textureId	OpenGL ID of the texture to be drawn.
	*/
	void draw(GLuint textureId);

	//! Sets the width of the overlay texture.
	void setWidth(int width);
	//! Sets the height of the overlay texture.
	void setHeight(int height);
	//! Sets the screen x position of the overlay texture.
	void setX(int x);
	//! Sets the screen y position of the overlay texture.
	void setY(int y);

	//! Sets the screen position (x,y) of the overlay texture.
	void setPosition(int x, int y);
	//! Sets the screen dimensions (width, height) of the overlay texture.
	void setDimensions(int width, int height);
	//! Sets the screen position and screen size attributes of the overlay texture.
	void setAttributes(int x, int y, int width, int height);

	//! Returns the screen width.
	int getWidth();
	//! Returns the screen height.
	int getHeight();
	//! Returns the screen x position.
	int getX();
	//! Returns the screen y position.
	int getY();

	//! Returns the name or texture ID if either the texture pointer or OpenGL texture ID members are set.
	std::string getBoundTextureName();

	//! Refreshes the VBO with current position (x,y) and size (width, height) settings.
	void refreshVBO();

private:


	VariableManager *vars;	//!< VariableManager for this texture
	ShaderProgram *shader;	//!< Shader used to draw overlay textures to screen

	std::string shaderName = "overlayTexture";	//!< Name of the shader to be used


	int width;		//!< Screen width of the overlay texture
	int height;		//!< Screen height of the overlay texture
	int x;			//!< Screen x position of the overlay texture
	int y;			//!< Screen y position of the overlay texture

	GLuint VAO;		//!< VAO of the overlay texture
	GLuint VBO;		//!< VBO of the overlay texture

	//! Initializes the VAO and VBO buffers.
	void initBuffers();

	//! Draws a simple quad using the overlay texture VAO.
	void drawQuad();



};

