///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Texture.h
* \author     Martin Cap
* \date       2018/12/23
*
*  Texture class that provides basic texture functionality using stb_image header file for loading.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>
#include "ShaderProgram.h"

//! Simple texture class.
/*!
	Texture class that provides basic functionality.
	Uses stb_image header file for loading texture files.
*/
class Texture {
public:

	//! Possible material type in the Blinn-Phong shader pipeline.
	enum eTextureMaterialType {
		DIFFUSE,		//!< Diffuse texture type
		SPECULAR,		//!< Specular texture type
		NORMAL_MAP		//!< Normal map texture type
	};

	unsigned int id;			//!< Texture id (for OpenGL)
	unsigned int textureUnit;	//!< Texture unit we want to use the texture in
	int width;					//!< Width of the texture image
	int height;					//!< Height of the texture image
	int numChannels;			//!< Number of channels of the texture image
	std::string filename;		//!< Filename associated with this texture

	//! Default constructor.
	Texture();
	
	//! Constructor that creates the texture object from an existing OpenGL texture, i.e. does not load the texture.
	/*!
		Assumes the user wants to load it manually with specific properties.
		\param[in] id			OpenGL texture id of the already existing texture.
		\param[in] width		Width of the texture - should correspond with the width of the given texture.
		\param[in] height		Height of the texture - should correspond with the height of the given texture.
		\param[in] numChannels	Number of channels the texture uses.
		\param[in] filename		Filename/name associated with the texture to be displayed in UI.
		\param[in] textureUnit	Default texture unit to be used when this texture is bound.
	*/
	Texture(unsigned int id, int width, int height, int numChannels, std::string filename = "", unsigned int textureUnit = 0);

	//! Constructs Texture instance and loads the texture right away.
	/*!
		Constructs Texture instance and loads the texture right away.
		Function loadTexture is called inside the constructor.
		\param[in] path				Path to the texture file.
		\param[in] textureUnit		Texture unit which should be used when the texture is used.
		\param[in] sRGB				Load the texture with the sRGB flag.
		\param[in] clampEdges		Whether to use GL_CLAMP_TO_EDGE or GL_REPEAT.
	*/
	Texture(const char *path, unsigned int textureUnit = 0, bool sRGB = false, bool clampEdges = false);

	//! Default destructor.
	~Texture();


	//! Loads the texture.
	/*!
		Loads the texture using stb_image header library.
		If not loaded, loads pink texture to alert the user visually when this texture is used.

		\param[in] path				Path to the texture file.
		\param[in] sRGB				Whether to load the texture as an sRGB texture.
		\param[in] clampEdges		Whether to use GL_CLAMP_TO_EDGE or GL_REPEAT.
		\return						True if loaded correctly, false otherwise.
	*/
	bool loadTexture(const char *path, bool sRGB = false, bool clampEdges = false);

	//! Activates and binds the texture to the default textureUnit.
	void useTexture();

	//! Activates and binds the texture to the specified textureUnit.
	/*!
		Binds the texture to the specified textureUnit.
		\param[in] textureUnit		Texture unit that should be used.
	*/
	void use(unsigned int textureUnit);

	//! Sets wrapping options.
	/*!
		Sets wrapping options. Only GL_REPEAT AND GL_CLAMP_TO_EDGE are accepted 
		(this is from old framework it should be later updated for much more general usage).
		\param[in] wrapS	Wrap on the S axis.
		\param[in] wrapT	Wrap on the T axis.
	*/
	void setWrapOptions(unsigned int wrapS, unsigned int wrapT);

};

//! Return the name/filename of the texture.
/*!
	Returns "NONE" if given a nullptr!
	
	\param[in] texture	Texture for which we want to know its name.
	\return				Texture name if texture is not a nullptr, "NONE" otherwise.
*/
std::string getTextureName(const Texture *texture);


