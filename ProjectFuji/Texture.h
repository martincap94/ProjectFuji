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

