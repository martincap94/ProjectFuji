///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       TextureManager.h
* \author     Martin Cap
*
*	TextureManager namespace provides easily accessible management of textures across the application.
*	The TextureManager must be initialized and torn down before and after use, respectively!
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <map>

#include "VariableManager.h"
#include "Texture.h"
#include "OverlayTexture.h"

//! Namespace containing texture management utility functions.
/*!
	Namespace containing texture management utility functions.
	Must be initialized and torn down (destroyed) before and after use, respectively!
*/
namespace TextureManager {

	namespace {

	}

	//! Initializes the TextureManager.
	/*!
		Initializes the TextureManager by creating overlay textures that are used for debugging.
		\param[in] vars		VariableManager to be used by the TextureManager.
	*/
	bool init(VariableManager *vars);

	//! Destroys all allocated data of the TextureManager.
	bool tearDown();



	// This is for custom textures (non-Texture objects) that were created for special purposes such as multi-pass algorithms (e.g. shadow mapping), these are usually an attachment to some framebuffer, it is suggested that you name them (optional argument) since this is the name that will represent them in the UI (otherwise their id (as string) is used) - this is then used to obtain them if you wish
	//void pushCustomTexture(GLuint texId, std::string name = "");

	//! Creates new Texture object from an already existing OpenGL texture.
	/*!	
		Creates new Texture object from the already existing OpenGL texture (identified by its texId).
		This allows users to create textures manually and then push them to the TextureManager where their memory is managed.
		\param[in] texId			OpenGL assigned identification number to the texture.
		\param[in] width			Width of the texture.
		\param[in] height			Height of the texture.
		\param[in] numChannels		Number of channels of the texture.
		\param[in] name				Name to be used in the UserInterface.
		\param[in] textureUnit		Texture unit to be used when Texture's use() is called without specifying the texture unit.
		\return Pointer to the created Texture object.

	*/
	Texture *pushCustomTexture(GLuint texId, int width, int height, int numChannels = 4, string name = "", GLuint textureUnit = 0);

	//! Deletes the texture by its name.
	/*!
		\param[in] name		Name of the texture to be deleted.
		\return True if found and deleted, false otherwise.
	*/
	bool deleteTexture(string name);

	//! Pushes heap allocated (!!!) texture to the TextureManager.
	/*!
		This is useful when we want to track this texture globally, expects textures allocated by new (!!!) 
		and the TextureManager takes responsibility for deallocating it (checks if it still exists though)!

		\param[in] tex	Pointer to the heap allocated texture.
	*/
	void pushTexturePtr(Texture *tex);

	//! Loads the texture with the given filename.
	/*!
		\param[in] filename		Filename of the texture to be loaded.
		\param[in] sRGB			Whether to load the texture in sRGB mode.
		\return					Pointer to the texture.
	*/
	Texture *loadTexture(std::string filename, bool sRGB = false);

	//! Loads a triplet of textures.
	/*!
		Loading of texture triplet is particularly useful for creating materials.
		\param[in] diffuseFilename		Filename of the diffuse texture.
		\param[in] specularFilename		Filename of the specular texture.
		\param[in] nomralMapFilename	Filename of the normal map.
		\return							Vector containing pointers to the textures.
	*/
	std::vector<Texture *> loadTextureTriplet(std::string diffuseFilename, std::string specularFilename, std::string normalMapFilename);

	//! Returns pointer to the textures map.
	/*!
		\return		Pointer to the textures map that stores all the TextureManager textures.
	*/
	std::map<std::string, Texture *> *getTexturesMapPtr();

	//! Refreshes VBOs of all overlay textures.
	void refreshOverlayTextures();

	//! Draws all overlay textures.
	void drawOverlayTextures();

	//! Draws given textures as overlay textures.
	/*!
		\param[in] textureIds	OpenGL IDs of the textures to be drawn.
	*/
	void drawOverlayTextures(std::vector<GLuint> textureIds);

	//! Pushes the custom overlay texture to the TextureManager.
	/*!
		This means that the TextureManager takes the responsibility of the overlay texture. 
		It will deallocate memory on destruction.
		\param[in] overlayTexture	The overlay texture to be pushed to the manager.
		\return						Index of the texture in the manager.
	*/
	int pushOverlayTexture(OverlayTexture *overlayTexture);

	//! Creates new overlay texture with the given parameters.
	/*!
		\param[in] x		Screen x position of the texture.
		\param[in] y		Screen y position of the texture.
		\param[in] width	Width of the texture.
		\param[in] height	Height of the texture.
		\param[in] tex		Optional pointer to texture to be used in the overlay texture.
		\return				Pointer to the created overlay texture.
	*/
	OverlayTexture *createOverlayTexture(int x, int y, int width, int height, Texture *tex = nullptr);

	//! Returns pointer to the overlay texture with the given index.
	/*!
		\param[in] idx		Index of the texture.
		\return				Pointer to the overlay texture if it exists, nullptr otherwise.
	*/
	OverlayTexture *getOverlayTexture(int idx);

	//! Returns the number of available overlay textures.
	/*!
		\return		Number of available overlay textures.
	*/
	int getNumAvailableOverlayTextures();

	//! Returns pointer to the vector of overlay textures managed by this TextureManager.
	/*!
		\return	Pointer to the vector of overlay textures.
	*/
	std::vector<OverlayTexture *> *getOverlayTexturesVectorPtr();

	//! Sets the overlay texture.
	/*!
		\param[in] tex		Texture to be drawn as an overlay texture.
		\param[in] idx		Index of the overlay texture.
	*/
	void setOverlayTexture(Texture *tex, int idx);

}
