#pragma once

#include <string>
#include <vector>
#include <map>

#include "VariableManager.h"
#include "Texture.h"
#include "OverlayTexture.h"


namespace TextureManager {

	namespace {

	}


	bool init(VariableManager *vars);
	bool tearDown();



	// This is for custom textures (non-Texture objects) that were created for special purposes such as multi-pass algorithms (e.g. shadow mapping), these are usually an attachment to some framebuffer, it is suggested that you name them (optional argument) since this is the name that will represent them in the UI (otherwise their id (as string) is used) - this is then used to obtain them if you wish
	//void pushCustomTexture(GLuint texId, std::string name = "");

	// Creates new Texture object from the already existing OpenGL texture (identified by its texId) - this allows users to create manually textures and then push them to the TextureManager where their memory allocation is managed
	Texture *pushCustomTexture(GLuint texId, int width, int height, int numChannels = 4, string name = "", GLuint textureUnit = 0);


	// This is useful when we want to track this texture globally, expects textures allocated by new (!!!) and the TextureManager takes responsibility for deallocating it (checks if it still exists though)!
	void pushTexturePtr(Texture *tex);
	Texture *loadTexture(std::string filename, bool sRGB = false);
	Texture *getTexturePtr(std::string filename);
	std::vector<Texture *> getTextureTripletPtrs(std::string diffuseFilename, std::string specularFilename, std::string normalMapFilename);
	std::map<std::string, Texture *> *getTexturesMapPtr();


	void refreshOverlayTextures();
	void drawOverlayTextures();
	void drawOverlayTextures(std::vector<GLuint> textureIds);

	// This means that the TextureManager takes the responsibility of the overlay texture (meaning it will deallocate memory on deconstruction)
	int pushOverlayTexture(OverlayTexture *overlayTexture);
	OverlayTexture *createOverlayTexture(int x, int y, int width, int height, Texture *tex = nullptr);
	OverlayTexture *getOverlayTexture(int idx);

	int getNumAvailableOverlayTextures();
	std::vector<OverlayTexture *> *getOverlayTexturesVectorPtr();

	void setOverlayTexture(Texture *tex, int idx);

}
