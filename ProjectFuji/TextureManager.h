#pragma once

#include <string>
#include <vector>

#include "VariableManager.h"
#include "Texture.h"
#include "OverlayTexture.h"


namespace TextureManager {

	namespace {

	}


	bool init(VariableManager *vars);
	bool tearDown();

	Texture *getTexturePtr(std::string filename);
	std::vector<Texture *> getTextureTripletPtrs(std::string diffuseFilename, std::string specularFilename, std::string normalMapFilename);


	void refreshOverlayTextures();
	void drawOverlayTextures();
	void drawOverlayTextures(std::vector<GLuint> textureIds);

	// This means that the TextureManager takes the responsibility of the overlay texture (meaning it will deallocate memory on deconstruction)
	int pushOverlayTexture(OverlayTexture *overlayTexture);
	OverlayTexture *createOverlayTexture(int x, int y, int width, int height, Texture *tex = nullptr);
	OverlayTexture *getOverlayTexture(int idx);

}
