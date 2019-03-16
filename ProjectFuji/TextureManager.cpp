#include "TextureManager.h"

#include <iostream>
#include <map>

using namespace std;

namespace TextureManager {

	namespace {

		VariableManager *vars;
		map<string, Texture *> textures;
		vector<OverlayTexture *> overlayTextures;

	}


	bool init(VariableManager * vars) {
		TextureManager::vars = vars;

		// prepare debug overlay textures
		int debugOverlayTexturesRes = 250;
		for (int i = 0; i < 4; i++) {
			overlayTextures.push_back(new OverlayTexture(0, vars->screenHeight - (i + 1) * debugOverlayTexturesRes, debugOverlayTexturesRes, debugOverlayTexturesRes, TextureManager::vars));
		}

		return true;
	}

	bool tearDown() {
		for (const auto &kv : textures) {
			if (kv.second) { // make sure it is valid, even though it should always be
				delete kv.second;
			}
		}
		for (int i = 0; i < overlayTextures.size(); i++) {
			if (overlayTextures[i]) {
				delete overlayTextures[i];
			}
		}
		return true;
	}

	Texture *getTexturePtr(string filename) {
		if (textures.count(filename) == 0) {
			cout << "No texture with filename " << filename << " found! Loading..." << endl;
			textures.insert(make_pair(filename, new Texture(filename.c_str())));
		}
		return textures[filename];
	}

	vector<Texture*> getTextureTripletPtrs(string diffuseFilename, string specularFilename, string normalMapFilename) {
		vector<Texture *> tmp;
		tmp.push_back(getTexturePtr(diffuseFilename));
		tmp.push_back(getTexturePtr(specularFilename));
		tmp.push_back(getTexturePtr(normalMapFilename));
		return tmp;
	}

	void refreshOverlayTextures() {
		for (int i = 0; i < overlayTextures.size(); i++) {
			overlayTextures[i]->refreshVBO();
		}
	}

	void drawOverlayTextures() {
		for (int i = 0; i < overlayTextures.size(); i++) {
			overlayTextures[i]->draw();
		}
	}

	void drawOverlayTextures(std::vector<GLuint> textureIds) {
		int size = (textureIds.size() <= overlayTextures.size() - 1) ? textureIds.size() : overlayTextures.size();
		for (int i = 0; i < size; i++) {
			overlayTextures[i]->draw(textureIds[i]);
		}
	}

	int pushOverlayTexture(OverlayTexture * overlayTexture) {
		overlayTextures.push_back(overlayTexture);
		return overlayTextures.size() - 1;
	}

	OverlayTexture * createOverlayTexture(int x, int y, int width, int height, Texture * tex) {
		overlayTextures.push_back(new OverlayTexture(x, y, width, height, vars, tex));
		return overlayTextures.back();
	}

	OverlayTexture * getOverlayTexture(int idx) {
		if (idx < overlayTextures.size()) {
			return overlayTextures[idx];
		}
		return nullptr;
	} 



}