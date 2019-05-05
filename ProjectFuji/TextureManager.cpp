#include "TextureManager.h"

#include <iostream>

using namespace std;

namespace TextureManager {

	namespace {

		VariableManager *vars;
		map<string, Texture *> textures;
		//map<string, GLuint> customTextures;
		vector<OverlayTexture *> overlayTextures;

	}


	bool init(VariableManager * vars) {
		TextureManager::vars = vars;

		// prepare debug overlay textures
		for (int i = 0; i < vars->numDebugOverlayTextures; i++) {
			overlayTextures.push_back(new OverlayTexture((int)vars->leftSidebarWidth + 20 + i * vars->debugOverlayTextureRes, 0/*vars->screenHeight - (i + 1) * debugOverlayTexturesRes*/, vars->debugOverlayTextureRes, vars->debugOverlayTextureRes, TextureManager::vars));
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

	Texture *pushCustomTexture(GLuint texId, int width, int height, int numChannels, string name, GLuint textureUnit) {
		Texture *tex = new Texture(texId, width, height, numChannels, name, textureUnit);
		textures.insert(make_pair(tex->filename, tex));
		return tex;
	}

	bool deleteTexture(string name) {
		if (textures.count(name) != 0) {
			delete textures[name];
			textures.erase(name);
			return true;
		}
		return false;
	}


	void pushTexturePtr(Texture * tex) {
		textures.insert(make_pair(tex->filename, tex));
	}

	Texture *loadTexture(std::string filename, bool sRGB) {
		if (textures.count(filename) == 0) {
			cout << "Loading texture " << filename << "..." << endl;
			textures.insert(make_pair(filename, new Texture(filename.c_str(), 0, sRGB)));
		}
		return textures[filename];
	}

	vector<Texture*> loadTextureTriplet(string diffuseFilename, string specularFilename, string normalMapFilename) {
		vector<Texture *> tmp;
		tmp.push_back(loadTexture(diffuseFilename));
		tmp.push_back(loadTexture(specularFilename));
		tmp.push_back(loadTexture(normalMapFilename));
		return tmp;
	}

	std::map<std::string, Texture *> *getTexturesMapPtr() {
		return &textures;
	}

	void refreshOverlayTextures() {
		for (int i = 0; i < overlayTextures.size(); i++) {
			overlayTextures[i]->refreshVBO();
		}
	}

	void drawOverlayTextures() {
		if (vars->hideUI) {
			return;
		}
		//cout << "Drawing overlay textures" << endl;
		for (int i = 0; i < overlayTextures.size(); i++) {
			overlayTextures[i]->draw();
		}
	}

	// old
	void drawOverlayTextures(std::vector<GLuint> textureIds) {
		if (vars->hideUI) {
			return;
		}
		int size = (int)((textureIds.size() <= overlayTextures.size() - 1) ? textureIds.size() : overlayTextures.size());
		for (int i = 0; i < size; i++) {
			overlayTextures[i]->draw(textureIds[i]);
			overlayTextures[i]->texId = textureIds[i];
		}
	}

	int pushOverlayTexture(OverlayTexture * overlayTexture) {
		overlayTextures.push_back(overlayTexture);
		return (int)overlayTextures.size() - 1;
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

	int getNumAvailableOverlayTextures() {
		return (int)overlayTextures.size();
	}

	std::vector<OverlayTexture*>* getOverlayTexturesVectorPtr() {
		return &overlayTextures;
	}

	void setOverlayTexture(Texture * tex, int idx) {
		if (idx < overlayTextures.size()) {
			overlayTextures[idx]->texture = tex;
		}
	}



}