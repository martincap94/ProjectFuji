#include "TextureManager.h"

#include <iostream>
#include <map>

using namespace std;

namespace TextureManager {

	namespace {

		VariableManager *vars;
		map<string, Texture *> textures;

	}


	bool init(VariableManager * vars) {
		TextureManager::vars = vars;
		return true;
	}

	bool tearDown() {
		for (const auto &kv : textures) {
			if (kv.second) { // make sure it is valid, even though it should always be
				delete kv.second;
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



}