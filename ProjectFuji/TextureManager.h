#pragma once

#include <string>
#include <vector>

#include "VariableManager.h"
#include "Texture.h"


namespace TextureManager {

	namespace {

	}


	bool init(VariableManager *vars);
	bool tearDown();

	Texture *getTexturePtr(std::string filename);
	std::vector<Texture *> getTextureTripletPtrs(std::string diffuseFilename, std::string specularFilename, std::string normalMapFilename);



}
