#include "Texture.h"

#include <iostream>
//#include <glad/glad.h>
#include <stb_image.h>

#include "Utils.h"

using namespace std;

Texture::Texture() {
	textureUnit = 0;
	//loadTexture("SP_MissingThumbnail.png");
}

Texture::Texture(unsigned int id, int width, int height, int numChannels, string filename, unsigned int textureUnit) : id(id), textureUnit(textureUnit), width(width), height(height), numChannels(numChannels), filename(filename) {
	if (filename.empty()) {
		filename = to_string(id);
	}
}


Texture::Texture(const char *path, unsigned int textureUnit, bool sRGB, bool clampEdges) : textureUnit(textureUnit) {
	loadTexture(path, sRGB, clampEdges);
}


Texture::~Texture() {
}

bool Texture::loadTexture(const char *path, bool sRGB, bool clampEdges) {
	filename = path;

	stbi_set_flip_vertically_on_load(true);
	unsigned char *data = stbi_load(path, &width, &height, &numChannels, NULL);

	bool retVal = true;
	if (!data) {
		std::cout << "Error loading texture at " << path << std::endl;
		stbi_image_free(data);

		data = stbi_load("textures/missing.png", &width, &height, &numChannels, NULL);

		retVal = false;
	}
	glGenTextures(1, &id);

	glActiveTexture(GL_TEXTURE0 + textureUnit); // activate the texture unit first before binding texture
	glBindTexture(GL_TEXTURE_2D, id);


	GLenum format;
	switch (numChannels) {
		case 1:
			format = GL_RED;
			break;
		case 3:
			format = sRGB ? GL_SRGB : GL_RGB;
			//format = GL_SRGB
			break;
		case 4:
			format = sRGB ? GL_SRGB_ALPHA : GL_RGBA;
			break;
		default:
			format = GL_RGB;
			break;
	}


	// set the texture wrapping/filtering options (on the currently bound texture)
	if (clampEdges) {
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		
	} else {
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}



	glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
					
	stbi_image_free(data);
	return retVal;

}

void Texture::useTexture() {
	//std::cout << "Activating texture: " << (GL_TEXTURE0 + textureUnit) << " texture unit = " << textureUnit << std::endl;
	glActiveTexture(GL_TEXTURE0 + textureUnit);
	glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::use(unsigned int textureUnit) {
	glActiveTexture(GL_TEXTURE0 + textureUnit);
	glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::setWrapOptions(unsigned int wrapS, unsigned int wrapT) {
	if ((wrapS != GL_REPEAT && wrapS != GL_CLAMP_TO_EDGE) || (wrapT != GL_REPEAT && wrapT != GL_CLAMP_TO_EDGE)) {
		std::cerr << "wrapS and wrapT option can only have values: GL_REPEAT or GL_CLAMP_TO_EDGE at this moment!" << std::endl;
		return;
	}
	glBindTexture(GL_TEXTURE_2D, id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
}



std::string getTextureName(const Texture *texture) {
	if (texture) {
		return texture->filename;
	} else {
		return "NONE";
	}
}


