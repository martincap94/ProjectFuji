#include "Texture.h"

#include <iostream>
#include <glad/glad.h>
#include <stb_image.h>

Texture::Texture() {
	textureUnit = 0;
	//loadTexture("SP_MissingThumbnail.png");
}


Texture::Texture(const char *path, unsigned int textureUnit, bool clampEdges) : textureUnit(textureUnit) {
	loadTexture(path, clampEdges);
}


Texture::~Texture() {
}

bool Texture::loadTexture(const char *path, bool clampEdges) {
	stbi_set_flip_vertically_on_load(true);
	unsigned char *data = stbi_load(path, &width, &height, &numChannels, NULL);
	if (!data) {
		std::cout << "Error loading texture at " << path << std::endl;
		stbi_image_free(data);
		return false;
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
			format = GL_RGB;
			break;
		case 4:
			format = GL_RGBA;
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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
					
	stbi_image_free(data);
	return true;

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
