///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Cubemap.h
* \author     Martin Cap
*
*	Header file containing helper functions for loading cubemap textures for skybox creation.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <iostream>

#include "Utils.h"
#include "stb_image.h"

//! Loads the cubemap from the given list of face textures.
/*!
	Loads the cubemap from the given list of face textures. Regular ordering is assumed.
	\param[in] faces	Filenames of individual face textures.
	\return OpenGL id of the loaded cubemap texture.
*/
unsigned int loadCubemap(std::vector<std::string> faces) {
    
    unsigned int textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
    
    int width;
    int height;
    int numChannels;
    
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &numChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
            
            
        } else {
            std::cout << "Cubemap texture failed to load!" << std::endl;
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        
    }
    return textureID;
    
}
