#include "OverlayTexture.h"

#include "Utils.h"


OverlayTexture::OverlayTexture(VariableManager *vars, Texture *texture) : vars(vars), texture(texture) {
	shader = ShaderManager::getShaderPtr(shaderName);
	shader->use();
	shader->setInt("u_Texture", 0);
	initBuffers();
	//refreshVBO(); // we expect the user to set the attributes later and refresh the VBO then
}

OverlayTexture::OverlayTexture(int x, int y, int width, int height, VariableManager * vars, Texture * texture) : x(x), y(y), width(width), height(height), vars(vars), texture(texture) {
	shader = ShaderManager::getShaderPtr(shaderName);
	shader->use();
	shader->setInt("u_Texture", 0);

	initBuffers();
	refreshVBO();
}

OverlayTexture::OverlayTexture() {
}

OverlayTexture::~OverlayTexture() {
}


void OverlayTexture::draw() {
	if (!active) {
		return;
	}
	if (texture) {
		//cout << "Texture set in OverlayTexture, drawing..." << endl;
		glBindTextureUnit(0, texture->id);
		drawQuad();
	} else if (texId != -1) {
		glBindTextureUnit(0, texId);
		drawQuad();
	} else {
		//cout << "Texture in overlay texture is nullptr or -1!" << endl;
	}
	
}

void OverlayTexture::draw(Texture &tex) {
	if (!active) {
		return;
	}
	glBindTextureUnit(0, tex.id);
	drawQuad();
}

void OverlayTexture::draw(GLuint textureId) {
	if (!active) {
		return;
	}
	glBindTextureUnit(0, textureId);
	drawQuad();
}

void OverlayTexture::setWidth(int width) {
	this->width = width;
	refreshVBO();
}

void OverlayTexture::setHeight(int height) {
	this->height = height;
	refreshVBO();

}

void OverlayTexture::setX(int x) {
	this->x = x;
	refreshVBO();

}

void OverlayTexture::setY(int y) {
	this->y = y;
	refreshVBO();

}

void OverlayTexture::setPosition(int x, int y) {
	this->x = x;
	this->y = y;
	refreshVBO();
}

void OverlayTexture::setDimensions(int width, int height) {
	this->width = width;
	this->height = height;
	refreshVBO();
}

void OverlayTexture::setAttributes(int x, int y, int width, int height) {
	this->x = x;
	this->y = y;
	this->width = width;
	this->height = height;
	refreshVBO();
}


int OverlayTexture::getWidth() {
	return width;
}

int OverlayTexture::getHeight() {
	return height;
}

int OverlayTexture::getX() {
	return x;
}

int OverlayTexture::getY() {
	return y;
}

std::string OverlayTexture::getBoundTextureName() {
	if (texture) {
		return texture->filename;
	} else if (texId != -1) {
		return to_string(texId);
	}
	return "NONE";
}

// Coordinate computation taken from PGR2 framework by David Ambroz and Petr Felkel
void OverlayTexture::refreshVBO() {

	glm::vec4 vp;
	glGetFloatv(GL_VIEWPORT, &vp.x);
	CHECK_GL_ERRORS();
	//vp.x = 0.0f;
	//vp.y = 0.0f;
	if (vars) {
		vp.z = (float)vars->screenWidth;
		vp.w = (float)vars->screenHeight;
	}

	const GLfloat normalized_coords_with_tex_coords[] = {
		(x - vp.x) / (vp.z - vp.x)*2.0f - 1.0f,          (y - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 0.0f, 0.0f,
		(x + width - vp.x) / (vp.z - vp.x)*2.0f - 1.0f,          (y - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 1.0f, 0.0f,
		(x + width - vp.x) / (vp.z - vp.x)*2.0f - 1.0f, (y + height - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 1.0f, 1.0f,
		(x - vp.x) / (vp.z - vp.x)*2.0f - 1.0f, (y + height - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 0.0f, 1.0f,
	};

	glNamedBufferData(VBO, sizeof(normalized_coords_with_tex_coords), &normalized_coords_with_tex_coords, GL_STATIC_DRAW);

	CHECK_GL_ERRORS();



}

void OverlayTexture::initBuffers() {
	CHECK_GL_ERRORS();


	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	glBindVertexArray(0);
	CHECK_GL_ERRORS();

}

void OverlayTexture::drawQuad() {
	if (!shader) {
		cout << "Shader not found for Overlay Texture!" << endl;
		return;
	}


	GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);

	glDisable(GL_DEPTH_TEST);
	shader->use();
	shader->setBool("u_ShowAlphaChannel", showAlphaChannel != 0);

	//glUniform1i(glGetUniformLocation(shader->id, "u_Texture"), 0);

	glBindVertexArray(VAO);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	if (depth_test_enabled) {
		glEnable(GL_DEPTH_TEST);
	}

}
