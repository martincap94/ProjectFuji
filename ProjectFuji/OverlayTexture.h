#pragma once

#include <string>
#include <glad\glad.h>

#include "Texture.h"
#include "ShaderManager.h"
#include "VariableManager.h"

class OverlayTexture {
public:

	OverlayTexture(VariableManager *vars, Texture *texture = nullptr);
	OverlayTexture(int x, int y, int width, int height, VariableManager *vars, Texture *texture = nullptr);
	~OverlayTexture();

	void draw();
	void draw(Texture &tex);
	void draw(GLuint textureId);

	void setWidth(int width);
	void setHeight(int height);
	void setX(int x);
	void setY(int y);

	void setPosition(int x, int y);
	void setDimensions(int width, int height);

	void setAttributes(int x, int y, int width, int height);

	int getWidth();
	int getHeight();
	int getX();
	int getY();

	void refreshVBO();

private:

	Texture *texture;

	VariableManager *vars;
	ShaderProgram *shader;

	std::string shaderName = "overlayTexture";


	int width;
	int height;
	int x;
	int y;

	GLuint VAO;
	GLuint VBO;

	void initBuffers();

	void drawQuad();



};

