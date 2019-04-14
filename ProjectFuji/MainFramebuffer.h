#pragma once

#include <glad\glad.h>
#include <string>
#include "VariableManager.h"
#include "ShaderProgram.h"
#include "Texture.h"

class MainFramebuffer {
public:

	GLuint id = 0;

	GLuint colorTex;
	GLuint depthTex;
	//Texture *mainTex = nullptr;
	//Texture *depthTex = nullptr;

	MainFramebuffer(VariableManager *vars);
	~MainFramebuffer();

	void drawToScreen();

	void init();
	void initBuffers();
	void initQuad();
	void refresh();
	void bind();
	void unbind();

private:

	const std::string colorTexName = "Main framebuffer COLOR";
	const std::string depthTexName = "Main framebuffer DEPTH";

	VariableManager *vars = nullptr;

	ShaderProgram *shader = nullptr;

	GLuint quadVAO;
	GLuint quadVBO;

};

