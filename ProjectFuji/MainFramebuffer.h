#pragma once

#include <glad\glad.h>
#include <string>
#include "VariableManager.h"
#include "ShaderProgram.h"
#include "Texture.h"

class MainFramebuffer {
public:


	GLuint framebufferId;
	GLuint multisampledFramebufferId;

	GLuint colorTex;
	GLuint multisampledColorTex;
	GLuint depthTex;
	GLuint multisampledDepthTex; // required to create complete multisampled framebuffer
	//Texture *mainTex = nullptr;
	//Texture *depthTex = nullptr;

	MainFramebuffer(VariableManager *vars);
	~MainFramebuffer();

	void prepareForNextFrame(glm::vec4 clearColor);
	void drawToScreen();
	void drawQuad();
	void blitMultisampledToRegular();

	void init();
	void initBuffers();
	void initQuad();
	void refresh();
	void bind();
	void unbind();

private:

	GLuint activeFramebuffer;

	const std::string colorTexName = "Main framebuffer COLOR";
	const std::string depthTexName = "Main framebuffer DEPTH";

	const int sampleCount = 12;
	const bool useMultisampling = true;


	VariableManager *vars = nullptr;

	ShaderProgram *shader = nullptr;

	GLuint quadVAO;
	GLuint quadVBO;

	void refreshActiveFramebuffer();

};

