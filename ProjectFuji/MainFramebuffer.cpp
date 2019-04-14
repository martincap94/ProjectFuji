#include "MainFramebuffer.h"

#include "Utils.h"
#include "DataStructures.h"
#include "TextureManager.h"
#include "ShaderManager.h"

MainFramebuffer::MainFramebuffer(VariableManager *vars) : vars(vars) {
	init();
}



MainFramebuffer::~MainFramebuffer() {
}

void MainFramebuffer::drawToScreen() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, vars->screenWidth, vars->screenHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);
	
	shader->use();
	shader->setInt("u_Texture", 0);
	glBindTextureUnit(0, colorTex);

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	
	glEnable(GL_DEPTH_TEST);

	CHECK_GL_ERRORS();

}

void MainFramebuffer::init() {
	shader = ShaderManager::getShaderPtr("pass_thru");
	initBuffers();
	initQuad();
}

void MainFramebuffer::initBuffers() {

	GLint format = GL_RGBA16F;

	colorTex = createTextureHelper(GL_TEXTURE_2D, vars->screenWidth, vars->screenHeight, format, GL_RGBA);
	depthTex = createTextureHelper(GL_TEXTURE_2D, vars->screenWidth, vars->screenHeight, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glGenFramebuffers(1, &id);
	glBindFramebuffer(GL_FRAMEBUFFER, id);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorTex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTex, 0);

	TextureManager::pushCustomTexture(colorTex, vars->screenWidth, vars->screenHeight, 4, colorTexName);
	TextureManager::pushCustomTexture(depthTex, vars->screenWidth, vars->screenHeight, 1, depthTexName);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Creation of framebuffers failed, glCheckFramebufferStatus() = 0x%x\n", status);
	}
	CHECK_GL_ERRORS();


}

void MainFramebuffer::initQuad() {

	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));


}

void MainFramebuffer::refresh() {
	if (id) {
		TextureManager::deleteTexture(colorTexName);
		TextureManager::deleteTexture(depthTexName);
		glDeleteTextures(1, &colorTex);
		glDeleteTextures(1, &depthTex);
		glDeleteFramebuffers(1, &id);
	}
	initBuffers();
}

void MainFramebuffer::bind() {
	glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void MainFramebuffer::unbind() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
