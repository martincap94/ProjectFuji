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

void MainFramebuffer::prepareForNextFrame() {
	refreshActiveFramebuffer();
	bind();
	glViewport(0, 0, vars->screenWidth, vars->screenHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void MainFramebuffer::drawToScreen() {

	if (activeFramebuffer == multisampledFramebufferId) {
		blitMultisampledToRegular();
	}


	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, vars->screenWidth, vars->screenHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);
	
	shader->use();
	shader->setInt("u_Texture", 0);

	glBindTextureUnit(0, colorTex);
	
	drawQuad();
	
	glEnable(GL_DEPTH_TEST);

	CHECK_GL_ERRORS();

}

void MainFramebuffer::drawQuad() {
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void MainFramebuffer::blitMultisampledToRegular() {
	if (useMultisampling) {
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferId);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, multisampledFramebufferId);


		glBlitFramebuffer(0, 0, vars->screenWidth, vars->screenHeight, 0, 0, vars->screenWidth, vars->screenHeight, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);


		//glBindFramebuffer(GL_READ_FRAMEBUFFER, framebufferId);
		//glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferId);

		CHECK_GL_ERRORS();

		activeFramebuffer = framebufferId;
		bind();
	}
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

	glGenFramebuffers(1, &framebufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferId);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorTex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTex, 0);

	if (useMultisampling) {
		//multisampledColorTex = /*createTextureHelper(GL_TEXTURE_2D_MULTISAMPLE, vars->screenWidth, vars->screenHeight, format, GL_RGBA);*/
		glGenTextures(1, &multisampledColorTex);
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, multisampledColorTex);
		glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, sampleCount, format, vars->screenWidth, vars->screenHeight, false);
		

		glGenFramebuffers(1, &multisampledFramebufferId);
		glBindFramebuffer(GL_FRAMEBUFFER, multisampledFramebufferId);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, multisampledColorTex, false);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTex, 0);

		glGenTextures(1, &multisampledDepthTex);
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, multisampledDepthTex);
		glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, sampleCount, GL_DEPTH_COMPONENT, vars->screenWidth, vars->screenHeight, false);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, multisampledDepthTex, 0);

		CHECK_GL_ERRORS();

	}



	TextureManager::pushCustomTexture(colorTex, vars->screenWidth, vars->screenHeight, 4, colorTexName);
	TextureManager::pushCustomTexture(depthTex, vars->screenWidth, vars->screenHeight, 1, depthTexName);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Creation of framebuffers failed, glCheckFramebufferStatus() = 0x%x\n", status);
	}
	CHECK_GL_ERRORS();

	refreshActiveFramebuffer();



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
	if (framebufferId) {
		TextureManager::deleteTexture(colorTexName);
		TextureManager::deleteTexture(depthTexName);
		glDeleteTextures(1, &colorTex);
		glDeleteTextures(1, &depthTex);
		glDeleteFramebuffers(1, &framebufferId);

		if (useMultisampling) {
			glDeleteTextures(1, &multisampledColorTex);
			glDeleteTextures(1, &multisampledDepthTex);
			glDeleteFramebuffers(1, &multisampledFramebufferId);
		}
	}
	initBuffers();
}

void MainFramebuffer::bind() {
	glBindFramebuffer(GL_FRAMEBUFFER, activeFramebuffer);
}

void MainFramebuffer::unbind() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void MainFramebuffer::refreshActiveFramebuffer() {
	activeFramebuffer = useMultisampling ? multisampledFramebufferId : framebufferId;
}
