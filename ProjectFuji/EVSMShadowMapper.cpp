#include "EVSMShadowMapper.h"

#include "DataStructures.h"
#include "ShaderManager.h"
#include "ShaderProgram.h"
#include "DirectionalLight.h"

EVSMShadowMapper::EVSMShadowMapper() {
}


EVSMShadowMapper::~EVSMShadowMapper() {
}

void EVSMShadowMapper::init() {

	glGenTextures(1, &depthMapTexture);
	glGenTextures(1, &firstPassBlurTexture);
	glGenTextures(1, &secondPassBlurTexture);

	glGenFramebuffers(1, &depthMapFramebuffer);
	glGenFramebuffers(1, &firstPassBlurFramebuffer);
	glGenFramebuffers(1, &secondPassBlurFramebuffer);

	GLfloat fLargest;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &fLargest);
	float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };


	glBindTexture(GL_TEXTURE_2D, depthMapTexture);
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	/////////////////////////////////////////////////////////////////////////////
	// DEPTH MAP TEXTURE AND FRAMEBUFFER
	/////////////////////////////////////////////////////////////////////////////
	glTextureParameteri(depthMapTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(depthMapTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(depthMapTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTextureParameteri(depthMapTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glTextureParameterfv(depthMapTexture, GL_TEXTURE_BORDER_COLOR, borderColor);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);

	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFramebuffer);
	glNamedFramebufferTexture(depthMapFramebuffer, GL_COLOR_ATTACHMENT0, depthMapTexture, 0);


	/////////////////////////////////////////////////////////////////////////////
	// First pass blur texture and framebuffer
	/////////////////////////////////////////////////////////////////////////////

	glGenTextures(1, &firstPassBlurTexture);
	glBindTexture(GL_TEXTURE_2D, firstPassBlurTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);

	glTextureParameterfv(depthMapTexture, GL_TEXTURE_BORDER_COLOR, borderColor);


	glBindFramebuffer(GL_FRAMEBUFFER, firstPassBlurFramebuffer);
	glNamedFramebufferTexture(firstPassBlurFramebuffer, GL_COLOR_ATTACHMENT0, firstPassBlurTexture, 0);



	/////////////////////////////////////////////////////////////////////////////
	// Second pass blur texture and framebuffer
	/////////////////////////////////////////////////////////////////////////////

	glGenTextures(1, &secondPassBlurTexture);
	glBindTexture(GL_TEXTURE_2D, secondPassBlurTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);

	glTextureParameterfv(depthMapTexture, GL_TEXTURE_BORDER_COLOR, borderColor);


	glBindFramebuffer(GL_FRAMEBUFFER, secondPassBlurFramebuffer);
	glNamedFramebufferTexture(secondPassBlurFramebuffer, GL_COLOR_ATTACHMENT0, secondPassBlurTexture, 0);


	// QUAD
	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));


	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	blurShader = ShaderManager::getShaderPtr("gaussianBlur");


	firstPassShader = ShaderManager::getShaderPtr("evsm_1st_pass");
	secondPassShader = ShaderManager::getShaderPtr("evsm_2nd_pass");
	//secondPassShader = ShaderManager::getShaderPtr("dirLightOnly_evsm");


	firstPassShader = ShaderManager::getShaderPtr("vsm_1st_pass");
	secondPassShader = ShaderManager::getShaderPtr("vsm_2nd_pass");


/*
	firstPassShader = ShaderManager::getShaderPtr("shadow_mapping_1st_pass");
	secondPassShader = ShaderManager::getShaderPtr("shadow_mapping_2nd_pass");
*/
}

void EVSMShadowMapper::preFirstPass() {
	if (!isReady()) {
		return;
	}

	//glCullFace(GL_FRONT);
	glViewport(0, 0, resolution, resolution);
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFramebuffer);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	GLuint pid = firstPassShader->id;
	firstPassShader->use();

	lightViewMatrix = dirLight->getViewMatrix();
	lightProjectionMatrix = dirLight->getProjectionMatrix();

	glUniform1i(glGetUniformLocation(pid, "u_PCFMode"), 2);

	glUniform2f(glGetUniformLocation(pid, "u_Exponents"), 40.0f, 40.0f);
	glUniformMatrix4fv(glGetUniformLocation(pid, "u_ProjectionMatrix"), 1, GL_FALSE, &lightProjectionMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(pid, "u_ModelViewMatrix"), 1, GL_FALSE, &lightViewMatrix[0][0]);



}

void EVSMShadowMapper::postFirstPass() {
	if (!isReady() || !useBlurPass) {
		return;
	}

	
	GLuint pid = blurShader->id;
	blurShader->use();

	glBindFramebuffer(GL_FRAMEBUFFER, firstPassBlurFramebuffer);
	glClear(GL_COLOR_BUFFER_BIT);

	glUniform2f(glGetUniformLocation(pid, "u_TexelSize"), 1.0f / resolution, 0.0f);
	glBindTextureUnit(0, depthMapTexture);

	glUniform1i(glGetUniformLocation(pid, "u_InputTexture"), 0);

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	
	glBindFramebuffer(GL_FRAMEBUFFER, secondPassBlurFramebuffer);
	glClear(GL_COLOR_BUFFER_BIT);

	glUniform2f(glGetUniformLocation(pid, "u_TexelSize"), 0.0f, 1.0f / resolution);
	glBindTextureUnit(0, firstPassBlurTexture);

	glUniform1i(glGetUniformLocation(pid, "u_InputTexture"), 0);

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	


}

void EVSMShadowMapper::preSecondPass(int screenWidth, int screenHeight) {
	if (!isReady()) {
		return;
	}

	//glCullFace(GL_BACK);
	glViewport(0, 0, screenWidth, screenHeight);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	glBindTextureUnit(0, secondPassBlurTexture);
	//glBindTextureUnit(0, depthMapTexture);

	secondPassShader->use();

	GLuint pid = secondPassShader->id;

	glUniform2f(glGetUniformLocation(pid, "u_Exponents"), 40.0f, 40.0f);

	glUniform1i(glGetUniformLocation(pid, "u_PCFMode"), 2);


	glm::mat4 lightSpaceMatrix = lightProjectionMatrix * lightViewMatrix;

	glUniformMatrix4fv(glGetUniformLocation(pid, "u_LightSpaceMatrix"), 1, GL_FALSE, &lightSpaceMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(pid, "u_LightViewMatrix"), 1, GL_FALSE, &lightViewMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(pid, "u_LightProjectionMatrix"), 1, GL_FALSE, &lightProjectionMatrix[0][0]);

	glUniform1i(glGetUniformLocation(pid, "u_DepthMapTexture"), 0);



}

void EVSMShadowMapper::postSecondPass() {
	if (!isReady()) {
		return;
	}


}

bool EVSMShadowMapper::isReady() {
	return (dirLight && firstPassShader && blurShader);
}
