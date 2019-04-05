#include "EVSMShadowMapper.h"

#include "DataStructures.h"
#include "ShaderManager.h"
#include "ShaderProgram.h"
#include "DirectionalLight.h"
#include "TextureManager.h"
#include "Utils.h"

#include <limits>


EVSMShadowMapper::EVSMShadowMapper() {
}


EVSMShadowMapper::~EVSMShadowMapper() {
}

void EVSMShadowMapper::init() {
	CHECK_GL_ERRORS();


	glGenTextures(1, &zBufferTexture);
	glGenTextures(1, &depthMapTexture);
	glGenTextures(1, &firstPassBlurTexture);
	glGenTextures(1, &secondPassBlurTexture);

	glGenFramebuffers(1, &depthMapFramebuffer);
	glGenFramebuffers(1, &firstPassBlurFramebuffer);
	glGenFramebuffers(1, &secondPassBlurFramebuffer);

	GLfloat fLargest;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &fLargest);
	float maxFloat = numeric_limits<float>().max();
	float borderColor[] = { maxFloat, maxFloat, maxFloat, maxFloat };


	GLenum status;
	CHECK_GL_ERRORS();


	/////////////////////////////////////////////////////////////////////////////
	// DEPTH MAP TEXTURE AND FRAMEBUFFER
	/////////////////////////////////////////////////////////////////////////////

	glBindTexture(GL_TEXTURE_2D, depthMapTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr);


	glTextureParameteri(depthMapTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(depthMapTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(depthMapTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTextureParameteri(depthMapTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glTextureParameterfv(depthMapTexture, GL_TEXTURE_BORDER_COLOR, borderColor);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);



	glBindTexture(GL_TEXTURE_2D, zBufferTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, resolution, resolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	// comparison mode of the shadow map (for sampler2DShadow)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);


	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, depthMapTexture, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, zBufferTexture, 0);

	CHECK_GL_ERRORS();


	/////////////////////////////////////////////////////////////////////////////
	// First pass blur texture and framebuffer
	/////////////////////////////////////////////////////////////////////////////

	glBindTexture(GL_TEXTURE_2D, firstPassBlurTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);

	glTextureParameterfv(firstPassBlurTexture, GL_TEXTURE_BORDER_COLOR, borderColor);


	glBindFramebuffer(GL_FRAMEBUFFER, firstPassBlurFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, firstPassBlurTexture, 0);

	CHECK_GL_ERRORS();


	/////////////////////////////////////////////////////////////////////////////
	// Second pass blur texture and framebuffer
	/////////////////////////////////////////////////////////////////////////////

	glBindTexture(GL_TEXTURE_2D, secondPassBlurTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution, resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, fLargest);

	glTextureParameterfv(secondPassBlurTexture, GL_TEXTURE_BORDER_COLOR, borderColor);


	glBindFramebuffer(GL_FRAMEBUFFER, secondPassBlurFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, secondPassBlurTexture, 0);


	CHECK_GL_ERRORS();

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

	CHECK_GL_ERRORS();

	GLenum error = glGetError();
	status = glCheckNamedFramebufferStatus(depthMapFramebuffer, GL_FRAMEBUFFER);
	status |= glCheckNamedFramebufferStatus(firstPassBlurFramebuffer, GL_FRAMEBUFFER);
	status |= glCheckNamedFramebufferStatus(secondPassBlurFramebuffer, GL_FRAMEBUFFER);

	if (!((status == GL_FRAMEBUFFER_COMPLETE) && (error == GL_NO_ERROR))) {
		printf("Creation of framebuffers failed, glCheckFramebufferStatus() = 0x%x, glGetError() = 0x%x\n", status, error);
	}


	blurShader = ShaderManager::getShaderPtr("gaussianBlur");


	//firstPassShader = ShaderManager::getShaderPtr("evsm_1st_pass");
	//secondPassShader = ShaderManager::getShaderPtr("evsm_2nd_pass");
	//secondPassShader = ShaderManager::getShaderPtr("terrain");

	firstPassShaders.push_back(ShaderManager::getShaderPtr("evsm_1st_pass"));


	secondPassShaders.push_back(ShaderManager::getShaderPtr("terrain"));
	secondPassShaders.push_back(ShaderManager::getShaderPtr("normals_instanced"));
	secondPassShaders.push_back(ShaderManager::getShaderPtr("normals"));

	TextureManager::pushCustomTexture(depthMapTexture, resolution, resolution, 4, "depthMapTexture");
	TextureManager::pushCustomTexture(firstPassBlurTexture, resolution, resolution, 4, "firstPassBlurTexture");
	TextureManager::pushCustomTexture(secondPassBlurTexture, resolution, resolution, 4, "secondPassBlurTexture");

	/*firstPassShader = ShaderManager::getShaderPtr("vsm_1st_pass");
	secondPassShader = ShaderManager::getShaderPtr("vsm_2nd_pass");*/


/*
	firstPassShader = ShaderManager::getShaderPtr("shadow_mapping_1st_pass");
	secondPassShader = ShaderManager::getShaderPtr("shadow_mapping_2nd_pass");
*/
}

void EVSMShadowMapper::preFirstPass() {
	if (!isReady()) {
		return;
	}
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_FRONT);

	//glCullFace(GL_FRONT);
	glViewport(0, 0, resolution, resolution);
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFramebuffer);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	lightViewMatrix = dirLight->getViewMatrix();
	lightProjectionMatrix = dirLight->getProjectionMatrix();


	for (int i = 0; i < firstPassShaders.size(); i++) {

		GLuint pid = firstPassShaders[i]->id;
		firstPassShaders[i]->use();

		glUniform1i(glGetUniformLocation(pid, "u_PCFMode"), 2);

		glUniform2f(glGetUniformLocation(pid, "u_Exponents"), exponent, exponent);
		glUniformMatrix4fv(glGetUniformLocation(pid, "u_Projection"), 1, GL_FALSE, &lightProjectionMatrix[0][0]);
		glUniformMatrix4fv(glGetUniformLocation(pid, "u_View"), 1, GL_FALSE, &lightViewMatrix[0][0]);
	}


}

void EVSMShadowMapper::postFirstPass() {
	if (!isReady()) {
		return;
	}

	//glCullFace(GL_BACK);

	if (useBlurPass) {
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

	prevLightSpaceMatrix = lightSpaceMatrix;
	lightSpaceMatrix = lightProjectionMatrix * lightViewMatrix;


	//glActiveTextire(GL_)
	if (useBlurPass) {
		glBindTextureUnit(TEXTURE_UNIT_DEPTH_MAP, secondPassBlurTexture); // requires OpenGL 4.5
	} else {
		glBindTextureUnit(TEXTURE_UNIT_DEPTH_MAP, depthMapTexture);
		//glBindTextureUnit(3, depthMapTexture);
	}


	for (int i = 0; i < secondPassShaders.size(); i++) {
		//secondPassShader->use();
		secondPassShaders[i]->use();

		//GLuint pid = secondPassShader->id;
		GLuint pid = secondPassShaders[i]->id;

		secondPassShaders[i]->setBool("u_ShadowOnly", (bool)shadowOnly);

		//secondPassShaders[i]->setInt("u_DepthMapTexture", 10);

		glUniform2f(glGetUniformLocation(pid, "u_Exponents"), exponent, exponent);
		secondPassShaders[i]->setFloat("u_ShadowBias", shadowBias);
		secondPassShaders[i]->setFloat("u_LightBleedReduction", lightBleedReduction);
		//secondPassShader->setFloat("u_VarianceMinLimit", varianceMinLimit);


		glUniformMatrix4fv(glGetUniformLocation(pid, "u_LightSpaceMatrix"), 1, GL_FALSE, &lightSpaceMatrix[0][0]);
		glUniformMatrix4fv(glGetUniformLocation(pid, "u_PrevLightSpaceMatrix"), 1, GL_FALSE, &prevLightSpaceMatrix[0][0]);


		//glUniform1i(glGetUniformLocation(pid, "u_DepthMapTexture"), 0);
		//glUniform1i(glGetUniformLocation(pid, "u_DepthMapTexture"), 3);

		//secondPassShader->setVec3("v_ViewPos", di);
	}



}

void EVSMShadowMapper::postSecondPass() {
	if (!isReady()) {
		return;
	}


}

GLuint EVSMShadowMapper::getDepthMapTextureId() {
	return useBlurPass ? secondPassBlurTexture : depthMapTexture;
}

GLuint EVSMShadowMapper::getZBufferTextureId() {
	return zBufferTexture;
}


bool EVSMShadowMapper::isReady() {
	return (dirLight && blurShader);
}
