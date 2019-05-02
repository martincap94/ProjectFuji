///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       EVSMShadowMapper.h
* \author     Martin Cap
*
*	Header file containing helper functions for loading cubemap textures for skybox creation.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include <vector>

class ShaderProgram;
class DirectionalLight;
class VariableManager;

class EVSMShadowMapper {
public:

	bool useBlurPass = true;
	const GLint resolution = 1024;

	float shadowBias = 0.001f;
	float lightBleedReduction = 0.2f;
	float varianceMinLimit = 0.0001f;
	float exponent = 42.0f;
	float shadowIntensity = 0.95f;

	glm::mat4 lightProjectionMatrix;
	glm::mat4 lightViewMatrix;

	glm::mat4 lightSpaceMatrix;
	glm::mat4 prevLightSpaceMatrix;

	int shadowOnly = 0;


	EVSMShadowMapper(VariableManager *vars, DirectionalLight *dirLight);
	~EVSMShadowMapper();

	void init();

	void preFirstPass();
	void postFirstPass();
	void preSecondPass();
	void postSecondPass();


	GLuint getDepthMapTextureId();
	GLuint getZBufferTextureId();


// let it be public for testing now
//private:

	bool isReady();

	std::vector<ShaderProgram *> firstPassShaders;
	std::vector<ShaderProgram *> secondPassShaders;

	ShaderProgram *blurShader;

	// initial depth map generation
	GLuint depthMapTexture;
	GLuint zBufferTexture;
	GLuint depthMapFramebuffer;

	GLuint firstPassBlurTexture;
	GLuint firstPassBlurFramebuffer;

	GLuint secondPassBlurTexture;
	GLuint secondPassBlurFramebuffer;



	GLuint quadVAO;
	GLuint quadVBO;

private:

	VariableManager *vars = nullptr;
	DirectionalLight *dirLight = nullptr;



};

