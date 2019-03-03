#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

class ShaderProgram;
class DirectionalLight;

class EVSMShadowMapper {
public:

	bool useBlurPass = true;
	GLint resolution = 1024;

	float shadowBias = 0.0f;
	float lightBleedReduction = 0.01f;
	float varianceMinLimit = 0.0001f;
	float exponent = 40.0f;

	glm::mat4 lightProjectionMatrix;
	glm::mat4 lightViewMatrix;

	int shadowOnly = 0;

	DirectionalLight *dirLight = nullptr;


	EVSMShadowMapper();
	~EVSMShadowMapper();

	void init();

	void preFirstPass();

	void postFirstPass();

	void preSecondPass(int screenWidth, int screenHeight);

	void postSecondPass();


// let it be public for testing now
//private:

	bool isReady();

	ShaderProgram *firstPassShader;
	ShaderProgram *secondPassShader;
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

};

