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
	GLint resolution = 2048;

	float shadowBias = 0.001f;
	float lightBleedReduction = 0.01f;
	float varianceMinLimit = 0.0001f;
	float exponent = 40.0f;

	glm::mat4 lightProjectionMatrix;
	glm::mat4 lightViewMatrix;

	glm::mat4 lightSpaceMatrix;
	glm::mat4 prevLightSpaceMatrix;

	int shadowOnly = 0;

	DirectionalLight *dirLight = nullptr;


	EVSMShadowMapper();
	~EVSMShadowMapper();

	void init(VariableManager *vars);

	void preFirstPass();

	void postFirstPass();


	void preSecondPass(int screenWidth, int screenHeight);

	void postSecondPass();


	GLuint getDepthMapTextureId();
	GLuint getZBufferTextureId();


// let it be public for testing now
//private:

	bool isReady();

	std::vector<ShaderProgram *> firstPassShaders;
	std::vector<ShaderProgram *> secondPassShaders;

	//ShaderProgram *firstPassShader;
	//ShaderProgram *secondPassShader;
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


};

