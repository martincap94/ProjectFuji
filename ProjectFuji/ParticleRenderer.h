// Particle Renderer based on the sample CUDA project - add proper citations!

#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>


#include "ShaderProgram.h"
#include "Camera.h"
#include "DirectionalLight.h"
#include "VariableManager.h"
#include "ParticleSystem.h"

class ParticleRenderer {
public:

	glm::vec3 lightVec;
	glm::vec3 lightPosEye;
	glm::vec3 viewVec;
	glm::vec3 halfVec;

	glm::mat4 eyeViewMatrix;
	glm::mat4 lightViewMatrix;
	glm::mat4 lightProjectionMatrix;

	glm::mat4 shadowMatrix;

	glm::vec3 eyePos;
	glm::vec3  halfVecEye;

	bool invertedView = false;

	int lightBufferResolution = 512;

	int maxNumSlices = 4096;
	int numSlices = 256;
	int numDisplayedSlices = numSlices;

	int shaderSet = 0;

	int batchSize;

	int downSample = 1;
	int imageWidth;
	int imageHeight;

	int compositeResultToFramebuffer = 1;

	// for easier testing
	int firstPassShaderMode = 0;
	int numFirstPassShaderModes = 2;

	int secondPassShaderMode = 0;
	int numSecondPassShaderModes = 2;

	float shadowAlpha100x = 0.5f;
	float shadowAlpha = shadowAlpha100x * 0.01f;


	Texture *spriteTexture;
	std::vector<Texture *> spriteTextures;


	ParticleRenderer(VariableManager *vars);
	~ParticleRenderer();


	void clearLightFramebuffer(DirectionalLight *dirLight);


	void setShaderUniforms(ShaderProgram *shader);
	void render(ParticleSystem *ps, DirectionalLight *dirLight, Camera *cam);

	void recalcVectors(Camera *cam, DirectionalLight *dirLight);

	glm::vec3 getSortVec();


	void preSceneRenderImage();
	void postSceneRenderImage();


	void updateShaderSet();



private:

	// Helper members so we do not have to send them through all the functions
	ParticleSystem *ps = nullptr;
	DirectionalLight *dirLight = nullptr;
	Camera *cam = nullptr;

	VariableManager *vars = nullptr;

	ShaderProgram *firstPassShader;
	ShaderProgram *secondPassShader;
	ShaderProgram *passThruShader;

	GLuint imageFramebuffer;
	GLuint imageTexture;
	GLuint imageDepthTexture;

	GLuint lightFramebuffer;
	GLuint lightTexture[2]; // for swapping as in the CUDA samples
	GLuint lightDepthTexture;

	int prevShaderSet = shaderSet;



	GLuint quadVAO;
	GLuint quadVBO;


	GLuint createTextureHelper(GLenum target, int w, int h, GLint internalFormat, GLenum format);
	void initFramebuffers();

	void drawSlices();
	void drawSlice(int i);
	void drawSliceLightView(int i);

	void drawPointSprites(ShaderProgram *shader, int start, int count, bool shadowed);
	void drawPoints(int start, int count, bool sorted);

	void compositeResult();

	void switchToExperimentalShaders();
	void switchToDefaultShaders();



};

