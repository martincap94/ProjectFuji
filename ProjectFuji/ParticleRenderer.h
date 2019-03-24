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

	int numSlices = 32;
	int numDisplayedSlices = numSlices;
	int batchSize;

	int downSample = 1;
	int imageWidth;
	int imageHeight;

	float shadowAlpha100x = 0.5f;
	float shadowAlpha = shadowAlpha100x * 0.01f;


	Texture *spriteTexture;


	ParticleRenderer(VariableManager *vars);
	~ParticleRenderer();


	void clearLightFramebuffer(DirectionalLight *dirLight);


	void setShaderUniforms(ShaderProgram *shader);
	void render(ParticleSystem *ps, DirectionalLight *dirLight, Camera *cam);

	void recalcVectors(Camera *cam, DirectionalLight *dirLight);

	glm::vec3 getSortVec();


private:

	// Helper members so we do not have to send them through all the functions
	ParticleSystem *ps = nullptr;
	DirectionalLight *dirLight = nullptr;
	Camera *cam = nullptr;

	VariableManager *vars = nullptr;

	ShaderProgram *firstPassShader;
	ShaderProgram *secondPassShader;

	GLuint imageFramebuffer;
	GLuint imageTexture;
	GLuint imageDepthTexture;

	GLuint lightFramebuffer;
	GLuint lightTexture[2]; // for swapping as in the CUDA samples
	GLuint lightDepthTexture;


	GLuint createTextureHelper(GLenum target, int w, int h, GLint internalFormat, GLenum format);
	void initFramebuffers();

	void drawSlices();
	void drawSlice(int i);
	void drawSliceLightView(int i);

	void drawPointSprites(ShaderProgram *shader, int start, int count, bool shadowed);
	void drawPoints(int start, int count, bool sorted);

	//GLuint 



};

