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

	enum ePhaseFunction {
		NONE = 0,
		RAYLEIGH,
		HENYEY_GREENSTEIN,
		DOUBLE_HENYEY_GREENSTEIN,
		SCHLICK,
		CORNETTE_SHANK,
		_NUM_PHASE_FUNCTIONS
	};


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

	int lightBufferResolution = 1024;

	int maxNumSlices = 4096;
	int numSlices = 256;
	int numDisplayedSlices = numSlices;

	int srcLightTexture = 0;

	int shaderSet = 2;

	int batchSize;

	int downSample = 1;
	int imageWidth;
	int imageHeight;

	int compositeResultToFramebuffer = 1;

	int useBlurPass = 0;
	float blurAmount = 1.0f;

	// for easier testing
	int firstPassShaderMode = 0;
	int numFirstPassShaderModes = 2;

	int secondPassShaderMode = 0;
	int numSecondPassShaderModes = 2;

	float shadowAlpha100x = 0.5f;
	float shadowAlpha = shadowAlpha100x * 0.01f;

	int forceHalfVecToFaceCam = 0;
	int showParticleTextureIdx = 0;
	int useAtlasTexture = 0;
	int showParticlesBelowCCL = 0;

	float inversionThreshold = 0.0f;

	ePhaseFunction phaseFunction = ePhaseFunction::HENYEY_GREENSTEIN;
	int multiplyPhaseByShadow = 1;
	float symmetryParameter = 0.5f; // for Henyey-Greenstein phase function (only)
	float symmetryParameter2 = -0.5f;
	float dHenyeyGreensteinInterpolationParameter = 0.5f;

	Texture *spriteTexture;
	Texture *atlasSpriteTexture;

	std::vector<Texture *> spriteTextures;


	ParticleRenderer(VariableManager *vars, ParticleSystem *ps);
	~ParticleRenderer();


	void setLightTextureBorders(DirectionalLight *dirLight);


	void setShaderUniforms(ShaderProgram *shader);
	void draw(ParticleSystem *ps, DirectionalLight *dirLight, Camera *cam);

	void recalcVectors(Camera *cam, DirectionalLight *dirLight);

	glm::vec3 getSortVec();

	void refreshImageBuffer();

	void preSceneRenderImage();
	void postSceneRenderImage();


	void updateShaderSet();

	const char *getPhaseFunctionName(int phaseFunc);



private:

	// Helper members so we do not have to send them through all the functions
	ParticleSystem *ps = nullptr;
	DirectionalLight *dirLight = nullptr;
	Camera *cam = nullptr;

	VariableManager *vars = nullptr;

	ShaderProgram *firstPassShader;
	ShaderProgram *secondPassShader;
	ShaderProgram *passThruShader;
	ShaderProgram *blurShader;

	GLuint imageFramebuffer;
	GLuint imageTexture;
	GLuint imageDepthTexture;

	GLuint lightFramebuffer;
	GLuint lightTexture[2]; // for swapping as in the CUDA samples
	GLuint lightDepthTexture;

	int prevShaderSet = -1; // so shader set update in constructor is performed (the function checks whether we are changing shader set)



	GLuint quadVAO;
	GLuint quadVBO;


	void initFramebuffers();
	void initImageBuffer();
	void initLightBuffers();

	void drawSlices();
	void drawSlice(int i);
	void drawSliceLightView(int i);

	void drawPointSprites(ShaderProgram *shader, int start, int count, bool shadowed);
	void drawPoints(int start, int count, bool sorted);

	void compositeResult();

	void blurLightTexture();


};

