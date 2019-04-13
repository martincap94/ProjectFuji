#include "ParticleRenderer.h"

#include <iostream>
#include <algorithm>

#include "Utils.h"
#include "TextureManager.h"


using namespace std;



ParticleRenderer::ParticleRenderer(VariableManager * vars, ParticleSystem *ps) : vars(vars), ps(ps) {
	initFramebuffers();

	/*
	firstPassShader = ShaderManager::getShaderPtr("volume_1st_pass");
	secondPassShader = ShaderManager::getShaderPtr("volume_2nd_pass");
	*/
	updateShaderSet();


	passThruShader = ShaderManager::getShaderPtr("pass_thru");
	blurShader = ShaderManager::getShaderPtr("blur_basic");

	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "grad.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "testTexture.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "testTexture2.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "cloud_particle_1024.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "cloud_particle_256.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "testTexture3_256.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "testTexture3.png")); // still the best looking I would say
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "testTexture4.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "testTexture5.png"));
	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "testTexture6.png"));


	spriteTexture = spriteTextures.back();

	spriteTextures.push_back(TextureManager::getTexturePtr((string)TEXTURES_DIR + "white.png"));

	atlasSpriteTexture = TextureManager::getTexturePtr((string)TEXTURES_DIR + "cloud_atlas.png");

}

ParticleRenderer::~ParticleRenderer() {
}

void ParticleRenderer::setLightTextureBorders(DirectionalLight * dirLight) {

	GLfloat borderColor[4] = { 1.0f - dirLight->color.x, 1.0f - dirLight->color.y, 1.0f - dirLight->color.z, 0.0f };

	glBindTexture(GL_TEXTURE_2D, lightTexture[0]);
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

	glBindTexture(GL_TEXTURE_2D, lightTexture[1]);
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

	glBindTexture(GL_TEXTURE_2D, 0);


}

void ParticleRenderer::setShaderUniforms(ShaderProgram * shader) {
	shader->use();
	shader->setInt("u_Texture", 0); // just to be sure
	shader->setInt("u_AtlasTexture", 2);
	shader->setBool("u_UseAtlasTexture", (bool)useAtlasTexture);
	shader->setVec3("u_CameraPos", cam->position);
	shader->setVec3("u_LightPos", dirLight->position);
	shader->setFloat("u_WorldPointSize", ps->pointSize);
	shader->setFloat("u_Opacity", vars->opacityMultiplier);
	shader->setVec3("u_TintColor", vars->tintColor);
	shader->setFloat("u_ShadowAlpha", shadowAlpha);
	shader->setBool("u_ShowParticleTextureIdx", (bool)showParticleTextureIdx);
	shader->setInt("u_PhaseFunction", (int)phaseFunction);
	shader->setBool("u_MultiplyPhaseByShadow", (bool)multiplyPhaseByShadow);
	shader->setFloat("u_g", symmetryParameter);
	shader->setFloat("u_g2", symmetryParameter2);
	shader->setFloat("u_f", dHenyeyGreensteinInterpolationParameter);
	shader->setBool("u_ShowParticlesBelowCCL", (bool)showParticlesBelowCCL);

}

void ParticleRenderer::draw(ParticleSystem * ps, DirectionalLight *dirLight, Camera *cam) {

	// Set member variables for later use - must precede all rendering steps
	this->ps = ps;
	this->dirLight = dirLight;
	this->cam = cam;

	shadowAlpha = shadowAlpha100x * 0.01f;


	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glClear(GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, lightFramebuffer);
	glClear(GL_DEPTH_BUFFER_BIT);

	/*
	glBindFramebuffer(GL_FRAMEBUFFER, imageFramebuffer);
	glClear(GL_DEPTH_BUFFER_BIT);
	*/

	setShaderUniforms(firstPassShader);
	// here, we know that the firstPassShader was used
	firstPassShader->use(); // just to be sure, later reorganize
	firstPassShader->setViewMatrix(lightViewMatrix);
	firstPassShader->setProjectionMatrix(lightProjectionMatrix);
	firstPassShader->setInt("u_Mode", firstPassShaderMode);


	setShaderUniforms(secondPassShader);
	secondPassShader->use();
	//secondPassShader->setMat4fv("u_LightSpaceView", lightViewMatrix);
	//secondPassShader->setMat4fv("u_LightSpaceProjection", lightProjectionMatrix);
	glm::mat4 lightSpaceMatrix = lightProjectionMatrix * lightViewMatrix;
	secondPassShader->setMat4fv("u_LightSpaceMatrix", lightSpaceMatrix);
	secondPassShader->setInt("u_Mode", secondPassShaderMode);

	spriteTexture->use(0);
	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, spriteTexture->id);

	atlasSpriteTexture->use(2);
	//glActiveTexture(GL_TEXTURE2);
	//glBindTexture(GL_TEXTURE_2D, atlasSpriteTexture->id);

	drawSlices();


	if (compositeResultToFramebuffer) {
		compositeResult();
	}



	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, vars->screenWidth, vars->screenHeight);
}


void ParticleRenderer::recalcVectors(Camera *cam, DirectionalLight *dirLight) {

	eyeViewMatrix = cam->getViewMatrix();

	viewVec = glm::normalize(cam->front); // normalize just to be sure (camera front should always be normalized)
	

	lightVec = -dirLight->getDirection(); // this is surely normalized since getDirection() returns glm::normalized vec

	//viewVec.y = 0.0f;
	//lightVec.y = 0.0f;
	//viewVec = glm::normalize(viewVec);
	//lightVec = glm::normalize(lightVec);


	//cout << glm::dot(viewVec, lightVec) << endl;

	if (glm::dot(viewVec, lightVec) > inversionThreshold) {
		halfVec = glm::normalize(viewVec + lightVec);
		if (forceHalfVecToFaceCam) {
			halfVec.y = viewVec.y;
			halfVec = glm::normalize(halfVec);
		}
		invertedView = false;
	} else {
		halfVec = glm::normalize(-viewVec + lightVec);
		if (forceHalfVecToFaceCam) {
			halfVec.y = -viewVec.y;
			halfVec = glm::normalize(halfVec);
		}
		invertedView = true;
	}

	/*
	lightVec = -dirLight->getDirection(); // this is surely normalized since getDirection() returns glm::normalized vec
	//lightVec = glm::normalize(dirLight->position); // according to the Nvidia source code

	//cout << glm::dot(viewVec, lightVec) << endl;

	if (glm::dot(viewVec, lightVec) > 0.0f) {
		halfVec = glm::normalize(viewVec + lightVec);
		invertedView = false;
	} else {
		halfVec = glm::normalize(-viewVec + lightVec);
		invertedView = true;
	}
	*/

	lightPosEye = eyeViewMatrix * glm::vec4(dirLight->position, 1.0f);


	lightViewMatrix = dirLight->getViewMatrix();
	lightProjectionMatrix = dirLight->getProjectionMatrix();

	shadowMatrix = lightProjectionMatrix * lightViewMatrix * glm::inverse(eyeViewMatrix);
	shadowMatrix = glm::scale(shadowMatrix, glm::vec3(0.5f));
	shadowMatrix = glm::translate(shadowMatrix, glm::vec3(0.5f));

	//eyePos = glm::inverse(cam->getViewMatrix()) * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	//cout << eyePos.x << " " << eyePos.y << " " << eyePos.z << ", " << cam->position.x << " " << cam->position.y << " " << cam->position.z << endl;
	
	eyePos = cam->position;

	halfVecEye = eyeViewMatrix * glm::vec4(halfVec, 0.0f);

}

glm::vec3 ParticleRenderer::getSortVec() {
	return halfVec;
}

void ParticleRenderer::preSceneRenderImage() {
	glBindFramebuffer(GL_FRAMEBUFFER, imageFramebuffer);
	glViewport(0, 0, imageWidth, imageHeight);

	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_TRUE);
	glClear(GL_DEPTH_BUFFER_BIT);


}

void ParticleRenderer::postSceneRenderImage() {

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, vars->screenWidth, vars->screenHeight);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

}

void ParticleRenderer::updateShaderSet() {
	if (shaderSet != prevShaderSet) {
		switch (shaderSet) {
			case 0:
				shadowAlpha100x = 0.1f;
				firstPassShader = ShaderManager::getShaderPtr("volume_1st_pass");
				secondPassShader = ShaderManager::getShaderPtr("volume_2nd_pass");
				break;
			case 1:
				shadowAlpha100x = 10.0f;
				firstPassShader = ShaderManager::getShaderPtr("volume_1st_pass_alt");
				secondPassShader = ShaderManager::getShaderPtr("volume_2nd_pass_alt");
				break;
			case 2:
				shadowAlpha100x = 10.0f;
				firstPassShader = ShaderManager::getShaderPtr("volume_1st_pass_alt2");
				secondPassShader = ShaderManager::getShaderPtr("volume_2nd_pass_alt2");
				break;
			default:
				break;
		}
		prevShaderSet = shaderSet;

		//ps->stlpSim->uploadProfileIndicesUniforms(firstPassShader);
		//ps->stlpSim->uploadProfileIndicesUniforms(secondPassShader);

	}
}

const char * ParticleRenderer::getPhaseFunctionName(int phaseFunc) {
	switch (phaseFunc) {
		case ePhaseFunction::NONE:
			return "None";
			break;
		case ePhaseFunction::RAYLEIGH:
			return "Rayleigh";
			break;
		case ePhaseFunction::HENYEY_GREENSTEIN:
			return "Henyey-Greenstein";
			break;
		case ePhaseFunction::DOUBLE_HENYEY_GREENSTEIN:
			return "Double Henyey-Greenstein";
			break;
		case ePhaseFunction::SCHLICK:
			return "Schlick";
			break;
		case ePhaseFunction::CORNETTE_SHANK:
			return "Cornette-Shank";
			break;
		default:
			return "None";
			break;
	}
}


GLuint ParticleRenderer::createTextureHelper(GLenum target, int w, int h, GLint internalFormat, GLenum format) {
	GLuint texid;
	glGenTextures(1, &texid);
	glBindTexture(target, texid);

	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(target, 0, internalFormat, w, h, 0, format, GL_FLOAT, 0);
	return texid;
}

void ParticleRenderer::initFramebuffers() {

	// IMAGE FRAMEBUFFER
	if (imageFramebuffer) {
		glDeleteTextures(1, &imageTexture);
		glDeleteTextures(1, &imageDepthTexture);
		glDeleteFramebuffers(1, &imageFramebuffer);
	}

	imageWidth = vars->screenWidth / downSample;
	imageHeight = vars->screenHeight / downSample;

	GLint format = GL_RGBA16F;

	imageTexture = createTextureHelper(GL_TEXTURE_2D, imageWidth, imageHeight, format, GL_RGBA);
	imageDepthTexture = createTextureHelper(GL_TEXTURE_2D, imageWidth, imageHeight, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glGenFramebuffers(1, &imageFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, imageFramebuffer);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageTexture, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, imageDepthTexture, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Creation of framebuffers failed, glCheckFramebufferStatus() = 0x%x\n", status);
	}
	// LIGHT FRAMEBUFFER

	CHECK_GL_ERRORS();

	//GLint format = GL_RGBA16F;

	lightTexture[0] = createTextureHelper(GL_TEXTURE_2D, lightBufferResolution, lightBufferResolution, format, GL_RGBA);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	lightTexture[1] = createTextureHelper(GL_TEXTURE_2D, lightBufferResolution, lightBufferResolution, format, GL_RGBA);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);


	lightDepthTexture = createTextureHelper(GL_TEXTURE_2D, lightBufferResolution, lightBufferResolution, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT);

	glGenFramebuffers(1, &lightFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, lightFramebuffer);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightTexture[0], 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, lightDepthTexture, 0);

	CHECK_GL_ERRORS();

	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Creation of framebuffers failed, glCheckFramebufferStatus() = 0x%x\n", status);
	}





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




	TextureManager::pushCustomTexture(lightTexture[0], lightBufferResolution, lightBufferResolution, 4, "lightTexture[0]");
	TextureManager::pushCustomTexture(lightTexture[1], lightBufferResolution, lightBufferResolution, 4, "lightTexture[1]");
	TextureManager::pushCustomTexture(imageTexture, imageWidth, imageHeight, 4, "imageTexture");

}

void ParticleRenderer::drawSlices() {

	CHECK_GL_ERRORS();

	srcLightTexture = 0;

	//cout << "NUM ACTIVE PARTICLES = " << ps->numActiveParticles << endl;
	//cout << "num slices = " << numSlices << endl;
	//cout << "setting batchsize to " << (ps->numActiveParticles / numSlices) << endl;

	batchSize = ceil((float)ps->numActiveParticles / (float)numSlices);

	//cout << "BATCH SIZE = " << batchSize << endl;

	// clear light buffer
	
	glBindFramebuffer(GL_FRAMEBUFFER, lightFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightTexture[srcLightTexture], 0);
	glClearColor(1.0f - dirLight->color.x, 1.0f - dirLight->color.y, 1.0f - dirLight->color.z, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, imageFramebuffer);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	CHECK_GL_ERRORS();

	numDisplayedSlices = min(numDisplayedSlices, numSlices);


	//glActiveTexture(GL_TEXT)

	for (int i = 0; i < numDisplayedSlices; i++) {
		drawSlice(i);
		CHECK_GL_ERRORS();
		drawSliceLightView(i);
		CHECK_GL_ERRORS();

		if (useBlurPass) {
			blurLightTexture();
		}
		CHECK_GL_ERRORS();
	}





}

void ParticleRenderer::drawSlice(int i) {


	//glBindFramebuffer(GL_FRAMEBUFFER, imageFramebuffer);
	//glViewport(0, 0, imageWidth, imageHeight);

	if (vars->renderVolumeParticlesDirectly) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, vars->screenWidth, vars->screenHeight);
	} else {
		glBindFramebuffer(GL_FRAMEBUFFER, imageFramebuffer);
		glViewport(0, 0, imageWidth, imageHeight);
	}
	
	if (invertedView) {
		glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE);
	} else {
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	}
	

	drawPointSprites(secondPassShader, i * batchSize, batchSize, true);

}

void ParticleRenderer::drawSliceLightView(int i) {

	// TODO -> set matrix uniforms for lightView and lightProjection


	glBindFramebuffer(GL_FRAMEBUFFER, lightFramebuffer);
	glViewport(0, 0, lightBufferResolution, lightBufferResolution);

	// TODO -> set color (or color modifier if color taken from texture)
	//			-> if texture, set texture uniforms, activate texture unit and bind sprite texture

	// TODO: SET SHADOW COLOR ALPHA

	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);


	drawPointSprites(firstPassShader, i * batchSize, batchSize, false);





}

void ParticleRenderer::drawPointSprites(ShaderProgram * shader, int start, int count, bool shadowed) {

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE); // do not write depth (but do depth test)
	glEnable(GL_BLEND);

	shader->use();

	// TODO - set shader uniforms and bind textures
	//shader->

	if (shadowed) {
		shader->setInt("u_ShadowTexture", 1);
		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, lightTexture[srcLightTexture]);

		// TESTING - this is for cloud shadow rendering on the terrain!
		glActiveTexture(GL_TEXTURE0 + TEXTURE_UNIT_CLOUD_SHADOW_MAP);
		glBindTexture(GL_TEXTURE_2D, lightTexture[srcLightTexture]);
	}

	drawPoints(start, count, true);

	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);


}

void ParticleRenderer::drawPoints(int start, int count, bool sorted) {
	if (start > ps->numActiveParticles) {
		return;
	}

	glBindVertexArray(ps->particlesVAO);

	// they have positions, colors (each particle has a color?) and velocities -> use vertex attribute arrays

	if (start + count > ps->numActiveParticles) {
		count = ps->numActiveParticles - start;
	}

	if (sorted) {
		glDrawElements(GL_POINTS, count, GL_UNSIGNED_INT, (void*)(start * sizeof(unsigned int)));
	} else {
		glDrawArrays(GL_POINTS, start, count);
	}


	glBindVertexArray(0);


}

void ParticleRenderer::compositeResult() {

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, vars->screenWidth, vars->screenHeight);
	glDisable(GL_DEPTH_TEST);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glDepthMask(GL_TRUE);

	// draw texture
	passThruShader->use();
	passThruShader->setInt("u_Texture", 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, imageTexture);

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glDisable(GL_BLEND);


}

void ParticleRenderer::blurLightTexture() {

	GLuint pid = blurShader->id;
	blurShader->use();

	glDisable(GL_DEPTH_TEST);

	glBindFramebuffer(GL_FRAMEBUFFER, lightFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightTexture[1 - srcLightTexture], 0);

	//glClear(GL_COLOR_BUFFER_BIT);

	glViewport(0, 0, lightBufferResolution, lightBufferResolution);


	glUniform2f(glGetUniformLocation(pid, "u_TexelSize"), 1.0f / (float)lightBufferResolution, 1.0f / (float)lightBufferResolution);
	glBindTextureUnit(0, lightTexture[srcLightTexture]);

	glUniform1i(glGetUniformLocation(pid, "u_InputTexture"), 0);

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);

	srcLightTexture = 1 - srcLightTexture;



	/*

	glBindFramebuffer(GL_FRAMEBUFFER, lightFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightTexture[1], 0);

	//glClear(GL_COLOR_BUFFER_BIT);

	glViewport(0, 0, lightBufferResolution, lightBufferResolution);


	glUniform2f(glGetUniformLocation(pid, "u_TexelSize"), 1.0f / lightBufferResolution, 0.0f);
	glBindTextureUnit(0, lightTexture[0]);

	glUniform1i(glGetUniformLocation(pid, "u_InputTexture"), 0);

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightTexture[0], 0);
	//glClear(GL_COLOR_BUFFER_BIT);

	glUniform2f(glGetUniformLocation(pid, "u_TexelSize"), 0.0f, 1.0f / lightBufferResolution);
	glBindTextureUnit(0, lightTexture[1]);

	glUniform1i(glGetUniformLocation(pid, "u_InputTexture"), 0);

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);


	*/


	glEnable(GL_DEPTH_TEST);
	

}

