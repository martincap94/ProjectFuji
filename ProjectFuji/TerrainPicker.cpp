#include "TerrainPicker.h"

#include "ShaderManager.h"
#include "TextureManager.h"
#include "MainFramebuffer.h"

TerrainPicker::TerrainPicker(VariableManager *vars) : vars(vars) {
	initFramebuffer();
	shader = ShaderManager::getShaderPtr("terrain_picker");
}

TerrainPicker::~TerrainPicker() {
}


void TerrainPicker::drawTerrain() {
	drawTerrain(vars->heightMap);
}

void TerrainPicker::drawTerrain(HeightMap *heightMap) {

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(0, 0, vars->screenWidth, vars->screenHeight);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	heightMap->drawGeometry(shader);

	vars->mainFramebuffer->bind();
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);




}


void TerrainPicker::initFramebuffer() {

	glGenFramebuffers(1, &framebuffer);
	
	glGenTextures(1, &texture);
	glGenTextures(1, &depthTexture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, vars->screenWidth, vars->screenHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


	glBindTexture(GL_TEXTURE_2D, depthTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, vars->screenWidth, vars->screenHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);


	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);


	vars->mainFramebuffer->bind();
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);

	TextureManager::pushCustomTexture(texture, vars->screenWidth, vars->screenHeight, 4, "Terrain Picker Data");





}

glm::vec3 TerrainPicker::getPixelData(int x, int y, bool &outTerrainHit, bool invertY) {
	if (invertY) {
		y = vars->screenHeight - y;
	}

	GLfloat readData[4];

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, readData);
	vars->mainFramebuffer->bind();
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	/*
	cout << readData[0] << endl;
	cout << readData[1] << endl;
	cout << readData[2] << endl;
	cout << readData[3] << endl;
	*/

	outTerrainHit = ((float)readData[3] == 1.0f);
	return glm::vec3(readData[0], readData[1], readData[2]);
}

glm::vec3 TerrainPicker::getPixelData(glm::ivec2 screenPos, bool &outTerrainHit, bool invertY) {
	return getPixelData(screenPos.x, screenPos.y, outTerrainHit, invertY);
}
