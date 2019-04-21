#pragma once

#include <glad\glad.h>

#include <glm\glm.hpp>

#include "VariableManager.h"
#include "ShaderProgram.h"
#include "HeightMap.h"

class TerrainPicker {
public:


	TerrainPicker(VariableManager *vars);
	~TerrainPicker();

	void drawTerrain();
	void drawTerrain(HeightMap *heightMap);
	void initFramebuffer();


	glm::vec3 getPixelData(int x, int y, bool &outTerrainHit, bool invertY = true);
	glm::vec3 getPixelData(glm::ivec2 screenPos, bool &outTerrainHit, bool invertY = true);

private:

	ShaderProgram *shader = nullptr;

	VariableManager *vars = nullptr;

	GLuint framebuffer;
	GLuint texture;
	GLuint depthTexture;


};

