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

	void drawTerrain(HeightMap *heightMap);
	void initFramebuffer();

	glm::vec4 getPixelData(int x, int y);
	glm::vec4 getPixelData(glm::ivec2 screenPos);


private:

	ShaderProgram *shader = nullptr;

	VariableManager *vars = nullptr;

	GLuint framebuffer;
	GLuint texture;
	GLuint depthTexture;


};

