///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       StreamlineParticleSystem.h
* \author     Martin Cap
*
*	Special particle system that is used for generating streamlines. This is particularly useful
*	for LBM testing.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>
#include <glm\glm.hpp>
#include <glad\glad.h>

#include "ShaderProgram.h"

class LBM3D_1D_indices;
//class HeightMap;
class VariableManager;

// only for streamlines in LBM
class StreamlineParticleSystem {

private:
	struct HorizontalParametersSettings {
		float xOffset = 0.0f;
		float yOffset = 0.5f;
	};

	struct VerticalParametersSettings {
		float xOffset = 0.0f;
		float zOffset = 0.5f;
	};


public:

	VariableManager *vars = nullptr;

	LBM3D_1D_indices *lbm = nullptr;
	//HeightMap *heightMap = nullptr;

	int maxNumStreamlines;
	int maxStreamlineLength;

	// bools for UI
	int visible = 1;
	int liveLineCleanup = 1;

	bool editingHorizontalParameters = false;
	bool editingVerticalParameters = false;

	HorizontalParametersSettings hParams;
	VerticalParametersSettings vParams;


	bool active = false;
	bool initialized = false;

	int frameCounter = 0; // counts how many frames were drawn during streamline simulation

	GLuint streamlinesVAO;

	GLuint streamlinesVBO;

	struct cudaGraphicsResource *cudaStreamlinesVBO = nullptr;
	int *d_currActiveVertices; // counters for all streamlines
	int *currActiveVertices = nullptr;

	//std::vector<GLuint> streamlineVBOs;
	//GLuint *streamlineVBOs = nullptr;
	//GLuint streamlinesVBO;
	//GLuint streamlineOffsetsVBO;


	//struct cudaGraphicsResource *cudaStreamlinesVBO;
	//struct cudaGraphicsResource *cudaStreamlineOffsetsVBO;
	//std::vector<struct cudaGraphicsResource *> cudaStreamlinesVBOs;



	//std::vector<glm::vec3> streamlineVertices;


	StreamlineParticleSystem(VariableManager *vars, LBM3D_1D_indices *lbm);
	~StreamlineParticleSystem();

	void draw();
	void init();
	void update();
	void initBuffers();
	void initCUDA();

	void tearDown();


	void activate();
	void deactivate();
	void reset();


	void cleanupLines();


	void setPositionInHorizontalLine();
	void setPositionInVerticalLine();
	void setPositionCross();


	//void centerHParamsYOffset();
	//void centerVParamsZOffset();

private:

	ShaderProgram *shader = nullptr;





};

