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
public:

	VariableManager *vars = nullptr;

	LBM3D_1D_indices *lbm = nullptr;
	//HeightMap *heightMap = nullptr;

	int maxNumStreamlines;
	int maxStreamlineLength;

	// bools for UI
	int visible = 1;
	int liveLineCleanup = 1;

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


	StreamlineParticleSystem(VariableManager *vars);
	~StreamlineParticleSystem();

	void draw();
	void init();
	void update();
	void initBuffers();
	void initCUDA();


	void activate();
	void deactivate();
	void reset();


	void cleanupLines();


	void setPositionInHorizontalLine();
	void setPositionInVerticalLine();

private:

	ShaderProgram *shader = nullptr;





};

