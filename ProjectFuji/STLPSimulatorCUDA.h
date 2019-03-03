#pragma once

#include "STLPDiagram.h"
#include "VariableManager.h"
#include "Particle.h"

#include <cuda_gl_interop.h>


class HeightMap;

class STLPSimulatorCUDA {
public:

	VariableManager *vars;
	STLPDiagram *stlpDiagram;

	float delta_t = 60.0f;

	int simulateWind = 0;
	int usePrevVelocity = 1;
	int showCCLLevelLayer = 1;
	int showELLevelLayer = 1;

	HeightMap *heightMap;

	//int maxNumParticles = MAX_PARTICLE_COUNT;
	int maxNumParticles = 10000;

	float groundHeight = 0.0f;
	float simulationBoxHeight = 20000.0f;
	float boxTopHeight;

	ShaderProgram *layerVisShader;

	vector<Particle> particles;
	vector<glm::vec3> particlePositions;
	int numParticles = 0;

	struct cudaGraphicsResource *cudaParticleVerticesVBO;


	float *d_verticalVelocities;
	int *d_profileIndices;
	float *d_particlePressures;

	// for now, let us have curves here (that are copied from the CPU precomputation)
	glm::vec2 *d_ambientTempCurve;
	
	//vector<glm::vec2 *> d_dryAdiabatProfiles;
	//vector<glm::vec2 *> d_moistAdiabatProfiles;
	//vector<glm::vec2 *> d_CCLProfiles;
	//vector<glm::vec2 *> d_TcProfiles;
	
	// use flattened arrays (with offsets)
	glm::vec2 *d_dryAdiabatProfiles;
	int *d_dryAdiabatOffsets;
	glm::vec2 *d_moistAdiabatProfiles;
	int *d_moistAdiabatOffsets;
	glm::vec2 *d_CCLProfiles;
	glm::vec2 *d_TcProfiles;

	dim3 gridDim;
	dim3 blockDim;


	STLPSimulatorCUDA(VariableManager *vars, STLPDiagram *stlpDiagram);
	~STLPSimulatorCUDA();



	/// Initializes buffers for the particles.
	void initBuffers();

	void initCUDA();

	/// Does single step of the simulation.
	void doStep();

	void resetSimulation();


	/// Generates single particle on the terrain.
	void generateParticle();

	/// Draws the heightmap and particles.
	void draw(ShaderProgram &particlesShader);

	void initParticles();

	void mapToSimulationBox(float &val);
	void mapFromSimulationBox(float &val);

private:

	GLuint particlesVAO;
	GLuint particlesVBO;

	GLuint CCLLevelVAO;
	GLuint CCLLevelVBO;

	GLuint ELLevelVAO;
	GLuint ELLevelVBO;


};

