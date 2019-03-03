#pragma once

#include "STLPDiagram.h"
#include "VariableManager.h"

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

	int maxNumParticles = MAX_PARTICLE_COUNT;

	float groundHeight = 0.0f;
	float simulationBoxHeight = 20000.0f;
	float boxTopHeight;

	ShaderProgram *layerVisShader;


	float *d_verticalVelocities;
	int *d_profileIndices;




	STLPSimulatorCUDA(VariableManager *vars, STLPDiagram *stlpDiagram);
	~STLPSimulatorCUDA();



	/// Initializes buffers for the particles.
	void initBuffers();

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

