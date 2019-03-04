///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Simulator.h
* \author     Martin Cap
* \date       2019/01/18
* \brief	  Initial version of the Simulator.
*
*  Simulator to be extended for orographic cloud simulation. Currently just spawns
*  particles on the terrain.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////


#pragma once

#include <glad\glad.h>
#include <vector>

#include "HeightMap.h"
#include "Particle.h"
#include "STLPDiagram.h"
#include "VariableManager.h"
#include "ppmImage.h"



/// Simulator to be used for orographic cloud simulation.
/**
	Simulator to be extended for orographic cloud simulation. Currently just spawns
	particles on the terrain.
*/
class STLPSimulator {
public:

	STLPDiagram *stlpDiagram;

	// testing stuff out
	//glm::vec2 T_c;
	//glm::vec2 CCL;
	//glm::vec2 EL;

	float delta_t = 1.0f;
	//float delta_t = 0.01f;

	bool testing = false;

	int simulateWind = 0;
	int usePrevVelocity = 1;
	int showCCLLevelLayer = 1;
	int showELLevelLayer = 1;

	HeightMap *heightMap;	///< Pointer to the heightmap
	ppmImage *profileMap; // needs to have the same parameters as the height map (width, height), or at least larger

	vector<Particle> particles;
	vector<glm::vec3> particlePositions;	///< Particle positions
	int numParticles = 0;					///< Current number of particles
	float simulationSpeedMultiplier = 1.0f;

	//int maxNumParticles = MAX_PARTICLE_COUNT;
	int maxNumParticles = 100000;

	float groundHeight = 0.0f;
	float simulationBoxHeight = 20000.0f; // 20km
	float boxTopHeight;

	VariableManager *vars;

	ShaderProgram *layerVisShader;

	/// Initializes buffers (calls initBuffers).
	STLPSimulator(VariableManager *vars, STLPDiagram *stlpDiagram);

	/// Default destructor.
	~STLPSimulator();


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

