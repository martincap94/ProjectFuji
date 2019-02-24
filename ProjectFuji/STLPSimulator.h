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

	Particle testParticle; // test particle for initial implementation of convection process
	bool testing = true;

	HeightMap *heightMap;	///< Pointer to the heightmap

	vector<Particle> particles;
	vector<glm::vec3> particlePositions;	///< Particle positions
	int numParticles = 0;					///< Current number of particles

	/// Initializes buffers (calls initBuffers).
	STLPSimulator();

	/// Default destructor.
	~STLPSimulator();


	/// Initializes buffers for the particles.
	void initBuffers();

	/// Does single step of the simulation.
	void doStep();

	/// Generates single particle on the terrain.
	void generateParticle(bool setTestParticle = false);

	/// Draws the heightmap and particles.
	void draw(ShaderProgram &particlesShader);

	void initParticles();

private:

	GLuint particlesVAO;
	GLuint particlesVBO;

};

