///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       STLPSimulator.h
* \author     Martin Cap
* \date       2019/01/18
*
*	--- DEPRECATED ---
*	\deprecated
*	SkewT/LogP simulator that runs the cloud simulation on CPU.
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
#include "Texture.h"


//! STLPSimulator that runs on the CPU.
/*!
	--- DEPRECATED ---
	\deprecated STLPSimulator that runs Duarte's cloud simulation on the CPU.
	Currently it is not in use and not tested.
*/
class STLPSimulator {
public:

	STLPDiagram *stlpDiagram;		//!< STLPDiagram to be used for simulation

	float delta_t = 1.0f;			//!< Delta t of the simulation step
	//float delta_t = 0.01f;

	bool testing = false;			//!< Debug variable for testing purposes

	int simulateWind = 0;			//!< Whether to simulate wind naively
	int usePrevVelocity = 1;		//!< Whether to use previous velocity (or set it to zero)

	float pointSize = 10.0f;		//!< Point size of the particles

	GLuint particlesVBO;			//!< VBO of the particle vertex positions

	Texture spriteTexture;			//!< Sprite texture used for drawing the particles

	HeightMap *heightMap;	//!< Pointer to the heightmap
	ppmImage *profileMap;	//!< Profile map that determines ranges of the particle convective temperature profiles

	vector<Particle> particles;				//!< CPU vector of particles
	vector<glm::vec3> particlePositions;	//!< Particle positions
	int numParticles = 0;					//!< Current number of particles
	float simulationSpeedMultiplier = 1.0f;	//!< Multiplier of the simulation speed

	int maxNumParticles = 10000;			//!< Maximum number of particles in the CPU simulation

	float groundHeight = 0.0f;				//!< Ground height taken from the sounding data used for coordinate system changes
	float simulationBoxHeight = 20000.0f;	//!< Height of the simulation box
	float boxTopHeight;						//!< Maximum height reached by the simulation box

	glm::vec3 currCameraPos;				//!< Current position of the camera

	VariableManager *vars;					//!< VariableManager for this simulator

	ShaderProgram *layerVisShader;			//!< Shader that is used in CCL/LCL, EL level visualizations

	//! Initializes buffers (calls initBuffers).
	/*!
		\param[in] vars				VariableManager to be used by this simulator.
		\param[in] stlpDiagram		STLPDiagram to be used in this simulator.
	*/
	STLPSimulator(VariableManager *vars, STLPDiagram *stlpDiagram);

	//! Default destructor.
	~STLPSimulator();


	//! Initializes buffers for the particles.
	void initBuffers();

	//! Does single step of the simulation.
	void doStep();

	//! Resets the simulation to initial state.
	void resetSimulation();

	//! Generates single particle on the terrain.
	void generateParticle();

	//! Draws the heightmap and particles.
	/*!
		\param[in] particlesShader		Shader to be used when drawing particles.
		\param[in] cameraPos			Position of the camera.
	*/
	void draw(ShaderProgram &particlesShader, glm::vec3 cameraPos = glm::vec3(0.0f));



	//! Maps the value to the simulation box from the world size coordinate system.
	/*!
	\param[in] val	Value to be mapped.
	*/
	void mapToSimulationBox(float &val);

	//! Maps the value from the simulation box to world size coordinate system.
	/*!
	\param[in] val	Value to be mapped.
	*/
	void mapFromSimulationBox(float &val);

private:

	GLuint particlesVAO;	//!< VAO for the particle VBOs

	GLuint CCLLevelVAO;		//!< VAO for the CCL level visualization
	GLuint CCLLevelVBO;		//!< VBO for the CCL level visualization

	GLuint ELLevelVAO;		//!< VAO for the EL level visualization
	GLuint ELLevelVBO;		//!< VBO for the EL level visualization

};

