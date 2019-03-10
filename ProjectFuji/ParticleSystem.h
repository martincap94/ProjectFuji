///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       ParticleSystemLBM.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines ParticleSystemLBM class that is used in both 2D and 3D simulations.
*
*  Defines ParticleSystemLBM class that is used in both 2D and 3D simulations.
*  As you may notice, the class uses glm::vec3 for particle vertices representation which is
*  very inefficient when 2D simulation is used. This stems from the fact that I originally
*  planned to remove 2D simulation in the process but it proved very useful for testing concepts
*  and for visualizing scenes that are difficult to debug in 3D.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <glm\glm.hpp>
#include "ShaderProgram.h"
#include "HeightMap.h"
#include <vector>
#include <deque>

#include "Texture.h"
#include "Emitter.h"
#include "VariableManager.h"
#include "STLPSimulatorCUDA.h"

class LBM;

class ParticleSystem {
public:

	std::vector<Emitter *> emitters;

	HeightMap *heightMap;

	VariableManager *vars;

	GLuint particleVerticesVBO;			///< VBO of the particle vertices
	GLuint particleProfilesVBO;
	GLuint colorsVBO;	///< VBO of the particle colors


	LBM *lbm;
	STLPSimulatorCUDA *stlpSim;

	int numParticles;
	int *d_numParticles;

	float *d_verticalVelocities;
	int *d_profileIndices;
	float *d_particlePressures;

	Texture spriteTexture;
	Texture secondarySpriteTexture;

	float pointSize = 10.0f;

	struct cudaGraphicsResource *cudaParticleVerticesVBO;
	struct cudaGraphicsResource *cudaParticleProfilesVBO;

	float positionRecalculationThreshold = 0.5f;
	int maxPositionRecalculations = 10;


	glm::vec3 particlesColor = glm::vec3(0.8f, 0.8f, 0.8f);


	//glm::vec3 *particleVertices = nullptr;


	ParticleSystem(VariableManager *vars);
	~ParticleSystem();

	void initBuffers();
	void initCUDA();

	void draw(const ShaderProgram &shader, glm::vec3 cameraPos);

	void initParticlesWithZeros();
	void initParticlesOnTerrain();
	void initParticlesAboveTerrain();



	///// Initializes particle positions for 3D simulation.
	//void initParticlePositions(int width, int height, int depth, const HeightMap *hm);

	///// Copies data from VBO to CPU when we want to switch from GPU to CPU implementation.
	//void copyDataFromVBOtoCPU();



private:

	GLuint particlesVAO;			///< VAO of the particle vertices

	GLuint streamLinesVAO;	///< Streamlines VAO
	GLuint streamLinesVBO;	///< Streamlines VBO

	void generateParticleOnTerrain(std::vector<glm::vec3> &outVector);

};

