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

class LBM;

class ParticleSystem {
public:

	LBM *lbm;

	int numParticles;
	int *d_numParticles;


	float pointSize = 1.0f;

	glm::vec3 particlesColor = glm::vec3(0.8f, 0.8f, 0.8f);

	Texture spriteTexture;

	glm::vec3 *particleVertices = nullptr;


	ParticleSystem();
	ParticleSystem(int numParticles, bool drawStreamlines = false);
	~ParticleSystem();

	void draw(const ShaderProgram &shader, bool useCUDA);

	/// Initializes particle positions for 2D simulation.
	void initParticlePositions(int width, int height, bool *collider);

	/// Initializes particle positions for 3D simulation.
	void initParticlePositions(int width, int height, int depth, const HeightMap *hm);

	/// Copies data from VBO to CPU when we want to switch from GPU to CPU implementation.
	void copyDataFromVBOtoCPU();

	GLuint VBO;			///< VBO of the particle vertices
	GLuint colorsVBO;	///< VBO of the particle colors

private:

	GLuint VAO;			///< VAO of the particle vertices

	GLuint streamLinesVAO;	///< Streamlines VAO
	GLuint streamLinesVBO;	///< Streamlines VBO

};

