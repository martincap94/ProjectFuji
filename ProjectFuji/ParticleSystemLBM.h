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

class LBM; // forward declaration
/// Particle system that is used in both 2D and 3D simulations.
/**
	Particle system that is used in both 2D and 3D simulations.
	Provides functionality to create particles, set their positions and to draw them.
	Drawing particles supports basic points, point sprites and also visualizing their
	colors through second VBO.
*/
class ParticleSystemLBM {
public:

	LBM *lbm; ///< Owner LBM

	int numParticles;				///< Number of particles in the simulation
	int *d_numParticles;			///< Device pointer to the number of particles

	bool drawStreamlines = false;	///< Whether the streamline visualization should be used (2D CPU only) - smallest priority (configuration file overwrites this setting as well as setting in the main function

	float pointSize = 1.0f;			///< Point size of the particles

	glm::vec3 particlesColor = glm::vec3(0.8f, 0.8f, 0.8f);	///< Color of the particles (if they're not drawn as point sprites or their velocity isn't visualized)

	Texture spriteTexture;	///< Point sprite texture	

	glm::vec3 *particleVertices = nullptr;		///< Particle vertex positions array
	glm::vec3 *streamLines = nullptr;			///< Array of all streamline points

	/// Default constructor.
	ParticleSystemLBM();

	/// Constructs particle system with the given number of particles and sets whether we should draw streamlines.
	/**
		Constructs particle system with the given number of particles and sets whether we should draw streamlines.
		\param[in] numParticles		Number of particles of the system.
		\param[in] drawStreamlines	Whether streamlines should be drawn (2D CPU only).
	*/
	ParticleSystemLBM(int numParticles, bool drawStreamlines = false);

	/// Deletes particleVertices, streamLines and frees d_numParticles on the device (GPU)
	~ParticleSystemLBM();

	/// Draws the particle system.
	/**
		Draws the particle system.
		If we do not use CUDA, we need to copy their positions to the buffer object in each frame.
		Based on the provided shader we draw either simple points or point sprites.
	*/
	void draw(const ShaderProgram &shader, bool useCUDA);

	/// Initializes particle positions for 2D simulation.
	void initParticlePositions(int width, int height, bool *collider);

	/// Initializes particle positions for 3D simulation.
	void initParticlePositions(int width, int height, int depth, const HeightMap *hm);

	/// Copies data from VBO to CPU when we want to switch from GPU to CPU implementation.
	void copyDataFromVBOtoCPU();

	GLuint vbo;			///< VBO of the particle vertices
	GLuint colorsVBO;	///< VBO of the particle colors

private:

	GLuint vao;			///< VAO of the particle vertices

	GLuint streamLinesVAO;	///< Streamlines VAO
	GLuint streamLinesVBO;	///< Streamlines VBO

};

