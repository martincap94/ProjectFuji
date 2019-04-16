#pragma once

#include "STLPDiagram.h"
#include "VariableManager.h"
#include "Particle.h"
#include "Texture.h"
#include "ppmImage.h"

#include <cuda_gl_interop.h>


class HeightMap;
class ParticleSystem;

class STLPSimulatorCUDA {
public:

	VariableManager *vars;
	STLPDiagram *stlpDiagram;

	ParticleSystem *particleSystem;

	GLuint particlesVBO;
	GLuint profileDataSSBO;

	Texture spriteTexture;
	Texture secondarySpriteTexture;

	float pointSize = 1500.0f;


	float delta_t = 1.0f;

	int simulateWind = 0;
	int usePrevVelocity = 1;

	HeightMap *heightMap;


	ppmImage *profileMap; // needs to have the same parameters as the height map (width, height), or at least larger

	float groundHeight = 0.0f;
	float simulationBoxHeight = 20000.0f;
	float boxTopHeight;

	ShaderProgram *layerVisShader;

	vector<Particle> particles;
	vector<glm::vec3> particlePositions;
	int numParticles = 0;

	struct cudaGraphicsResource *cudaParticleVerticesVBO;


	float *d_verticalVelocities;
	float *d_particlePressures;

	// for now, let us have curves here (that are copied from the CPU precomputation)
	glm::vec2 *d_ambientTempCurve;
	

	
	// use flattened arrays (with offsets)
	glm::vec2 *d_dryAdiabatProfiles;
	glm::ivec2 *d_dryAdiabatOffsetsAndLengths;
	glm::vec2 *d_moistAdiabatProfiles;
	glm::ivec2 *d_moistAdiabatOffsetsAndLengths;
	glm::vec2 *d_CCLProfiles;
	glm::vec2 *d_TcProfiles;

	dim3 gridDim;
	dim3 blockDim;


	STLPSimulatorCUDA(VariableManager *vars, STLPDiagram *stlpDiagram);
	~STLPSimulatorCUDA();



	/// Initializes buffers for the particles.
	void initBuffers();
	void uploadProfileIndicesUniforms(ShaderProgram *shader);

	void initCUDA();

	// More general approach that would enable us to change the diagram (profiles) at runtime
	void initCUDAGeneral();
	void uploadDataFromDiagramToGPU();

	/// Does single step of the simulation.
	void doStep();

	void resetSimulation();


	/// Generates single particle on the terrain.
	void generateParticle();

	/// Draws the heightmap and particles.
	void draw(glm::vec3 cameraPos);
	void drawDiagramParticles(ShaderProgram *shader);

	//void initParticles();

	void mapToSimulationBox(float &val);
	void mapFromSimulationBox(float &val);


private:


	GLuint particleProfilesVBO;

	GLuint groundLevelVAO;
	GLuint groundLevelVBO;

	GLuint CCLLevelVAO;
	GLuint CCLLevelVBO;

	GLuint ELLevelVAO;
	GLuint ELLevelVBO;


};

