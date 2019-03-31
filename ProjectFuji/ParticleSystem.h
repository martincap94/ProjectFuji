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
#include "VariableManager.h"
#include "STLPSimulatorCUDA.h"
#include "ShaderProgram.h"
#include "Model.h"


class Emitter;
class CircleEmitter;

class LBM;

class ParticleSystem {
private:
	struct FormBoxSettings {
		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 size = glm::vec3(100.0f);
	};



public:

	std::vector<Emitter *> emitters;

	HeightMap *heightMap;

	VariableManager *vars;

	GLuint particleVerticesVBO;			///< VBO of the particle vertices
	GLuint particleProfilesVBO;
	GLuint colorsVBO;	///< VBO of the particle colors

	float *d_particleDistances;

	dim3 blockDim;
	dim3 gridDim;

	GLuint diagramParticlesVAO;
	GLuint diagramParticleVerticesVBO;

	bool editingFormBox = false;


	// testing -> provide setters later
	GLuint particlesVAO;			///< VAO of the particle vertices
	GLuint particlesEBO;


	FormBoxSettings newFormBoxSettings;
	FormBoxSettings formBoxSettings;


	LBM *lbm;
	STLPSimulatorCUDA *stlpSim;

	int numParticles;
	int numActiveParticles;

	int *d_numParticles;

	float *d_verticalVelocities;
	int *d_profileIndices;
	//float *d_particlePressures;

	Texture *spriteTexture;
	Texture *secondarySpriteTexture;

	float pointSize = 1500.0f;

	struct cudaGraphicsResource *cudaParticleVerticesVBO;
	struct cudaGraphicsResource *cudaParticleProfilesVBO;
	struct cudaGraphicsResource *cudaParticlesEBO;
	struct cudaGraphicsResource *cudaDiagramParticleVerticesVBO;

	// will be overwritten from config file if set!
	float positionRecalculationThreshold = 0.5f;
	int maxPositionRecalculations = 0;

	int opacityBlendMode = 1;
	float opacityBlendRange = 10.0f;

	int showHiddenParticles = 1;


	glm::vec3 particlesColor = glm::vec3(0.8f, 0.8f, 0.8f);


	// EMITTER TEMPORARY DATA FOR CURRENT FRAME
	std::vector<glm::vec3> particleVerticesToEmit;
	std::vector<int> particleProfilesToEmit;
	std::vector<float> verticalVelocitiesToEmit; // not necessary


	//glm::vec3 *particleVertices = nullptr;


	ParticleSystem(VariableManager *vars);
	~ParticleSystem();

	void doStep();
	void update();

	void initBuffers();
	void initCUDA();

	void emitParticles();

	void draw(const ShaderProgram &shader, glm::vec3 cameraPos);
	void drawGeometry(ShaderProgram *shader, glm::vec3 cameraPos);
	void drawDiagramParticles(ShaderProgram *shader);

	void drawHelperStructures();

	void drawHarris_1st_pass(glm::vec3 lightPos);
	void drawHarris_2nd_pass(glm::vec3 cameraPos);

	void sortParticlesByDistance(glm::vec3 referencePoint, eSortPolicy sortPolicy);
	void sortParticlesByProjection(glm::vec3 sortVector, eSortPolicy sortPolicy);

	void initParticlesWithZeros();
	void initParticlesOnTerrain();
	void initParticlesAboveTerrain();

	void formBox();
	void formBox(glm::vec3 pos, glm::vec3 size);

	void refreshParticlesOnTerrain();


	void activateAllParticles();
	void deactivateAllParticles();

	void enableAllEmitters();
	void disableAllEmitters();
	

	///// Initializes particle positions for 3D simulation.
	//void initParticlePositions(int width, int height, int depth, const HeightMap *hm);

	///// Copies data from VBO to CPU when we want to switch from GPU to CPU implementation.
	//void copyDataFromVBOtoCPU();



private:

	Model *formBoxVisModel;
	ShaderProgram *formBoxVisShader;


	ShaderProgram *harris_1st_pass_shader;
	ShaderProgram *harris_2nd_pass_shader;


	GLuint streamLinesVAO;	///< Streamlines VAO
	GLuint streamLinesVBO;	///< Streamlines VBO

	void generateParticleOnTerrain(std::vector<glm::vec3> &outVector);

};

