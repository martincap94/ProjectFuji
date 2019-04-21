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
#include "CircleEmitter.h"
#include "CDFEmitter.h"
//#include "EmitterBrushMode.h"
#include "PositionalCDFEmitter.h"

#include "UserInterface.h"
#include <nuklear.h>



class EmitterBrushMode;
class Emitter;

class LBM;

class ParticleSystem {
private:
	struct FormBoxSettings {
		glm::vec3 position = glm::vec3(1000.0f, 4500.0f, 2000.0f);
		glm::vec3 size = glm::vec3(4000.0f, 1000.0f, 3000.0f);
	};

	struct EmitterCreationHelper {
		CircleEmitter circleEmitter;
		CDFEmitter cdfEmitter;
		PositionalCDFEmitter pcdfEmitter;
	};


public:
	EmitterCreationHelper ech;


	std::vector<Emitter *> emitters;
	EmitterBrushMode *ebm = nullptr;

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

	int numDiagramParticlesToDraw = 0;
	glm::vec3 diagramParticlesColor = glm::vec3(1.0f, 0.0f, 0.0f);


	// EMITTER TEMPORARY DATA FOR CURRENT FRAME
	std::vector<glm::vec3> particleVerticesToEmit;
	std::vector<int> particleProfilesToEmit;
	std::vector<float> verticalVelocitiesToEmit; // not necessary


	//glm::vec3 *particleVertices = nullptr;


	ParticleSystem(VariableManager *vars);
	~ParticleSystem();


	void update();

	void initBuffers();
	void initCUDA();

	void emitParticles();

	void draw(glm::vec3 cameraPos);
	void drawGeometry(ShaderProgram *shader, glm::vec3 cameraPos);
	void drawDiagramParticles();

	void drawHelperStructures();

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

	void activateAllDiagramParticles();
	void deactivateAllDiagramParticles();

	void enableAllEmitters();
	void disableAllEmitters();

	void createPredefinedEmitters();
	void createEmitter(int emitterType, std::string emitterName);
	void deleteEmitter(int idx);

	void constructEmitterCreationWindow(struct nk_context *ctx, UserInterface *ui, int emitterType, bool &closeWindowAfterwards);


	
	void pushParticleToEmit(Particle p);

	///// Initializes particle positions for 3D simulation.
	//void initParticlePositions(int width, int height, int depth, const HeightMap *hm);

	///// Copies data from VBO to CPU when we want to switch from GPU to CPU implementation.
	//void copyDataFromVBOtoCPU();



private:

	ShaderProgram *curveShader = nullptr;

	Model *formBoxVisModel;
	ShaderProgram *formBoxVisShader;

	ShaderProgram *pointSpriteTestShader = nullptr;
	ShaderProgram *singleColorShader = nullptr;


	GLuint streamLinesVAO;	///< Streamlines VAO
	GLuint streamLinesVBO;	///< Streamlines VBO

	void generateParticleOnTerrain(std::vector<glm::vec3> &outVector);

};

