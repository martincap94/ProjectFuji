///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       ParticleSystem.h
* \author     Martin Cap
*
*	Defines ParticleSystem class that manages all particle data on the GPU. Particle vertices and
*	convective temperature indices are stored in OpenGL VBOs, while other data such as their velocities
*	are stored in CUDA global memory on the GPU.
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

//! ParticleSystem used throughout the whole application.
/*!
	Holds all necessary information about the particles that are used in both STLP and LBM simulators.
	The data is stored mainly on GPU, either in OpenGL VBOs or in CUDA device global memory based on its usage.

*/
class ParticleSystem {
private:

	//! Defines settings for the box that is used to generate box of particles.
	struct FormBoxSettings {
		glm::vec3 position = glm::vec3(1000.0f, 4500.0f, 2000.0f);	//!< Position of the box (bottom, left corner)
		glm::vec3 size = glm::vec3(4000.0f, 1000.0f, 3000.0f);		//!< Size of the box (in meters)
	};

	//! Helper structure that holds uninitialized emitters that are used in UI in emitter creation window.
	struct EmitterCreationHelper {
		CircleEmitter circleEmitter;		//!< Uninitialized CircleEmitter
		CDFEmitter cdfEmitter;				//!< Uninitialized CDFEmitter
		PositionalCDFEmitter pcdfEmitter;	//!< Uninitialized PositionalCDFEmitter
	};


public:
	EmitterCreationHelper ech;	//!< Holds uninitialized emitters for UI usage


	std::vector<Emitter *> emitters;	//!< List of all emitters associated with this particle system
	EmitterBrushMode *ebm = nullptr;	//!< Brush mode that is used with this particle system

	HeightMap *heightMap;		//!< Heightmap used in the application
	VariableManager *vars;		//!< VariableManager pointer for this class

	GLuint particleVerticesVBO;	//!< VBO of the particle vertices (world positions)
	GLuint particleProfilesVBO;	//!< VBO of the particle convective temperature profile indices
	GLuint colorsVBO;			//!< --- OLD --- VBO of the particle colors

	float *d_particleDistances;	//!< Helper CUDA array that is used when sorting particles by key

	dim3 blockDim;	//!< Block dimensions for CUDA kernel calls
	dim3 gridDim;	//!< Grid dimensions for CUDA kernel calls

	GLuint diagramParticlesVAO;			//!< VAO for particle vertices in the diagram visualization
	GLuint diagramParticleVerticesVBO;	//!< VBO for particle vertices in the diagram visualization

	bool editingFormBox = false;	//!< Whether we are currently editing the box for generating particles


	// testing -> provide setters later
	GLuint particlesVAO;		//!< VAO of the particles
	GLuint particlesEBO;		//!< EBO of the particles - useful for sorting!s


	FormBoxSettings newFormBoxSettings;	//!< Newly created form box settings (using the UI)
	FormBoxSettings formBoxSettings;	//!< Current form box settings


	LBM *lbm;	//!< Pointer to the LBM simulator
	STLPSimulatorCUDA *stlpSim;	//!< Pointer to the STLPSimulator

	int numParticles;		//!< Number of particles (denotes maximum, all are allocated and prepared for use)
	int numActiveParticles;	//!< Number of active particles (visible particles)

	int *d_numParticles;	//!< Number of particles (value for the GPU)
	float *d_verticalVelocities;	//!< Array of vertical velocities on the GPU, used in STLPSimulator

	Texture *spriteTexture;				//!< Sprite texture used for drawing the particles
	Texture *secondarySpriteTexture;	//!< Debug secondary sprite texture

	float pointSize = 1500.0f;			//!< Relative scale of the particle points

	struct cudaGraphicsResource *cudaParticleVerticesVBO;			//!< CUDA pointer to particle vertices VBO
	struct cudaGraphicsResource *cudaParticleProfilesVBO;			//!< CUDA pointer to particle profiles VBO
	struct cudaGraphicsResource *cudaParticlesEBO;					//!< CUDA pointer to particles EBO
	struct cudaGraphicsResource *cudaDiagramParticleVerticesVBO;	//!< CUDA pointer to diagram particle vertices VBO


	int opacityBlendMode = 1;			//!< Mode for opacity blending from the old particle rendering
	float opacityBlendRange = 10.0f;	//!< Range of blending (when particles are crossing CCL/LCL) from the old rendering pipeline

	int showHiddenParticles = 1;	//!< Whether to show particles that should be hidden/invisible

	int synchronizeDiagramParticlesWithActiveParticles = 1;

	glm::vec3 particlesColor = glm::vec3(0.8f, 0.8f, 0.8f);	//!< Old particle color setting

	int numDiagramParticlesToDraw = 0;	//!< Number of particles that should be drawn in the diagram
	glm::vec3 diagramParticlesColor = glm::vec3(1.0f, 0.0f, 0.0f);	//!< Color of particles that are drawn in the diagram


	// EMITTER TEMPORARY DATA FOR CURRENT FRAME - the indices of particles must correspond
	std::vector<glm::vec3> particleVerticesToEmit;	//!< Particle vertices to be emitted in the next frame
	std::vector<int> particleProfilesToEmit;		//!< Particle profiles to be emitted in the next frame
	std::vector<float> verticalVelocitiesToEmit;	//!< Particle velocities to be emitted in the next frame


	//! Initializes the particle system including its shaders, buffers, CUDA memory and emitters.]
	/*!
		\param[in] vars		VariableManager to be used by this instance.
		\see initBuffers()
		\see initCUDA()
	*/
	ParticleSystem(VariableManager *vars);

	//! Frees all allocated data by this particle system (including CUDA memory).
	~ParticleSystem();

	//! Updates the emitters and brush mode (emits particles).
	void update();

	//! Initializes OpenGL buffers for the particles and maps them to CUDA pointers if necessary.
	void initBuffers();

	//! Allocates global memory for the particle arrays on GPU device.
	void initCUDA();

	//! Emits the particles from all active emitters.
	void emitParticles();

	//! --- DEPRECATED --- Draws the particles using the old rendering system.
	/*!
		\param[in] cameraPos	World position of the camera.
		\deprecated				Use new rendering available in the ParticleRenderer class.
	*/
	void draw(glm::vec3 cameraPos);

	//! --- DEPRECATED --- Draws only the particle points.
	/*!
		\param[in] shader		Shader to be used.
		\param[in] cameraPos	World position of the camera.
		\deprecated 			Use new rendering available in the ParticleRenderer class.
	*/
	void drawGeometry(ShaderProgram *shader, glm::vec3 cameraPos);

	//! Draws the diagram particles.
	void drawDiagramParticles();

	//! Draws helper structures and visualizations of all visible emitters.
	void drawHelperStructures();

	//! Sorts particles by distance using Thrust library.
	/*!
		\param[in] referencePoint	Point to which the distance is computed and used as a key for sorting.
		\param[in] sortPolicy		How the particles are sorted (<, <=, >, >=).
	*/
	void sortParticlesByDistance(glm::vec3 referencePoint, eSortPolicy sortPolicy);

	//! Sorts particles by projected distances using Thrust library.
	/*!
		\param[in] sortVector		Vector onto which the particle positions are projected using dot product.
		\param[in] sortPolicy		How the particles are sorted (<, <=, >, >=).
	*/
	void sortParticlesByProjection(glm::vec3 sortVector, eSortPolicy sortPolicy);

	//! Calls kernel that checks if particle positions are valid (not NaN or infinity).
	/*!
		If a particle is not valid, resets it to world origin.
		This prevents problems when the simulation of LBM becomes unstable where NaN values are generated.
	*/
	void checkParticleValidity();

	//! --- NOT IMPPLEMENTED YET ---
	void initParticlesWithZeros();

	//! Initializes the particles on terrain.
	void initParticlesOnTerrain();

	//! --- NOT IMPLEMENTED YET ---
	void initParticlesAboveTerrain();

	//! Forms a box of particles using current form box settings.
	void formBox();

	//! Forms a box of particles using the provided box settings.
	/*!
		\param[in] pos		Position of the box.
		\param[in] size		Size of the box.
	*/
	void formBox(glm::vec3 pos, glm::vec3 size);

	//! Moves all particles onto the terrain surface.
	void refreshParticlesOnTerrain();

	//! Runs a kernel that resets all particle vertical velocities to zeros.
	/*!
		\param[in] clearActiveOnly		Whether to clear velocities of active particles only.
	*/
	void clearVerticalVelocities(bool clearActiveOnly = false);

	//! Activates all particles.
	void activateAllParticles();

	//! Deactivates all particles.
	void deactivateAllParticles();

	//! Activates all diagram particles.
	void activateAllDiagramParticles();

	//! Deactivates all diagram particles.
	void deactivateAllDiagramParticles();

	//! Enables all emitters associated with this system.
	void enableAllEmitters();
	
	//! Disables all emitters associated with this system.
	void disableAllEmitters();

	//! Creates hardcoded emitters for debugging.
	void createPredefinedEmitters();

	//! Create an emitter with the given properties.
	/*!
		\param[in] emitterType		Type of the emitter to be created.
		\param[in] emitterName		Name given to the emitter.
	*/
	void createEmitter(int emitterType, std::string emitterName);

	//! Deletes an emitter with the given index.
	/*!
		\param[in] idx		Index of the emitter to be deleted.
	*/
	void deleteEmitter(int idx);

	//! Constructs emitter creation window for the user interface.
	/*!
		\param[in] ctx				Nuklear context used.
		\param[in] ui				UserInterface for which the window should be created.
		\param[in] emitterType		Emitter type to be created.
		\param[in] closeWindowAfterwards	Whether the window should be closed after this draw call.
	*/
	void constructEmitterCreationWindow(struct nk_context *ctx, UserInterface *ui, int emitterType, bool &closeWindowAfterwards);


	//! Pushes a particle to list of particles that are to be emitted in this frame.
	/*!
		\param[in] p		Particle to be emitted.
	*/
	void pushParticleToEmit(Particle p);

	///// Initializes particle positions for 3D simulation.
	//void initParticlePositions(int width, int height, int depth, const HeightMap *hm);

	///// Copies data from VBO to CPU when we want to switch from GPU to CPU implementation.
	//void copyDataFromVBOtoCPU();

	//! Saves the particle positions to file.
	/*!
		Warning: currently does not warn user that the file already exists (if it exists) and rewrites it.
		\param[in] filename				Name of the file.
		\param[in] saveOnlyActive		Whether to save active particles only.
	*/
	void saveParticlesToFile(std::string filename, bool saveOnlyActive = false);

	//! Constructs UI window for saving particle positions to file.
	void constructSaveParticlesWindow(struct nk_context *ctx, UserInterface *ui, bool &closeWindowAfterwards);

	//! Constructs UI window for loading particle positions from file.
	void constructLoadParticlesWindow(struct nk_context *ctx, UserInterface *ui, bool &closeWindowAfterwards);

	//! Loads particle positions from file.
	/*!
		\param[in] filename		Binary file where the particle positions are saved.
	*/
	void loadParticlesFromFile(std::string filename);

	//! Loads all particle save files stored in the particle saves directory.
	void loadParticleSaveFiles();



private:

	ShaderProgram *curveShader = nullptr;	//!< Shader used to draw diagram primitives

	Model *formBoxVisModel;				//!< Model used for visualizing the form box
	ShaderProgram *formBoxVisShader;	//!< Shader used to draw the form box

	ShaderProgram *pointSpriteTestShader = nullptr;	//!< Old shader that was used to draw particles
	ShaderProgram *singleColorShader = nullptr;		//!< Simple shader that draws everything using a single uniform color

	std::vector<std::string> particleSaveFiles;		//!< List of particle save files



	GLuint streamLinesVAO;	//!< Old streamlines VAO
	GLuint streamLinesVBO;	//!< Old streamlines VBO


};

