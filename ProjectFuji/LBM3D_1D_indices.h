///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       LBM3D_1D_indices.h
* \author     Martin Cap
* \date       2018/12/23
*
*  Defines the LBM3D class that implements the Lattice Boltzmann method 3D simulation on CPU and GPU.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"
#include "ParticleSystemLBM.h"
#include "DataStructures.h"
#include "HeightMap.h"
#include "VariableManager.h"
#include "STLPDiagram.h"
#include "GridLBM.h"

#include <cuda_gl_interop.h>


#include <vector>

#include "LBM.h"

#define WEIGHT_MIDDLE (1.0f / 3.0f)		//!< Initial weight for middle value of distribution function
#define WEIGHT_AXIS (1.0f / 18.0f)		//!< Initial weight for values that are axis aligned in the distribution function
#define WEIGHT_NON_AXIAL (1.0f / 36.0f)	//!< Initial weight for values that are not axis aligned (we could call them diagonal) in the distribution function

class ParticleSystem;
class StreamlineParticleSystem;



//! 3D LBM simulator.
/*!
	3D LBM simulator that supports both CPU and GPU simulations.
	GPU (CUDA) simulation is run through global kernels that are defined in LBM3D_1D_indices.cu.
	The LBM is indexed as an 1D array in this implementation.
	It used to be a subclass of LBM but for simplicity we have
	removed this hierarchy since LBM2D is no longer used.
*/
class LBM3D_1D_indices /*: public LBM*/ {

	// Direction vectors
	const glm::vec3 vMiddle = glm::vec3(0.0f, 0.0f, 0.0f);
	const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
	const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
	const glm::vec3 vBack = glm::vec3(0.0f, 0.0f, -1.0f);
	const glm::vec3 vFront = glm::vec3(0.0f, 0.0f, 1.0f);
	const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
	const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
	const glm::vec3 vBackRight = glm::vec3(1.0f, 0.0f, -1.0f);
	const glm::vec3 vBackLeft = glm::vec3(-1.0f, 0.0f, -1.0f);
	const glm::vec3 vFrontRight = glm::vec3(1.0f, 0.0f, 1.0f);
	const glm::vec3 vFrontLeft = glm::vec3(-1.0f, 0.0f, 1.0f);
	const glm::vec3 vTopBack = glm::vec3(0.0f, 1.0f, -1.0f);
	const glm::vec3 vTopFront = glm::vec3(0.0f, 1.0f, 1.0f);
	const glm::vec3 vBottomBack = glm::vec3(0.0f, -1.0f, -1.0f);
	const glm::vec3 vBottomFront = glm::vec3(0.0f, -1.0f, 1.0f);
	const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
	const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
	const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);
	const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);


public:

	//! Possible respawn modes of particles that leave the simulation area.
	enum eRespawnMode {
		CYCLE_ALL = 0,		//!< Cycle on all axes
		CYCLE_XZ,			//!< Cycle only on x and z axes (groud aligned)
		RANDOM_UNIFORM,		//!< Randomly respawn in the inlet wall
		_NUM_RESPAWN_MODES	//!< Number of the LBM respawn modes
	};

	//! --- DEPRECATED --- Possible out of bounds modes for particles that are out of the simulation area.
	enum eOutOfBoundsMode {
		IGNORE_PARTICLES = 0,		//!< Do nothing with these particles, ignore them
		DEACTIVATE_PARTICLES,		//!< Deactivate these particles (not implemented)
		RESPAWN_PARTICLES_INLET		//!< Respawn the particles in the inlet wall
	};

	GridLBM *grid = nullptr;		//!< Grid visualization of the LBM area (just a box at this moment)
	GridLBM *editGrid = nullptr;	//!< Secondary grid that is shown when we are editing the simulation area position/scale

	VariableManager *vars;			//!< VariableManager for this simulator


	glm::vec3 position = glm::vec3(0.0f);	//!< Position of the simulation area (bottom left corner)
	float scale = 100.0f;					//!< Scale of the simulation area (number of meters) / (cell unit size)


	ParticleSystemLBM *particleSystemLBM;	//!< Pointer to the old LBM particle system
	ParticleSystem *particleSystem;			//!< Pointer to the particle system
	StreamlineParticleSystem *streamlineParticleSystem;		//!< Pointer to the streamline particle system
	
	glm::vec3 *particleVertices;		//!< Pointer to the particle vertices array (on CPU)
	int *d_numParticles;	//!< Number of particles on the GPU; managed in memory by ParticleSystem class

	glm::vec3 *d_inletVelocities;	//!< --- EXPERIMENTAL --- Array of the inlet velocities
	STLPDiagram *stlpDiagram;		//!< STLPDiagram used to drive the STLPSimulatorCUDA


	int latticeWidth;		//!< Width of the lattice
	int latticeHeight;		//!< Height of the lattice
	int latticeDepth;		//!< Depth of the lattice

	int latticeSize;		//!< Number of lattice nodes = latticeWidth * latticeHeight * latticeDepth

	float tau = 0.52f;		//!< Tau parameter of the simulation, describes the viscosity of the simulated fluid
	float itau;				//!< Inverse value of tau = 1.0 / tau; it is used in the collision step, if tau is changed, this value must be recomputed
	float nu;				//!< Experimental value for subgrid model simulation

	glm::vec3 inletVelocity = glm::vec3(1.0f, 0.0f, 0.0f);		//!< Inlet velocity vector

	int useSubgridModel = 0;	//!< Whether the subgrid model should be used - EXPERIMENTAL - subgrid model not functional at the moment!
	int mirrorSides = 1;		//!< Whether the particles passing through sides of the scene bounding box should show up on the other side		
	int visualizeVelocity = 0;  //!< Whether the velocity of the particles should be visualized, currently only in 2D
	int respawnLinearly = 0;	//!< NOT USED YET! Whether particles should respawn linearly or randomly in the inlet nodes

	int respawnMode = CYCLE_ALL;	//!< Respawn mode used when particles are out of the simulation area
	int outOfBoundsMode = RESPAWN_PARTICLES_INLET;	//!< Out of bounds mode for particles that are outside the simulation area

	string sceneFilename;		//!< Filename of the scene that is used for the simulation


	int xLeftInlet = 1;		//!< Whether the x-left wall is an inlet
	int xRightInlet = 0;	//!< Whether the x-right wall is an inlet
	int zLeftInlet = 0;		//!< Whether the z-left wall is an inlet
	int zRightInlet = 0;	//!< Whether the z-right wall is an inlet
	int yBottomInlet = 0;	//!< Whether the y-bottom wall is an inlet
	int yTopInlet = 0;		//!< Whether the y-top wall is an inlet


	Node3D *frontLattice;			//!< Front lattice - the one currently drawn at end of each frame
	Node3D *backLattice;			//!< Back lattice - the one to which we prepare next frame to be drawn

	Node3D *d_frontLattice;			//!< Device pointer for the front lattice
	Node3D *d_backLattice;			//!< Device pointer for the back lattice

	glm::vec3 *velocities;			//!< Velocities vector for the lattice
	glm::vec3 *d_velocities;		//!< Device pointer to the velocities vector for the lattice


	HeightMap *heightMap;			//!< Heightmap for the simulation
	float *d_heightMap;				//!< Device pointer to heightmap for the simulation


	struct cudaGraphicsResource *cudaParticleVerticesVBO;	//!< Device pointer to the particle vertices VBO
	struct cudaGraphicsResource *cudaParticleColorsVBO;		//!< Device pointer to the particle colors VBO

	//! Default constructor.
	LBM3D_1D_indices();

	//! Constructs and initializes the LBM simulator.
	/*!
		Allocates data on GPU and initializes the lattice, grids and uploads all data to GPU.
		This includes reading the heightmap and sending it to the simulator.
		\param[in] vars				VariableManager to be used by this simulator.
		\param[in] particleSystem	ParticleSystem that is to be modified/moved by this simulator.
		\param[in] stlpDiagram		STLPDiagram that is used in the STLP simulation.
	*/
	LBM3D_1D_indices(VariableManager *vars, ParticleSystem *particleSystem, STLPDiagram *stlpDiagram);

	//! Frees CPU and GPU memory and unmaps CUDA graphics resources (VBOs).
	virtual ~LBM3D_1D_indices();



	//! Recalculates main simulation variables. Useful when the value of tau has been modified.
	virtual void recalculateVariables();

	//! Reupload heightmap data to the GPU to be used in the simulation.
	virtual void refreshHeightMap();

	//! Starts editing the simulation area and saves the current state.
	virtual void startEditing();

	//! Stops editing the simulation area.
	/*!
		If we do not save the changes, we revert to the saved state.
		\param[in] saveChanges		Whether we want to save the changes that were made.
	*/
	virtual void stopEditing(bool saveChanges);

	//! Saves changes to the simulation area and refreshes the heightmap.
	virtual void saveChanges();

	//! Resets the LBM to the saved previous state (when we started editing).
	virtual void resetChanges();

	//! Returns whether the LBM simulation area is under edit.
	/*!
		\return Whether the LBM simulation area is under edit.
	*/
	bool isUnderEdit();

	//! Draws the visualization grids of the LBM simulation area.
	void draw();

	//! --- DEPRECATED --- Draws the CPU velocity arrows and particle velocity arrows (if defined) and draws the heightmap.
	/*!
		\deprecated CPU version only & heightmap drawing undesired here.
	*/
	virtual void draw(ShaderProgram &shader);

	//! Does one step of the simulation on the CPU.
	/*!
		\see doStepCUDA()
	*/
	virtual void doStep();
	
	//! Does one step of the simulation using GPU device with CUDA capability.
	/*!
		This step can be decomposed into multiple small steps (combination of kernels and regular functions):
		1) clearBackLattice	-	clear back lattice into which we prepare next simulation step
		2) updateInlets		-	update inlet lattice nodes with given velocity
		3) streamingStep	-	stream distribution function values
		4) updateColliders	-	reverse distribution function directions
		5) collisionStep	-	Bhatnagar-Gross-Krook operator
		6) moveParticles	-	move particles using adjacent lattice node velocities
		7) swapLattices		-	swap back and front lattice
	*/
	virtual void doStepCUDA();

	//! Clears the back lattice in preparation for next simulation step.
	virtual void clearBackLattice();

	//! Streaming step updates the distribution function in the back lattice.
	/*!
		The update is based on the distribution function from the previous frame (frontLattice).
	*/
	virtual void streamingStep();


	//! Collision step computes microscopic collisions between particles.
	/*!
		Uses Bhatnagar-Gross-Krook operator for finding equilibrium.
		The macroscopic velocity and density are first computed.
	*/
	virtual void collisionStep();


	//! Moves particles in the direction of the velocity of adjacent lattice nodes.
	/*!
		Uses trilinear interpolation to move the particle based on its 8 adjacent nodes.
	*/
	virtual void moveParticles();

	//! Updates the inlets of the scene to the set inlet velocity.
	/*!
		Macroscopic density is therefore set to 1.0 and the same collision operation is done as in the collision step.
	*/
	virtual void updateInlets();

	//! Updates lattice nodes where there are colliders/obstacles.
	/*!
		Reverses the distribution function directions in each obstacle lattice node.
	*/
	virtual void updateColliders();

	//! Resets the lattice distribution functions to initial values.
	virtual void resetSimulation();

	//! Synchronize CUDA device (wait for GPU to finish).
	virtual void synchronize();

	//! Get width of the simulation area in world unit size.
	float getWorldWidth();
	//! Get height of the simulation area in world unit size.
	float getWorldHeight();
	//! Get depth of the simulation area in world unit size.
	float getWorldDepth();

	//! Snap the simulation area to ground (takes minimum from all 4 bottom corners).
	void snapToGround();

	//! Returns string for the given eRespawnMode enum value.
	const char *getRespawnModeString(int mode);

	//! Returns the model matrix of the simulation area.
	glm::mat4 getModelMatrix();

	//! Returns the model matrix of the previous state (used when drawing both boxes when editing LBM).
	glm::mat4 getPrevStateModelMatrix();

	//! Maps the given VBO (ParticleSystem vertices) to the cudaParticleVerticesVBO pointer.
	/*!
		\param[in] VBO	The VBO to be mapped to cudaParticleVerticesVBO.
	*/
	void mapVBO(GLuint VBO);

protected:

	//! Swaps both the CPU and GPU lattice pointers.
	virtual void swapLattices();

	//! Initializes OpenGL buffers for the CPU version.
	virtual void initBuffers();

	//! Initializes the lattice for the GPU version only!!!
	virtual void initLattice();


	vector<glm::vec3> velocityArrows;	//!< Array describing velocity arrows (starts in node, points in velocity direction) for visualization
	vector<glm::vec3> particleArrows;	//!< Array describing velocity arrows that visualize particle velocity (interpolated values)

	GLuint velocityVBO;		//!< VBO for node velocity visualization
	GLuint velocityVAO;		//!< VAO for node velocity visualization

	GLuint particleArrowsVAO;	//!< VAO for particle velocity (arrow) visualization
	GLuint particleArrowsVBO;	//!< VBO for particle velocity (arrow) visualization



private:

	//! Previous state data of the simulation area box.
	struct PrevStateData {
		glm::vec3 position;		//!< Position of the simulation area
		float scale;			//!< Scale of the simulation area


	};

	PrevStateData prevState;	//!< Previous position/scale state of the simulation area box
	bool editing = false;		//!< Whether the simulation area is under edit

	int frameId = 0;	//!< Frame id for debugging

	int respawnY = 0;	//!< Respawn y coordinate
	int respawnZ = 0;	//!< Respawn z coordinate

	//! Returns flattened index.
	int getIdx(int x, int y, int z) {
		return (x + latticeWidth * (y + latticeHeight * z));
	}

	dim3 blockDim;		//!< Dimension of the CUDA blocks
	dim3 gridDim;		//!< Dimension of the CUDA grid
	int cacheSize;		//!< Size of the shared memory cache (in Bytes!!!)


	//! Calculates macroscopic density of the node at coordinates x, y, z.
	float calculateMacroscopicDensity(int x, int y, int z);

	//! Calculates macroscopic velocity of the node at coordinates x, y, z with the specified macroscopic density.
	glm::vec3 calculateMacroscopicVelocity(int x, int y, int z, float macroDensity);

	//! Save the simulation area box state.
	void saveState();
	//! Reset the simulation area box state to the previous/saved state.
	void resetToPrevState();
};

