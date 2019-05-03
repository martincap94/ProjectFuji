///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       StreamlineParticleSystem.h
* \author     Martin Cap
*
*	Special particle system that is used for generating streamlines. This is particularly useful
*	for LBM testing.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>
#include <glm\glm.hpp>
#include <glad\glad.h>

#include "ShaderProgram.h"

class LBM3D_1D_indices;
class VariableManager;


//! Particle system used for visualizing streamlines in LBM.
/*!
	Note that this specialized particle system is expected to be only used when LBM active.
	It also assumes that the inlet is the x axis aligned wall of the LBM simulation area.
*/
class StreamlineParticleSystem {

private:

	//! Parameter settings for horizontal streamlines.
	struct HorizontalParametersSettings {
		float xOffset = 0.0f;	//!< Offset on the x axis 
		float yOffset = 0.5f;	//!< Offset on the y axis
	};

	//! Parameter settings for vertical streamlines.
	struct VerticalParametersSettings {
		float xOffset = 0.0f;	//!< Offset on the x axis
		float zOffset = 0.5f;	//!< Offset on the z axis
	};


public:

	VariableManager *vars = nullptr;	//!< VariableManager used by this class
	LBM3D_1D_indices *lbm = nullptr;	//!< LBM simulator for which we want to visualize its streamlines

	int maxNumStreamlines;			//!< Maximum amount of streamlines
	int maxStreamlineLength;		//!< Maximum streamline length (number of its vertices)

	// bools for UI
	int visible = 1;				//!< Whether the streamlines should be visible
	int liveLineCleanup = 1;		//!< Whether we want to cleanup the streamlines live (recommended)

	bool editingHorizontalParameters = false;	//!< Whether we are currently editing the horizontal parameters
	bool editingVerticalParameters = false;		//!< Whether we are currently editing the vertical parameters

	HorizontalParametersSettings hParams;		//!< Horizontal parameters used by this system
	VerticalParametersSettings vParams;			//!< Vertical parameters used by this system


	bool active = false;		//!< Whether this system is active
	bool initialized = false;	//!< Whether this system is initialized and ready to draw

	int frameCounter = 0;		//!< Counts how many frames were drawn during streamline simulation

	GLuint streamlinesVAO;		//!< VAO of the streamlines
	GLuint streamlinesVBO;		//!< VBO of the streamlines

	struct cudaGraphicsResource *cudaStreamlinesVBO = nullptr;	//!< CUDA pointer to the streamlines VBO
	int *d_currActiveVertices;			//!< GPU array of streamline lengths
	int *currActiveVertices = nullptr;	//!< GPU array of last active vertices of each streamline 

	//! Creates the streamline particle system.
	/*!
		\param[in] vars		VariableManager to be used.
		\param[in] lbm		LBM simulator that drives the particle motion.
	*/
	StreamlineParticleSystem(VariableManager *vars, LBM3D_1D_indices *lbm);

	//! Tears down the system by deallocating all the GPU and CPU data.
	~StreamlineParticleSystem();

	//! Draws the streamlines if the system is initialized.
	void draw();

	//! Initializes the streamline particle system.
	void init();

	//! Updates the system.
	void update();

	//! Initializes the streamline OpenGL buffers and maps them to CUDA pointers if necessary.
	void initBuffers();

	//! Allocates the GPU memory.
	void initCUDA();

	//! Tears down the system by freeing all the GPU and CPU memory.
	void tearDown();

	//! Activates the streamlines.
	void activate();

	//! Deactives the streamlines.
	void deactivate();

	//! Resets the streamlines to their original position.
	void reset();

	//! Cleans the lines (hides line edges that should not be shown).
	void cleanupLines();

	//! Creates a horizontal line of particle seeds that will be used to generate the streamlines.
	void setPositionInHorizontalLine();

	//! Creates a vertical line of particle seeds that will be used to generate the streamlines.
	void setPositionInVerticalLine();

	//! --- NOT IMPLEMENTED (yet) --- Set the particle seeds to form a cross (horizontal + vertical line).
	void setPositionCross();

private:

	ShaderProgram *shader = nullptr;	//!< Shader used to draw the streamlines.





};

