///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       STLPSimulatorCUDA.h
* \author     Martin Cap
* \brief      Describes the STLPSimulatorCUDA class.
*
*	Describes the STLPSimulatorCUDA class that runs the SkewT/LogP simulation on the GPU.
*	The diagram data is obtained from STLPDiagram class.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "STLPDiagram.h"
#include "VariableManager.h"
#include "Particle.h"
#include "Texture.h"
#include "ppmImage.h"

#include <cuda_gl_interop.h>


class HeightMap;
class ParticleSystem;

//! Simulator of Duarte's method that runs the simulation on GPU using CUDA.
/*!
	This simulator runs the optimized cloud simulation of Duarte on GPU.
	It uses (and uploads) diagram curves from STLPDiagram instance provided.
	There is also some legacy code that used to draw the particles from when the STLP
	and LBM simulators were separate entities.
*/
class STLPSimulatorCUDA {
public:

	VariableManager *vars;				//!< VariableManager for this simulator
	STLPDiagram *stlpDiagram;			//!< STLPDiagram from which the curves are read and used
	ParticleSystem *particleSystem;		//!< Pointer to the used ParticleSystem instance
	HeightMap *heightMap;
	ppmImage *profileMap = nullptr; // needs to have the same parameters as the height map (width, height), or at least larger

	float delta_t = 1.0f;				//!< Delta time of the simulator - decides how quickly the particles reach equilibrium


	float groundHeight = 0.0f;			//!< --- NOT USED --- Ground height 
										//!< Used in the old world size mapping system
	float simulationBoxHeight = 20000.0f;	//!< --- NOT USED --- Height of the LBM simulation box
											//!< Used in the old world size mapping system
	float boxTopHeight;	//!< --- NOT USED --- Maximum height of the simulation box

	ShaderProgram *layerVisShader;		//!< Shader for visualizing CCL/LCL and EL levels

	//struct cudaGraphicsResource *cudaParticleVerticesVBO;	//!< CUDA pointer to the particle vertices VBO

	float *d_verticalVelocities;	//!< GPU array of the particle vertical velocities


	glm::vec2 *d_ambientTempCurve;	//!< GPU array that describes the ambient temperature curve
	
	// use flattened arrays (with offsets)
	glm::vec2 *d_dryAdiabatProfiles;	//!< GPU array containing dry adiabat profile curves
	glm::ivec2 *d_dryAdiabatOffsetsAndLengths;		//!< GPU array describing dry adiabat offsets and lengths
													//!< (x denotes how many vertices in total are before the start of this curve
													//!< and y denotes how many vertices is the curve composed of)
	glm::vec2 *d_moistAdiabatProfiles;	//!< GPU array containing moist adiabat profile curves

	glm::ivec2 *d_moistAdiabatOffsetsAndLengths;	//!< GPU array describing moist adiabat offsets and lengths
													//!< (x denotes how many vertices in total are before the start of this curve
													//!< and y denotes how many vertices is the curve composed of)
	glm::vec2 *d_CCLProfiles;	//!< GPU array containing all CCL profiles (temperatures and pressures)
	glm::vec2 *d_TcProfiles;	//!< GPU array containing all Tc profiles (temperatures and pressures)

	dim3 gridDim;	//!< Grid dimension for CUDA kernel calls
	dim3 blockDim;	//!< Block dimension for CUDA kernel calls

	//! Constructs the STLP simulator, creates necessary buffers and member objects.
	/*!
		\param[in] vars				VariableManager for this simulator.
		\param[in] stlpDiagram		STLPDiagram to be used for simulation.
	*/
	STLPSimulatorCUDA(VariableManager *vars, STLPDiagram *stlpDiagram);

	//! Deallocates CPU and GPU array data.
	~STLPSimulatorCUDA();



	//! Initializes buffers and shader uniforms for the particles.
	/*!
		The CCL profiles are uploaded as a regular uniform data.
	*/
	void initBuffers();

	//! Uploads CCL profiles for the given shader.
	/*!
		\param[in] shader	Shader for which the uniform CCL data is to be uploaded.
	*/
	void uploadProfileIndicesUniforms(ShaderProgram *shader);

	//! Initializes the CUDA GPU arrays and uploads the diagram data.
	void initCUDA();

	//! Initializes the CUDA GPU arrays for later use.
	void initCUDAGeneral();

	//! Uploads the diagram data to GPU.
	/*!
		The diagram data is created/processed from the existing diagram.
	*/
	void uploadDataFromDiagramToGPU();

	//! Does single step of the simulation.
	/*!
		Maps the OpenGL resources to CUDA pointers and runs the simulation step kernel.
	*/
	void doStep();

	//void generateParticle();

	//! Draws the visualization levels.
	void draw();


	//void initParticles();

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

	//! Refreshes the visualization levels.
	void refreshLevelVisualizationBuffers();


private:

	GLuint CCLLevelVAO;		//!< VAO for the CCL level visualization
	GLuint CCLLevelVBO;		//!< VBO for the CCL level visualization

	GLuint ELLevelVAO;		//!< VAO for the EL level visualization
	GLuint ELLevelVBO;		//!< VBO for the EL level visualization

	int currAmbientCurveVertexCount;	//!< Vertex count of the ambient curve of the current STLPDiagram instance


};

