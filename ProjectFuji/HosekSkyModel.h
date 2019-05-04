///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       HosekSkyModel.h
* \author     Martin Cap
*
*	Describes HosekSkyModel class that is used to generate and feed atmosphere visualization
*	on GPU as well as CPU using Hosek-Wilkie's sky model with the provided data.
*	Hosek-Wilkie's model is desribed here: https://cgg.mff.cuni.cz/projects/SkylightModelling/ 
*	This is a C++ reimplementation of Ben Anderson's Rust implementation of the sky model that is
*	available here: https://github.com/benanders/Hosek-Wilkie
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>

#include "ShaderProgram.h"

#include "ArHosekSkyModel.h"
#include "DirectionalLight.h"


//! Class used to generate and feed atmosphere visualization.
/*!
	Works on GPU as well as CPU using Hosek-Wilkie's sky model with the provided data.
	Hosek-Wilkie's model is desribed here: https://cgg.mff.cuni.cz/projects/SkylightModelling/
	This is a C++ reimplementation of Ben Anderson's Rust implementation of the sky model that is
	available here: https://github.com/benanders/Hosek-Wilkie
*/
class HosekSkyModel {
public:

	double turbidity = 4.0;		//!< Turbidity of the sky
	double albedo = 0.5;		//!< Ground albedo that affects the sky

	int liveRecalc = 1;			//!< Whether the model should be recalculated live

	int calcParamMode = 0;		//!< Experimental setting for setting mode of calculation
	int useAndersonsRGBNormalization = 1;	//!< Whether to use normalization that is used in Ben Anderson's Rust implementation

	float elevation;			//!< Elevation of the sun
	double eta = 0.0;			//!< Angle "below" sun (between sun and xz plane)
	double sunTheta = 0.0;		//!< Angle "above" sun (between sun and y plane)
	

	// shader uniforms
	double horizonOffset = 0.01;	//!< Horizon offset that is 0.01 in Hosek's article
	float sunIntensity = 2.5f;		//!< Intensity of sun
	int sunExponent = 512;			//!< Exponent used in sun generation, the higher the exponent, the more "focused" the sun

	//! Initializes buffers and the main shader.
	/*!
		\see initBuffers()
	*/
	HosekSkyModel(DirectionalLight *dirLight);

	//! Default destructor.
	~HosekSkyModel();

	//! Draws the skybox using a cube for rendering.
	/*!
		\param[in] viewMatrix	View matrix used when rendering.
	*/
	void draw(const glm::mat4 &viewMatrix);

	//! Initializes buffers of the cube that is used for rendering.
	void initBuffers();

	//! Updates the parameters of Hosek's simulator.
	/*!
		The parameters are updated only when liveRecalc is enabled and 
		the elevation of the sun was changed.
	*/
	void update();

	//! Returns a color of the atmosphere at given angles.
	/*!
		\param[in] cosTheta		Cosine of theta (from zenith to view direction).
		\param[in] gamma		Gamma angle (between theta and sun theta).
		\param[in] cosGamma		Cosine of gamma.
		\return					Color sample at the specified position.
	*/
	glm::vec3 getColor(float cosTheta, float gamma, float cosGamma);

	//! Returns the color taken from sun's middle.
	/*!
		\return					Color sample taken from sun's middle.
	*/
	glm::vec3 getSunColor();

	//! Returns elevation in degrees.
	float getElevationDegrees();

	// UI helpers
	//! Returns name of the currently used parameter mode.
	std::string getCalcParamModeName();

	//! Returns name of the given parameter mode.
	std::string getCalcParamModeName(int mode);


private:

	glm::vec3 params[10];	//!< Parameters used in Hosek's sky model

	ArHosekSkyModelState *skymodel_state;	//!< Hosek's implementation of the sky state

	double prevEta = 0.0;				//!< Previous eta angle
	double prevTurbidity = turbidity;	//!< Previous turbidity
	double prevAlbedo = albedo;			//!< Previous albedo
	int prevCalcParamMode = calcParamMode;		//!< Previous parameter calculation mode
	int prevUseAndersonsRGBNormalization = useAndersonsRGBNormalization;	//!< Whether we used Anderson's normalization in previous frame

	double telev = 0.0;	//!< Transformed elevation angle (eta)

	ShaderProgram *shader = nullptr;		//!< Shader used for drawing the sky

	DirectionalLight *dirLight = nullptr;	//!< The sun

	GLuint VAO;	//!< VAO of the skybox cube
	GLuint VBO;	//!< VBO of the skybox cube
	GLuint EBO;	//!< EBO of the skybox cube

	//! Returns whether the model should update its parameters.
	/*!
		\param[in] newEta	New eta angle value.
		\return				Whether the Hosek's sky model parameters should be recalculated.
	*/
	bool shouldUpdate(float newEta);

	//! Recalculates the parameters using the given sun direction.
	/*!
		\param[in] sunDir	Direction of the sun.	
	*/
	void recalculateParams(glm::vec3 sunDir);

	// stride only different for radiosity dataset, otherwise always 9
	//! Calculates param using the given dataset and stride.
	double calculateParam(double *dataset, int stride);

	//! Calculates the bezier interpolation using the given dataset, start and stride values.
	double calculateBezier(double *dataset, int start, int stride);

	//! Uses the normalization process shown in Anderson's Rust implementation.
	/*!
		\param[in] sunDir	Direction of the sun.
	*/
	void normalizeRGBParams(glm::vec3 sunDir);

	//! Uses implementation provided by Hosek to recalculate parameters.
	/*!
		\param[in] sunDir	Direction of the sun.
	*/
	void recalculateParamsHosek(glm::vec3 sunDir);


};

