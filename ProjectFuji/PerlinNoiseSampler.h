///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       PerlinNoiseSampler.h
* \author     Martin Cap
*
*	This file describes the 3D perlin noise sampler that provides both static and instance options.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>
#include <string>

#include <nuklear.h>

//! Basic 3D perlin noise sampler.
/*!
	Extends Perlin's basic implementation of his improved noise with octaves, turbulency, 
	persistence & initial frequency settings.
	WARNING: Do not forget to load the permutations data before usage!!!

	The sampler can be used as a static object or an instance can be created.
	Instances are useful for saving the settings for sampling terrains or emitters for example.
	Based on the improved perlin noise: https://mrl.nyu.edu/~perlin/noise/
*/
class PerlinNoiseSampler {
public:

	//! Possible sampling modes.
	enum eSamplingMode {
		BASIC = 0,		//!< Samples are in range [-1, 1] as in the original article
		NORMALIZED,		//!< Samples are in range [0, 1] (moved x*0.5 + 0.5)
		TURBULENT,		//!< Samples are in range [0, 1] (by using abs(x))
		_NUM_MODES		//!< Number of sampling modes

	};

	float frequency = 1.0f;			//!< Initial frequency (or just generally frequency for 1 octave)
	int numOctaves = 1;				//!< Number of octaves to be used
	float persistence = 0.5f;		//!< Persistence of octaves (multiplies intensity of each octave).
	int samplingMode = NORMALIZED;	//!< Selected sampling mode

	//! Default constructor.
	PerlinNoiseSampler();

	//! Default destructor.
	~PerlinNoiseSampler();


	///////////////////////////////////////////////////////////////////////////////////////////////
	// INSTANCE FUNCTIONS
	///////////////////////////////////////////////////////////////////////////////////////////////

	//! Returns sample for the given position.
	/*!
		\see getSampleStatic()
	*/
	float getSample(glm::vec3 pos);

	//! Returns sample for the given position.
	/*!
		\see getSampleStatic()
	*/
	float getSample(float x, float y, float z);

	//! Returns sample for the given position using multiple octaves.
	/*!
		\see getSampleOctavesStatic()
	*/
	float getSampleOctaves(glm::vec3 pos);

	//! Returns sample for the given x, y, z coordinates using multiple octaves.
	/*!
		\see getSampleOctavesStatic()
	*/
	float getSampleOctaves(float x, float y, float z);


	///////////////////////////////////////////////////////////////////////////////////////////////
	// STATIC FUNCTIONS
	///////////////////////////////////////////////////////////////////////////////////////////////

	//! Returns a sample for the given position.
	/*!
		\param[in] pos				Position for which we want to generate the sample.
		\param[in] frequency		Frequency with which we want to generate the sample.
		\param[in] samplingMode		Sampling mode to be used (BASIC, NORMALIZED, TURBULENT).
		\return						The generated sample.
	*/
	static float getSampleStatic(glm::vec3 pos, float frequency = 1.0f, int samplingMode = 1);

	//! Returns a sample for the given x, y, z position.
	/*!
		Beware that when x and y are integers, this noise function always returns 0.0!
		\param[in] x				x coordinate value.
		\param[in] y				y coordinate value.
		\param[in] z				z coordinate value.
		\param[in] frequency		Frequency with which we want to generate the sample.
		\param[in] samplingMode		Sampling mode to be used (BASIC, NORMALIZED, TURBULENT).
		\return						The generated sample.
	*/	
	static float getSampleStatic(float x, float y, float z, float frequency = 1.0f, int samplingMode = 1);

	//! Returns a sample composed of multiple octaves for the given x, y, z position.
	/*!
		Beware that when x and y are integers, this noise function always returns 0.0!
		\param[in] x				x coordinate value.
		\param[in] y				y coordinate value.
		\param[in] z				z coordinate value.
		\param[in] startFrequency	Frequency with which we want to start (generate the first octave).
		\param[in] numOctaves		Number of octaves to be used.
		\param[in] persistence		Multiplies the intensity of each subsequent octave (the lower the persistence, the less high frequency noise is visible).
		\param[in] samplingMode		Sampling mode to be used (BASIC, NORMALIZED, TURBULENT).
		\return						The generated sample.
	*/
	static float getSampleOctavesStatic(float x, float y, float z, float startFrequency = 1.0f, int numOctaves = 1, float persistence = 0.5f, int samplingMode = 1);

	//! Returns a sample composed of multiple octaves for the given position.
	/*!
		\param[in] pos				Position for which we want to generate the sample.
		\param[in] startFrequency	Frequency with which we want to start (generate the first octave).
		\param[in] numOctaves		Number of octaves to be used.
		\param[in] persistence		Multiplies the intensity of each subsequent octave (the lower the persistence, the less high frequency noise is visible).
		\param[in] samplingMode		Sampling mode to be used (BASIC, NORMALIZED, TURBULENT).
		\return						The generated sample.
	*/
	static float getSampleOctavesStatic(glm::vec3 pos, float startFrequency = 1.0f, int numOctaves = 1, float persistence = 0.5f, int samplingMode = 1);


	//! Loads the permutations data from a file.
	/*!
		This must be called before we do any sampling!
		Thanks to this, we can change the permutations data to use different noise types (not used at the moment).
		\param[in] filename		Filename of the permutations data file.
	*/
	static void loadPermutationsData(std::string filename);

	//! Constructs the properties tab for the user interface.
	/*!
		\param[in] ctx		Nuklear ctx for which this tab is being constructed.
	*/
	void constructUIPropertiesTab(struct nk_context *ctx);

	//! Returns the string for the current sampling mode (of this instance).
	const char *getSamplingModeString();

	//! Returns the string for the provided sampling mode.
	static const char *getSamplingModeString(int samplingMode);

private:

	static int p[512];	//!< Permutations data

	//! Fade function used in Perlin's improved noise, taken from Perlin's Java source code.
	static float fade(float t);
	//! Basic linear interpolation between a and b.
	static float lerp(float t, float a, float b);
	//! Gradient function used in Perlin's improved noise, taken from Perlin's Java source code.
	static float grad(int hash, float x, float y, float z);


};

