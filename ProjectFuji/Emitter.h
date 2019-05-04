///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Emitter.h
* \author     Martin Cap
*
*	This file declares the abstract Emitter class that is used to generate particles.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include "HeightMap.h"
#include "ShaderProgram.h"
#include "UserInterface.h"

#include <nuklear.h>

#include <random>
#include <string>

class ParticleSystem;

//! Abstract emitter class that is used to generate particles.
/*!
	Abstract emitter class that is used to generate particles.
	We categorize emitters into two main groups: 
		- general (spawn particles across the whole terrain)
		- positional (have position, spawn particles in a given area based on emitter type)
*/
class Emitter {
public:

	//! Describes possible emitter types (used in UI mainly).
	enum eEmitterType {
		CIRCULAR = 0,		//!< Circular (positional) emitter that generates particles in a circle projected onto the ground
		CDF_TERRAIN,		//!< Emitter that uses CDF sampler to generate particles on the terrain
		CDF_POSITIONAL,		//!< Positional emitter that uses CDF sampler to generate particles in an area

		_NUM_EMITTER_TYPES	//!< Number of emitter types
	};

	std::string name;		//!< Name of the emitter

	ParticleSystem *owner;	//!< Owner of the emitter (emitter can belong to only one ParticleSystem at a time)
	HeightMap *heightMap;	//!< Heightmap to be used when generating particles

	ShaderProgram *shader;	//!< Shader that is used to visualize the emitter

	std::random_device rd;	//!< Random device to be used for random number generation
	std::mt19937_64 mt;		//!< Mersenne twister to be used for random number generation
	std::uniform_real_distribution<float> dist;			//!< Distribution with range [0,1]
	std::uniform_real_distribution<float> distRange;	//!< Distribution with range [-1,1]
	std::uniform_int_distribution<int> profileDist;		//!< Distribution for particle profiles

	int minProfileIndex = 0;	//!< Minimum profile index of emitted particles
	int maxProfileIndex = 0;	//!< Maximum profile index of emitted particles

	int maxParticlesToEmit = 0; //!< --- NOT USED (yet) --- Maximum particles emitted (globally) - 0 means unlimited
	int numParticlesToEmitPerStep = 1;	//!< Number of particles to be emitted during an emitParticles call (in single frame)

	// booleans (as integers for Nuklear)
	int initialized = 0;	//!< Whether the emitter is initialized
	int enabled = 0;		//!< Whether the emitter is enabled (emits particles)
	int visible = 0;		//!< Whether the emitter's visualization is visible

	//! Default constructor.
	/*!
		Default constructor creates non-initialized emitter!
		This is useful when we want to initialize the emitter later on, e.g. using the copy constructor.
		The main usage is in ParticleSystem where we use uninitialized emitters to feed data to emitter creation wizard.
	*/
	Emitter();

	//! Constructs the emitter with the given name.
	/*!
		Constructs the emitter with the given name, sets its owner and creates the sampler member object.
		The emitter is initialized at the end of construction.
		\see init()
		\param[in] name		Name of the emitter.
		\param[in] owner	Owning particle system.
	*/
	Emitter(std::string name, ParticleSystem *owner);
	
	//! Constructs the emitter by copying an existing non-initialized(!) one.
	/*!
		Constructs the emitter by copying an existing non-initialized(!) one.
		After copying is finished, the emitter is initialized!
		This way of construction is provided for easier implementation of user interface emitter creation wizard.
		\param[in] e		Non-initialized emitter to be copied.
		\param[in] owner	Owning particle system.
	*/
	Emitter(const Emitter &e, ParticleSystem *owner);

	//! Default destructor.
	~Emitter();

	//! Initializes the emitter.
	/*!
		Initializes the emitter by initializing these variables:
			- uniform distributions
			- heightMap pointer
			- min/max profile indices
			- other helper variables
	*/
	virtual void init();

	//! Whether the emitter can emit particles at the moment.
	__inline__ virtual bool canEmitParticle();

	//! Emit a single particle.
	virtual void emitParticle() = 0;

	//! Emit particles (amount determined by numParticlesToEmit).
	virtual void emitParticles();

	//! Emit the given amount of particles.
	virtual void emitParticles(int numParticles);
	
	//! Update the emitter so it is ready for next emission.
	virtual void update() = 0;

	//! Draw/visualize the emitter.
	virtual void draw();

	//! Draw/visualize the emitter using the given shader.
	virtual void draw(ShaderProgram *shader) = 0;

	//! Initialize buffers that are to be used when visualizing the emitter.
	virtual void initBuffers();

	//! Moves the min and max profile indices by the given amount in one direction.
	void setProfileIndexPos(int changeAmount);

	//! Moves the min and max profile indices toward/away from each other based on the amount.
	void setProfileIndexRange(int changeAmount);

	//! Returns a string for the given emitter type.
	/*!
		Returns a string for the given emitter type.
		\param[in] emitterType	eEmitterType enum (int) value.
	*/
	static const char *getEmitterTypeString(int emitterType);

	//! Constructs the emitter properties tab using the user interface context.
	/*!
		\param[in] ctx		Nuklear context to be used.
		\param[in] ui		User interface that is currently in usage.
	*/
	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

	//! Returns the emitter's name.
	static const char *getEmitterName(Emitter *emitter);


protected:

	// for refresh when UI changes some values
	int prevMinProfileIndex;
	int prevMaxProfileIndex;

	GLuint VAO;
	GLuint VBO;

	inline void updateProfileIndexDistribution();

	inline virtual int getRandomProfileIndex();


};


