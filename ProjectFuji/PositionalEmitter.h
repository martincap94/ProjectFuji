///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       PositionalEmitter.h
* \author     Martin Cap
*
*	PositionalEmitter is a subclass of Emitter that can be positioned anywhere in the 3D world.
*	Positional emitters are also used as brushes when in brush mode.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Emitter.h"

//! Emitter that can be position in the 3D world.
/*!
	These emitters are an important part of our emitter system.
	PositionalEmitter is used as a Brush in EmitterBrushMode.
	These can also wiggle around, which is fun.
*/
class PositionalEmitter : public Emitter {
public:

	glm::vec3 position = glm::vec3(0.0f); //!< Position of the emitter in world space (usually only x and z are utilized)

	int wiggle = 0; //!< When enabled, position of the emitter "wiggles" around (shifts) after each emission iteration by the given offset range

	float xWiggleRange = 0.5f;	//!< Wiggle change on the x axis
	float zWiggleRange = 0.5f;	//!< Wiggle change on the z axis

	//! Default constructor that creates non-initialized emitter!
	/*!
		This is useful when we want to initialize the emitter later on, e.g. using the copy constructor.
		The main usage is in ParticleSystem where we use uninitialized emitters to feed data to emitter creation wizard.
	*/
	PositionalEmitter();

	//! Constructs the emitter with the given name and position.
	/*!
		Constructs the emitter with the given name, sets its owner and creates the sampler member object.
		The emitter is initialized at the end of construction.
		\see init()
		\param[in] name			Name of the emitter.
		\param[in] owner		Owning particle system.
		\param[in] position		Initial position of the emitter.
	*/
	PositionalEmitter(std::string name, ParticleSystem *owner, glm::vec3 position = glm::vec3(0.0f));

	//! Constructs the emitter by copying an existing non-initialized(!) one.
	/*!
		Constructs the emitter by copying an existing non-initialized(!) one.
		After copying is finished, the emitter is initialized!
		This way of construction is provided for easier implementation of user interface emitter creation wizard.
		\param[in] e		Non-initialized emitter to be copied.
		\param[in] owner	Owning particle system.
	*/
	PositionalEmitter(const PositionalEmitter &e, ParticleSystem *owner);

	//! Default destructor.
	~PositionalEmitter();

	//! Sets the previous position to current position of the emitter.
	virtual void init();

	//! Updates the emitter (wiggle position update).
	virtual void update();

	//! Moves the emitter by random amount from [0, x/zWiggleRange] in x and z directions.
	virtual void wigglePosition();

	//! Change the size of this emitter.
	virtual void changeScale(float scaleChange) = 0;

	// inherited doc
	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);


protected:

	glm::vec3 prevPosition;		//!< Previous position before last position change

};


