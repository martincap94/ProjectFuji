///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CircleEmitter.h
* \author     Martin Cap
*
*	This file contains the CircleEmitter class. It is a type of positional emitter that generates
*	particles in a circular pattern that is projected onto the ground.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "PositionalEmitter.h"

class ParticleSystem;

//! PositionalEmitter that emits particles in a circle projected onto the terrain.
/*!
	PositionalEmitter that emits particles in a circle projected onto the terrain.
	This emitter can be used as a brush in brush mode.
*/
class CircleEmitter : public PositionalEmitter {
public:

	float radius = 10000.0f;		//!< Radius [m] of the emitter

	int numVisPoints = 120;			//!< Number of visualization points

	//! Default constructor.
	CircleEmitter();

	//! Constructs the emitter with given position and radius.
	/*!
		Constructs the emitter with given position and radius and initializes it.
		\param[in] name			Name of the emitter.
		\param[in] owner		Owning particle system.
		\param[in] position		Initial position of the emitter.
		\param[in] radius		Radius [m] of the emitter.
	*/
	CircleEmitter(std::string name, ParticleSystem *owner, glm::vec3 position = glm::vec3(0.0f), float radius = 1000.0f);

	//! Constructs the emitter by copying an existing non-initialized(!) one.
	/*!
		Constructs the emitter by copying an existing non-initialized(!) one.
		After copying is finished, the emitter is initialized!
		This way of construction is provided for easier implementation of user interface emitter creation wizard.
		\param[in] e		Non-initialized emitter to be copied.
		\param[in] owner	Owning particle system.
	*/
	CircleEmitter(const CircleEmitter &e, ParticleSystem *owner);

	//! Default destructor.
	virtual ~CircleEmitter();


	virtual void init();

	virtual void emitParticle();

	virtual void update();

	virtual void draw();
	virtual void draw(ShaderProgram *shader);

	//! Changes radius of the emitter.
	virtual void changeScale(float scaleChange);

	virtual void initBuffers();

	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

protected:


	float prevRadius;	//!< Previous radius before change
						//!< Used to check if we need to update visualization circle

	//! Update the visualization data and upload it to VBO.
	virtual void updateVBOPoints();



};

