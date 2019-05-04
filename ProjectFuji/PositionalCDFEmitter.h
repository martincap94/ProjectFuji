///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       PositionalCDFEmitter.h
* \author     Martin Cap
*
*	Describes the PositionalCDFEmitter class. It is a type of positional emitter that uses CDF
*	sampler to create patterns based on probability texture. It can also be used as a general
*	brush that can draw particles on terrain when in brush mode.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "PositionalEmitter.h"
#include "CDFSampler.h"
#include "ShaderProgram.h"
#include "Texture.h"

#include "UserInterface.h"

#include <nuklear.h>

#include <string>

class ParticleSystem;

//! Special positional emitter that uses CDFSampler to generate any pattern on the ground.
class PositionalCDFEmitter : public PositionalEmitter {
public:

	std::string probabilityTexturePath = "";	//!< Path to the probability texture
	const int numVisPoints = 4;		//!< Number of visualization points


	float scale = 1.0f;		//!< Scale of the emitter (relative scale of the texture)
	int centered = 1;		//!< Whether the mouse is centered inside the emitter when in brush mode

	//! Default constructor.
	/*!
		Default constructor creates non-initialized emitter!
		This is useful when we want to initialize the emitter later on, e.g. using the copy constructor.
		The main usage is in ParticleSystem where we use uninitialized emitters to feed data to emitter creation wizard.
	*/
	PositionalCDFEmitter();

	//! Constructs the emitter with the given name.
	/*!
		Constructs the emitter with the given name, sets its owner and creates the sampler member object.
		\param[in] name		Name of the emitter.
		\param[in] owner	Owning particle system.
		\param[in] probabilityTexturePath	Path to the probability texture used in sampler. Should be grayscale image!
	*/
	PositionalCDFEmitter(std::string name, ParticleSystem *owner, std::string probabilityTexturePath);

	//! Constructs the emitter by copying an existing non-initialized(!) one.
	/*!
		Constructs the emitter by copying an existing non-initialized(!) one.
		After copying is finished, the emitter is initialized!
		This way of construction is provided for easier implementation of user interface emitter creation wizard.
		\param[in] e		Non-initialized emitter to be copied.
		\param[in] owner	Owning particle system.
	*/
	PositionalCDFEmitter(const PositionalCDFEmitter &e, ParticleSystem *owner);

	//! Destroys the sampler.
	~PositionalCDFEmitter();


	virtual void init();

	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();

	virtual void changeScale(float scaleChange);


	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

	//! Returns the sampler probability texture.
	Texture *getSamplerTexture();

protected:

	float prevScale;	//!< Saved previous scale of the emitter

	CDFSampler *sampler = nullptr;		//!< CDFSampler used to generate particles

	struct nk_image nkSamplerTexture;	//!< Helper image for UI


	//! Updates the VBO containing visualization points for this emitter.
	virtual void updateVBOPoints();


};

