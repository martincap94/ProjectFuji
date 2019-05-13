///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CDFEmitter.h
* \author     Martin Cap
*
*	This file describes the CDFEmitter class that uses CDFSampler class to generate particles in a
*	pattern given by a probability grayscale texture.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Emitter.h"

#include "CDFSampler.h"
#include "DynamicCDFSampler.h"

#include <string>

//! Emitter that uses CDF sampler to emit particles in given patterns.
/*!
	Emitter that uses CDF sampler to emit particles in given patterns. These patterns are defined
	by a grayscale probability texture that is used to initialize the CDFSampler object.
*/
class CDFEmitter : public Emitter {
public:

	//! Default constructor.
	/*!
		Default constructor creates non-initialized emitter!
		This is useful when we want to initialize the emitter later on, e.g. using the copy constructor.
		The main usage is in ParticleSystem where we use uninitialized emitters to feed data to emitter creation wizard.
	*/
	CDFEmitter(); 

	//! Constructs the emitter with the given name.
	/*!
		Constructs the emitter with the given name, sets its owner and creates the sampler member object.
		\param[in] name		Name of the emitter.
		\param[in] owner	Owning particle system.
		\param[in] probabilityTexturePath	Path to the probability texture used in sampler. Should be grayscale image!
		\param[in] useDynamicSampler		Whether to use dynamic CDF sampler that can be later changed.
	*/
	CDFEmitter(std::string name, ParticleSystem *owner, std::string probabilityTexturePath, bool useDynamicSampler = false);

	//! Constructs the emitter by copying an existing non-initialized(!) one.
	/*!
		Constructs the emitter by copying an existing non-initialized(!) one.
		After copying is finished, the emitter is initialized!
		This way of construction is provided for easier implementation of user interface emitter creation wizard.
		\param[in] e		Non-initialized emitter to be copied.
		\param[in] owner	Owning particle system.
	*/
	CDFEmitter(const CDFEmitter &e, ParticleSystem *owner);

	//! Destroys the sampler.
	virtual ~CDFEmitter();


	virtual void init();

	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();

	virtual bool constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

	
protected:

	std::string probabilityTexturePath = "";
	int useDynamicSampler = 0;

	CDFSampler *sampler = nullptr;
	DynamicCDFSampler *dsampler = nullptr; // only set when useDynamicSampler is true - helper so we do not have to cast in each loop

};

