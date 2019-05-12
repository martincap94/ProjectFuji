#include "PositionalEmitter.h"

#include "ParticleSystem.h"

#include <nuklear.h>

PositionalEmitter::PositionalEmitter() {
}

PositionalEmitter::PositionalEmitter(string name, ParticleSystem * owner, glm::vec3 position) : Emitter(name, owner), position(position) {
	init();
}

PositionalEmitter::PositionalEmitter(const PositionalEmitter & e, ParticleSystem * owner) : Emitter(e, owner) {
	position = e.position;
	wiggle = e.wiggle;
	xWiggleRange = e.xWiggleRange;
	zWiggleRange = e.zWiggleRange;
	init();
}

//PositionalEmitter::PositionalEmitter(const PositionalEmitter & e) : Emitter(e) {
//	position = e.position;
//
//	prevPosition = e.prevPosition;
//
//	wiggle = e.wiggle;
//	xWiggleRange = e.xWiggleRange;
//	zWiggleRange = e.zWiggleRange;
//}


PositionalEmitter::~PositionalEmitter() {
}

void PositionalEmitter::init() {
	prevPosition = position;
}

void PositionalEmitter::update() {
	if (enabled) {
		if (wiggle) {
			wigglePosition();
		}
	}
}

void PositionalEmitter::draw() {
	Emitter::draw();
}


void PositionalEmitter::wigglePosition() {

	position.x += distRange(mt) * xWiggleRange;
	position.z += distRange(mt) * zWiggleRange;

}


void PositionalEmitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {
	Emitter::constructEmitterPropertiesTab(ctx, ui);

	nk_checkbox_label(ctx, "Wiggle", &wiggle);
	nk_property_float(ctx, "x wiggle", 1.0f, &xWiggleRange, 1000.0f, 1.0f, 1.0f);
	nk_property_float(ctx, "z wiggle", 1.0f, &zWiggleRange, 1000.0f, 1.0f, 1.0f);

	ui->nk_property_vec3(ctx, -1000000.0f, position, 1000000.0f, 1.0f, 1.0f, "Position");

	

}
