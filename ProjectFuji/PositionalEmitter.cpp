#include "PositionalEmitter.h"

#include "ParticleSystem.h"

#include <nuklear.h>

PositionalEmitter::PositionalEmitter() {
}

PositionalEmitter::PositionalEmitter(ParticleSystem * owner, glm::vec3 position) : Emitter(owner), position(position) {
	init();
}

PositionalEmitter::PositionalEmitter(const PositionalEmitter & e, ParticleSystem * owner) : Emitter(e, owner) {
	position = e.position;
	wiggle = e.wiggle;
	xWiggleRange = e.xWiggleRange;
	zWiggleRange = e.zWiggleRange;
	init();
}


PositionalEmitter::~PositionalEmitter() {
}

void PositionalEmitter::init() {
	prevPosition = position;
}


void PositionalEmitter::wigglePosition() {

	position.x += distRange(mt) * xWiggleRange;
	position.z += distRange(mt) * zWiggleRange;

}

void PositionalEmitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {
	cout << "HERE: " << __FILE__ << ":::" << __LINE__ << endl;

	nk_layout_row_dynamic(ctx, 15, 1);
	nk_checkbox_label(ctx, "Wiggle", &wiggle);
	ui->nk_property_vec3(ctx, -1000000.0f, position, 1000000.0f, 1.0f, 1.0f, "Position");

	cout << "HERE: " << __FILE__ << ":::" << __LINE__ << endl;



}
