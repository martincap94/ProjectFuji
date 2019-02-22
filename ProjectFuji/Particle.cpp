#include "Particle.h"



Particle::Particle() {
}


Particle::~Particle() {
}

void Particle::updatePressureVal() {
	pressure = getPressureVal();
}

float Particle::getPressureVal() {
	// based on CRC Handbook of Chemistry and Physics
	return pow(((44331.514f - position.y) / 11880.516f), 1 / 0.1902632f);
}
