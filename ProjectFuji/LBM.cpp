#include "LBM.h"


LBM::LBM() {}

LBM::LBM(glm::ivec3 dimensions, string sceneFilename, float tau, ParticleSystemLBM *particleSystem) : latticeWidth(dimensions.x), latticeHeight(dimensions.y), latticeDepth(dimensions.z), sceneFilename(sceneFilename), tau(tau), particleSystem(particleSystem) {
	itau = 1.0f / tau;
	nu = (2.0f * tau - 1.0f) / 6.0f;
}


LBM::~LBM() {}

void LBM::recalculateVariables() {
	itau = 1.0f / tau;
	nu = (2.0f * tau - 1.0f) / 6.0f;
}
