#include "PositionalCDFEmitter.h"

#include "ShaderManager.h"
#include "ParticleSystem.h"

PositionalCDFEmitter::PositionalCDFEmitter() {
}

PositionalCDFEmitter::PositionalCDFEmitter(ParticleSystem * owner, std::string probabilityTexturePath) : PositionalEmitter(owner), probabilityTexturePath(probabilityTexturePath) {
	init();
}

PositionalCDFEmitter::PositionalCDFEmitter(const PositionalCDFEmitter & e, ParticleSystem * owner) : PositionalEmitter(e, owner) {
	probabilityTexturePath = e.probabilityTexturePath;
	init();
}


PositionalCDFEmitter::~PositionalCDFEmitter() {
	if (sampler) {
		delete sampler;
	}
}

void PositionalCDFEmitter::init() {
	sampler = new CDFSampler(probabilityTexturePath);
	initBuffers();

	shader = ShaderManager::getShaderPtr("singleColor");

}

void PositionalCDFEmitter::emitParticle() {

	if (!canEmitParticle()) {
		return;
	}

	Particle p;
	glm::ivec2 sample = sampler->getSample();
	p.position.x = position.x + sample.x;
	p.position.z = position.z + sample.y;
	p.position.y = heightMap->getHeight(p.position.x, p.position.z);


	//p.position = /*sampler->getSample();*/

	p.profileIndex = getRandomProfileIndex();
	p.velocity = glm::vec3(0.0f);

	owner->pushParticleToEmit(p);
}

void PositionalCDFEmitter::update() {
	PositionalEmitter::update();


	if (prevPosition != position) {
		prevPosition = position;

		if (visible) {
			updateVBOPoints();
		}
	}
}

void PositionalCDFEmitter::draw() {
	if (!visible) {
		return;
	}
	shader->use();
	shader->setColor(glm::vec3(0.0f, 1.0f, 1.0f));
	glBindVertexArray(VAO);
	glPointSize(4.0f);
	glDrawArrays(GL_POINTS, 0, numVisPoints);
}

void PositionalCDFEmitter::draw(ShaderProgram * shader) {
}

void PositionalCDFEmitter::initBuffers() {
	Emitter::initBuffers();

	updateVBOPoints();

}

void PositionalCDFEmitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {
	PositionalEmitter::constructEmitterPropertiesTab(ctx, ui);
	Texture *selectedTexture = nullptr;
	ui->constructTextureSelection(&selectedTexture, probabilityTexturePath);
	if (selectedTexture != nullptr) {
		probabilityTexturePath = selectedTexture->filename;
	}
}

void PositionalCDFEmitter::updateVBOPoints() {
	vector<glm::vec3> vertices;

	glm::vec3 v;
	v.x = position.x;
	v.z = position.z;
	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);

	v.x += sampler->getWidth();
	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);

	v.z += sampler->getHeight();
	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);

	v.x -= sampler->getWidth();
	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);


	glNamedBufferData(VBO, sizeof(glm::vec3) * numVisPoints, vertices.data(), GL_STATIC_DRAW);
}
