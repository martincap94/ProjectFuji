#include "PositionalCDFEmitter.h"

#include "ShaderManager.h"
#include "ParticleSystem.h"

PositionalCDFEmitter::PositionalCDFEmitter() {
}

PositionalCDFEmitter::PositionalCDFEmitter(string name, ParticleSystem * owner, std::string probabilityTexturePath) : PositionalEmitter(name, owner), probabilityTexturePath(probabilityTexturePath) {
	init();
}

PositionalCDFEmitter::PositionalCDFEmitter(const PositionalCDFEmitter & e, ParticleSystem * owner) : PositionalEmitter(e, owner) {
	probabilityTexturePath = e.probabilityTexturePath;
	scale = e.scale;
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
	glm::vec2 sample = (glm::vec2)sampler->getSample();

	if (centered) {
		sample.x -= sampler->getWidth() / 2.0f;
		sample.y -= sampler->getHeight() / 2.0f;
	}
	sample *= scale;

	p.position.x = position.x + sample.x;
	p.position.z = position.z + sample.y;
	p.position.y = heightMap->getHeight(p.position.x, p.position.z);
	


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
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "Scale", 0.01f, &scale, 1000.0f, 0.01f, 0.01f);
	Texture *selectedTexture = nullptr;
	ui->constructTextureSelection(&selectedTexture, probabilityTexturePath);
	if (selectedTexture != nullptr) {
		probabilityTexturePath = selectedTexture->filename;
	}
}

void PositionalCDFEmitter::updateVBOPoints() {
	vector<glm::vec3> vertices;

	glm::vec3 v;

	float ws = (float)sampler->getWidth() * scale;
	float hs = (float)sampler->getHeight() * scale;

	if (centered) {
		v.x = position.x - (ws / 2.0f);
		v.z = position.z - (hs / 2.0f);
	} else {
		v.x = position.x;
		v.z = position.z;
	}

	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);

	v.x += ws;
	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);

	v.z += hs;
	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);

	v.x -= ws;
	v.y = heightMap->getHeight(v.x, v.z);
	vertices.push_back(v);


	glNamedBufferData(VBO, sizeof(glm::vec3) * numVisPoints, vertices.data(), GL_STATIC_DRAW);
}
