#include "PositionalCDFEmitter.h"

#include "ShaderManager.h"
#include "ParticleSystem.h"

#include <nuklear.h>

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
	nkSamplerTexture = nk_image_id(sampler->getTexture()->id);
	
	initBuffers();

	shader = ShaderManager::getShaderPtr("singleColor");
	prevScale = scale;

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


	if (prevPosition != position || prevScale != scale) {
		prevPosition = position;
		prevScale = scale;

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

void PositionalCDFEmitter::changeScale(float scaleChange) {
	scale += scaleChange * sqrt(scale) * 0.2f;
	scale = glm::clamp(scale, 0.001f, 100.0f);
}

bool PositionalCDFEmitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {
	bool canBeConstructed = PositionalEmitter::constructEmitterPropertiesTab(ctx, ui);
	nk_property_float(ctx, "Scale", 0.01f, &scale, 1000.0f, 0.01f, 0.01f);

	//nk_draw_image(ctx->)
	static Texture *selectedTexture = nullptr;

	if (initialized) {
		nk_layout_row_static(ctx, 100, 100, 1);
		nk_button_image(ctx, nkSamplerTexture);
		nk_layout_row_dynamic(ctx, 15, 1);

	} else {
		ui->constructTextureSelection_label(&selectedTexture, "Probability Texture: ", 0.3f, probabilityTexturePath);
		if (selectedTexture != nullptr) {
			probabilityTexturePath = selectedTexture->filename;
		}
		canBeConstructed = canBeConstructed && (selectedTexture != nullptr);

		if (!canBeConstructed) {
			nk_layout_row_dynamic(ctx, 15.0f, 1);
			nk_label_colored(ctx, "Please select a probability texture.", NK_TEXT_LEFT, nk_rgb(255, 150, 150));
		}
	}


	return canBeConstructed;

}

Texture *PositionalCDFEmitter::getSamplerTexture() {
	return sampler->getTexture();
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
