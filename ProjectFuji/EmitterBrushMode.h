#pragma once

#include "VariableManager.h"
#include "TerrainPicker.h"

#include "PositionalEmitter.h"
#include "CircleEmitter.h"

#include <vector>
#include <nuklear.h>
#include "UserInterface.h"

//#include "ParticleSystem.h"
class ParticleSystem;

typedef PositionalEmitter Brush;

class EmitterBrushMode {
public:


	EmitterBrushMode(VariableManager *vars, ParticleSystem *ps);
	~EmitterBrushMode();

	void update();
	void updateMousePosition(float x, float y);


	void onLeftMouseButtonPress(float x, float y);
	void onLeftMouseButtonDown(float x, float y);
	void onLeftMouseButtonRelease(float x, float y);

	/*
		This is a quick hack - only call this when brush mode turned on!
		Generally, it would maybe make more sense if each ParticleSystem had its own EmitterBrushMode object
		& when a new PositionalEmitter is created, it would be added to brushes.
		Another option is to store PositionalEmitters and more general Emitters separately since PositionalEmitters
		can be brushes.
	*/
	void loadBrushes();
	void setActiveBrush(Brush *brush);
	Brush *getActiveBrushPtr();
	bool hasActiveBrush();


	bool isActive();
	void setActive(bool active);

	void constructBrushSelectionUIPanel(struct nk_context *ctx, UserInterface *ui);


private:

	bool active = false;

	std::vector<Brush *> brushes;

	Brush *prevActiveBrush = nullptr;
	Brush *activeBrush = nullptr;

	ParticleSystem *ps = nullptr;
	VariableManager *vars = nullptr;
	TerrainPicker *tPicker = nullptr;

	bool terrainHit = false;
	glm::vec3 pos = glm::vec3(0.0f);


};

