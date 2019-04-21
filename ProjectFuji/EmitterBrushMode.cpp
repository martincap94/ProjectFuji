#include "EmitterBrushMode.h"

#include "ParticleSystem.h"
#include <iostream>
#include "Utils.h"

using namespace std;

EmitterBrushMode::EmitterBrushMode(VariableManager * vars, ParticleSystem * ps) : vars(vars), ps(ps) {
	tPicker = new TerrainPicker(vars);
}

EmitterBrushMode::~EmitterBrushMode() {
	if (tPicker) {
		delete tPicker;
	}

}

void EmitterBrushMode::update() {
	if (!active) {
		return;
	}
	tPicker->drawTerrain();
}

void EmitterBrushMode::updateMousePosition(float x, float y) {
	if (!active) {
		return;
	}
	pos = tPicker->getPixelData(x, y, terrainHit);

	if (terrainHit) {

		if (activeBrush) {
			activeBrush->position = pos;
			activeBrush->update();
		}

		//cout << "Terrain hit at pos: ";
		//printVec3(pos);
	}


}

void EmitterBrushMode::onLeftMouseButtonPress(float x, float y) {
	if (!active || !activeBrush) {
		return;
	}
	activeBrush->enabled = 1;

	//if (pos.w == 1.0f) {
	//	((PositionalEmitter*)particleSystem->emitters[0])->position = glm::vec3(pos);
	//}


}

void EmitterBrushMode::onLeftMouseButtonDown(float x, float y) {
	if (!active || !activeBrush) {
		return;
	}
	activeBrush->enabled = 1;



}

void EmitterBrushMode::onLeftMouseButtonRelease(float x, float y) {
	if (!active || !activeBrush) {
		return;
	}
	activeBrush->enabled = 0;
}

void EmitterBrushMode::loadBrushes() {
	brushes.clear();
	for (int i = 0; i < ps->emitters.size(); i++) {
		PositionalEmitter *pe = dynamic_cast<PositionalEmitter *>(ps->emitters[i]);
		if (pe != nullptr) {
			pe->visible = 0;
			brushes.push_back(pe);
			
		}
	}
}

void EmitterBrushMode::setActiveBrush(Brush * brush) {
	prevActiveBrush = activeBrush;
	if (prevActiveBrush) {
		prevActiveBrush->visible = 0;
	}
	activeBrush = brush;
	activeBrush->visible = 1;
	activeBrush->enabled = 0;

	
}

Brush *EmitterBrushMode::getActiveBrushPtr() {
	return activeBrush;
}

bool EmitterBrushMode::hasActiveBrush() {
	return activeBrush != nullptr;
}

bool EmitterBrushMode::isActive() {
	return active;
}

void EmitterBrushMode::setActive(bool active) {
	this->active = active;
	if (this->active) {
		loadBrushes();
		if (activeBrush) {
			setActiveBrush(activeBrush);
		}
	}
}

void EmitterBrushMode::constructBrushSelectionUIPanel(nk_context * ctx, UserInterface * ui) {

	nk_layout_row_dynamic(ctx, 15, 1);

	if (nk_combo_begin_label(ctx, Emitter::getEmitterName(activeBrush), nk_vec2(nk_widget_width(ctx), 500.0f))) {

		nk_layout_row_dynamic(ctx, 15, 1);
		for (int i = 0; i < brushes.size(); i++) {
			if (nk_combo_item_label(ctx, Emitter::getEmitterName(brushes[i]), NK_TEXT_LEFT)) {
				setActiveBrush(brushes[i]);
				nk_combo_close(ctx);
			}
		}
		nk_combo_end(ctx);
	}



}
