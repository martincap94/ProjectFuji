///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       EmitterBrushMode.h
* \author     Martin Cap
*
*	Describes the EmitterBrushMode class that is used to manipulate positional emitters as brushes
*	when the brush mode is turned on.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "VariableManager.h"
#include "TerrainPicker.h"

#include "PositionalEmitter.h"
#include "CircleEmitter.h"

#include <vector>


#include "UserInterface.h"
#include <nuklear.h>

//#include "ParticleSystem.h"
class ParticleSystem;

typedef PositionalEmitter Brush;

//! Helper class for the interactive brush mode.
/*!
	If we are in EmitterBrushMode, the terrain is drawn with special shader to enable pixel perfect terrain picking.
	This class then processes mouse events and uses the active brush (PositionalEmitter) to emit particles.
*/
class EmitterBrushMode {
public:

	int numParticlesEmittedPerFrame = 1000;				//!< Global number of particles emitted per frame for the active brushes
	const int maxNumParticlesEmittedPerFrame = 10000;	//!< Upper bound of particles emitted per frame

	//! Construct the emitter brush mode and prepares the TerrainPicker instance by creating it.
	/*!
		\param[in] vars		VariableManager to be used.
		\param[in] ps		ParticleSystem that will emit the particles.
	*/
	EmitterBrushMode(VariableManager *vars, ParticleSystem *ps);

	//! Destroys the TerrainPicker member object.
	~EmitterBrushMode();

	//! Draws the terrain to auxiliary framebuffer using the custom shader if active.
	void update();

	//! Updates the position of active brush based on screen position of the mouse cursor.
	/*!
		\param[in] x		Screen x coordinate of the mouse cursor.
		\param[in] y		Screen y coordinate of the mouse cursor.
	*/
	void updateMousePosition(float x, float y);

	//! Processes left mouse button press event.
	/*!
		This mainly enables the active brush if it exists.
		\param[in] x		Screen x coordinate of the mouse cursor.
		\param[in] y		Screen y coordinate of the mouse cursor.
	*/
	void onLeftMouseButtonPress(float x, float y);

	//! Processes left mouse button down event.
	/*!
		This mainly enables the active brush if it exists.
		\param[in] x		Screen x coordinate of the mouse cursor.
		\param[in] y		Screen y coordinate of the mouse cursor.
	*/
	void onLeftMouseButtonDown(float x, float y);

	//! Processes left mouse button release event.
	/*!
		This mainly disables the active brush if it exists.
		\param[in] x		Screen x coordinate of the mouse cursor.
		\param[in] y		Screen y coordinate of the mouse cursor.
	*/
	void onLeftMouseButtonRelease(float x, float y);

	//! Processes mouse wheel scroll event.
	/*!
		This gives users option to change number of emitter particles, brush size, or profile indices range.
		The mode is dependent on held down mod key such as control, shift or alt keys.
		\param[in] yoffset		The mouse wheel offset.
		\param[in] glfwMods		Bit mask of the used keyboard mod keys.
	*/
	void processMouseWheelScroll(float yoffset, int glfwMods);

	//! Loads all available brushes (positional emitters) from the bound ParticleSystem.
	/*!
		This is a quick hack - only call this when brush mode turned on!
		Generally, it would maybe make more sense if each ParticleSystem had its own EmitterBrushMode object
		& when a new PositionalEmitter is created, it would be added to brushes.
		Another option is to store PositionalEmitters and more general Emitters separately since only PositionalEmitters
		can be brushes.
	*/
	void loadBrushes();

	//! Sets the active brush and deactivates the previous active brush.
	/*!
		\param in brush		Brush to be set as the active brush.
	*/
	void setActiveBrush(Brush *brush);

	//! Returns the currently active brush.
	/*!
		\return				The currently active brush.
	*/
	Brush *getActiveBrushPtr();
	bool hasActiveBrush();

	//! Returns whether the EmitterBrushMode is active.
	/*!
		\return Whether the EmitterBrushMode object is active.
	*/
	bool isActive();

	//! Sets whether the EmitterBrushMode object is active.
	/*!
		\param[in] active		Whether the EmitterBrushMode object is active.
	*/
	void setActive(bool active);

	//! Toggles whether the EmitterBrushMode is active.
	void toggleActive();

	//! Constructs the brush selection panel for the user interface.
	/*!
		\param[in] ctx		Nuklear context object for which we create the panel.
		\param[in] ui		UserInterface for which this panel is created.
	*/
	void constructBrushSelectionUIPanel(struct nk_context *ctx, UserInterface *ui);

	//! Refreshes the brush mode by refreshing all its subsystems' framebuffers.
	/*!
		Should be called on screen change.
	*/
	void refreshFramebuffers();


private:

	bool active = false;		//!< Whether the brush mode is active


	std::vector<Brush *> brushes;		//!< List of all possible brushes

	Brush *prevActiveBrush = nullptr;	//!< Previous active brush
	Brush *activeBrush = nullptr;		//!< Currently active brush

	ParticleSystem *ps = nullptr;		//!< ParticleSystem that emits the particles generated by the active brush
	VariableManager *vars = nullptr;	//!< VariableManager for this helper object
	TerrainPicker *tPicker = nullptr;	//!< TerrainPicker that can determine world position for mouse cursor on the terrain

	bool terrainHit = false;			//!< Whether the terrain was hit with the mouse cursor in last frame
	glm::vec3 pos = glm::vec3(0.0f);	//!< Current world position of the mouse cursor



};

