///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       UserInterface.h
* \author     Martin Cap
*
*	UserInterface class encapsulates user interface generation into one (large) class.
*	The nuklear library is used for UI rendering and creation.
*	Nuklear library was created by Micha Mettke, licensed under Public domain;
*	available here: https://github.com/vurtun/nuklear
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

//#define GLFW_INCLUDE_NONE

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm\glm.hpp>

#include "Texture.h"
#include "CommonEnums.h"

#include <map>
#include <string>

#include "Actor.h"


#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT


#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024

#include <nuklear.h>



class LBM3D_1D_indices;
class ParticleSystem;
class Emitter;
class VariableManager;
class Camera;
class DirectionalLight;
class EVSMShadowMapper;
class STLPDiagram;
class STLPSimulatorCUDA;
class ParticleRenderer;
class HosekSkyModel;
class StreamlineParticleSystem;
class SceneGraph;
class EmitterBrushMode;


//! Class used for UI construction.
/*!
	Class that is used for constructing the main user interface of the application.
	The nuklear library is used for UI rendering and creation.
	Nuklear library was created by Micha Mettke, licensed under Public domain;
	available here: https://github.com/vurtun/nuklear

	Generates two sidebars and other floating/popup windows.
*/
class UserInterface {
private:

	//! Possible content modes of each sidebar.
	enum eContentMode {
		LBM = 0,			//!< Show LBM tab
		LIGHTING,			//!< Show lighting & shadow controls tab
		TERRAIN,			//!< Show terrain settings tab
		SKY,				//!< Show sky settings tab
		CLOUD_VIS,			//!< Show cloud visualization tab
		DIAGRAM,			//!< Show diagram controls tab
		LBM_DEBUG,			//!< Show LBM debug tab
		SCENE_HIERARCHY,	//!< Show scene hierarchy tab
		EMITTERS,			//!< Show emitter controls tab
		GENERAL_DEBUG,		//!< Show general debug tab
		VIEW,				//!< Show view (camera controls) tab
		PARTICLE_SYSTEM,	//!< Show particle system tab
		PROPERTIES,			//!< Show properties (of selected items) tab
		_NUM_CONTENT_MODES	//!< Number of content modes
	};

	//! Possible states of each tab.
	enum eTabState {
		INACTIVE = 0,	//!< The tab is inactive (hidden)
		ACTIVE = 1		//!< The tab is active (shown)
	};

	//! Possible naming conventions of vectors.
	enum eVecNaming {
		DEFAULT = 0,	//!< Default naming:		x, y, z, w
		COLOR,			//!< Color based naming:	r, g, b, a
		INDEX,			//!< Number indexation:		0, 1, 2, 3
		TEXTURE,		//!< Texture based naming:	s, t, p, q
	};

public:


	LBM3D_1D_indices *lbm = nullptr;
	ParticleSystem *particleSystem = nullptr;
	VariableManager *vars = nullptr;
	Camera *camera = nullptr;
	DirectionalLight *dirLight = nullptr;
	EVSMShadowMapper *evsm = nullptr;
	STLPDiagram *stlpDiagram = nullptr;
	STLPSimulatorCUDA *stlpSimCUDA = nullptr;
	ParticleRenderer *particleRenderer = nullptr;
	HosekSkyModel *hosek = nullptr;
	StreamlineParticleSystem *sps = nullptr;
	SceneGraph *scene = nullptr;
	EmitterBrushMode *ebm = nullptr;
	GLFWwindow *mainWindow = nullptr;


	int viewportMode = eViewportMode::VIEWPORT_3D;


	float prevAvgFPS;
	float prevAvgDeltaTime;

	UserInterface(GLFWwindow *window, VariableManager *vars);
	~UserInterface();

	void draw();

	void constructUserInterface();
	bool isAnyWindowHovered();

	void nk_property_vec2(struct nk_context *ctx, float min, glm::vec2 &target, float max, float step, float pixStep, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);
	void nk_property_vec3(struct nk_context *ctx, float min, glm::vec3 &target, float max, float step, float pixStep, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);

	void nk_property_vec4(struct nk_context *ctx, glm::vec4 &target);
	void nk_property_color_rgb(struct nk_context *ctx, glm::vec3 &target);
	void nk_property_color_rgba(struct nk_context *ctx, glm::vec4 &target);

	void nk_value_vec3(struct nk_context *ctx, const glm::vec3 &target, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);


	void constructTextureSelection(Texture **targetTexturePtr, std::string nullTextureNameOverride = "");

	void nk_property_string(struct nk_context *ctx, std::string &target, char *buffer, int bufferLength, int &length);

	void setButtonStyle(struct nk_context *ctx, bool active = true);

	void refreshWidgets();

	// All this functionality should be moved to special Window class, no time to do that now...
	void setFullscreen(bool useFullscreen);


private:

	struct nk_style_button activeButtonStyle;
	struct nk_style_button inactiveButtonStyle;


	const char *vecNames[16] = {
		"#x", "#y", "#z", "#w",
		"#r", "#g", "#b", "#a",
		"#0", "#1", "#2", "#3",
		"#s", "#t", "#p", "#q"
	};

	const float leftSidebarBorderWidth = 2.0f;
	float leftSidebarEditButtonRatio[2];


	const float selectionTabHeight = 65.0f;

	float leftSidebarWidth;

	//bool streamlinesAvailable = false;
	bool streamlineInitMode = false;

	int leftSidebarContentMode = CLOUD_VIS;
	int rightSidebarContentMode = PROPERTIES;

	bool aboutWindowOpened = false;
	bool terrainGeneratorWindowOpened = false;
	bool emitterCreationWindowOpened = false;
	bool saveParticlesWindowOpened = false;
	bool loadParticlesWindowOpened = false;

	int selectedEmitterType = 0;

	const float hudWidth = 100;
	const float hudHeight = 200;
	struct nk_rect hudRect;


	int hierarchyIdCounter = 0;
	std::vector<Actor *> activeActors;

	nk_context *ctx;
	struct nk_input *ctx_in;

	std::map<std::string, Texture *> *textures; // general textures of the app

	Texture *editIcon = nullptr;
	Texture *settingsIcon = nullptr;

	struct nk_image nkEditIcon;
	struct nk_image nkSettingsIcon;

	void constructLeftSidebar();
	void constructRightSidebar();
	void constructHorizontalBar();
	void constructSidebarSelectionTab(int *contentModeTarget, float xPos, float width);
	void constructSelectedContent(int contentMode);

	void constructLBMTab();
	void constructLightingTab();
	void constructTerrainTab();
	void constructTerrainGeneratorWindow();
	void constructSkyTab();
	void constructCloudVisualizationTab();
	void constructDiagramControlsTab();
	void constructLBMDebugTab();
	void constructSceneHierarchyTab();
	void addSceneHierarchyActor(Actor *actor);

	void constructEmittersTab();
	void constructEmitterCreationWindow();

	void constructGeneralDebugTab();
	void constructPropertiesTab();

	void constructViewTab();

	void constructSaveParticlesWindow();
	void constructLoadParticlesWindow();


	void constructDebugTab();
	void constructFavoritesMenu();

	void constructDirLightPositionPanel();
	void constructFormBoxButtonPanel();

	void constructHUD();



	// small quick functions
	void constructTauProperty();
	void constructWalkingPanel();

	const char *tryGetTextureFilename(Texture *tex, std::string nullTextureName = "");

	void openPopupWindow(bool &target);
	void closeAllPopupWindows();


};

