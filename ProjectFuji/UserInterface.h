#pragma once

#define GLFW_INCLUDE_NONE

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm\glm.hpp>

#include "Texture.h"
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

//struct nk_context;
//struct nk_image;

class UserInterface {
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

	float prevAvgFPS;
	float prevAvgDeltaTime;

	UserInterface(GLFWwindow *window, VariableManager *vars);
	~UserInterface();

	void draw();

	void constructUserInterface();
	bool isAnyWindowHovered();

private:

	enum eContentMode {
		LBM = 0,
		LIGHTING,
		TERRAIN,
		SKY,
		CLOUD_VIS,
		DIAGRAM,
		LBM_DEBUG,
		SCENE_HIERARCHY,
		EMITTERS,
		GENERAL_DEBUG,
		PROPERTIES

	};

	enum eVecNaming {
		DEFAULT = 0, // x, y, z, w
		COLOR, // r, g, b, a
		INDEX, // 0, 1, 2, 3
		TEXTURE, // s, t, p, q
	};

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

	int hierarchyIdCounter = 0;
	std::vector<Actor *> activeActors;

	nk_context *ctx;
	struct nk_input *ctx_in;

	std::map<std::string, Texture *> *textures; // general textures of the app

	Texture *editIcon = nullptr;
	Texture *settingsIcon = nullptr;

	struct nk_image nkEditIcon;
	struct nk_image nkSettingsIcon;

	void nk_property_vec2(glm::vec2 &target, float min, float max, float step, float pixStep, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);
	void nk_property_vec3(glm::vec3 &target, float min, float max, float step, float pixStep, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);

	void nk_property_vec4(glm::vec4 &target);
	void nk_property_color(glm::vec4 &target);

	void nk_value_vec3(const glm::vec3 &target, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);

	void constructLeftSidebar();
	void constructRightSidebar();
	void constructHorizontalBar();
	void constructSidebarSelectionTab(int *contentModeTarget, float xPos, float width);
	void constructSelectedContent(int contentMode);

	void constructLBMTab();
	void constructLightingTab();
	void constructTerrainTab();
	void constructSkyTab();
	void constructCloudVisualizationTab();
	void constructDiagramControlsTab();
	void constructLBMDebugTab();
	void constructSceneHierarchyTab();
	void addSceneHierarchyActor(Actor *actor);

	void constructEmittersTab();
	void constructGeneralDebugTab();
	void constructPropertiesTab();

	
	void constructDebugTab();
	void constructFavoritesMenu();

	void constructDirLightPositionPanel();
	void constructFormBoxButtonPanel();

	void constructTextureSelection(Texture **targetTexturePtr);


	// small quick functions
	void constructTauProperty();
	void constructWalkingPanel();

	std::string tryGetTextureFilename(Texture *tex);


};

