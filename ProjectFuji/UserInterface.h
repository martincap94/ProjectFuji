#pragma once

#define GLFW_INCLUDE_NONE

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm\glm.hpp>

#include "Texture.h"
#include <map>
#include <string>



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

	float prevAvgFPS;
	float prevAvgDeltaTime;

	UserInterface(GLFWwindow *window, VariableManager *vars);
	~UserInterface();

	void draw();

	void constructUserInterface();
	bool isAnyWindowHovered();

private:

	enum eVecNaming {
		DEFAULT = 0, // x, y, z, w
		COLOR, // r, g, b, a
		INDEX, // 0, 1, 2, 3
		TEXTURE, // s, t, p, q
	};

	const char *vecNames[16] = {
		"#x", "#y", "#z", "#w",
		"#r", "$g", "#b", "#a",
		"#0", "#1", "#2", "#3",
		"#s", "#t", "#p", "#q"
	};

	const float leftSidebarBorderWidth = 2.0f;
	float leftSidebarEditButtonRatio[2];


	float leftSidebarWidth;

	//bool streamlinesAvailable = false;
	bool streamlineInitMode = false;

	int uiMode = 4;


	nk_context *ctx;
	struct nk_input *ctx_in;

	std::map<std::string, Texture *> *textures; // general textures of the app

	Texture *editIcon = nullptr;
	Texture *settingsIcon = nullptr;

	struct nk_image nkEditIcon;
	struct nk_image nkSettingsIcon;


	void nk_property_vec2(glm::vec2 &target);
	void nk_property_vec3(glm::vec3 &target);
	void nk_property_vec3(glm::vec3 &target, float min, float max, float step, float pixStep, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);
	void nk_property_vec4(glm::vec4 &target);
	void nk_property_color(glm::vec4 &target);

	void nk_value_vec3(const glm::vec3 &target, std::string label = "", eVecNaming namingConvention = eVecNaming::DEFAULT);

	void constructLeftSidebar();
	void constructRightSidebar();
	void constructHorizontalBar();

	void constructLBMTab();
	void constructLightingTab();
	void constructTerrainTab();
	void constructSkyTab();
	void constructCloudVisualizationTab();
	void constructDiagramControlsTab();
	void constructLBMDebugTab();
	
	void constructDebugTab();

	void constructDirLightPositionPanel();
	void constructFormBoxButtonPanel();


};

