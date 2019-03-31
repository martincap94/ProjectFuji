#pragma once

#define GLFW_INCLUDE_NONE

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm\glm.hpp>

#include "Texture.h"
#include <map>
#include <string>


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

struct nk_context;

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

	UserInterface(GLFWwindow *window);
	~UserInterface();

	void draw();

	void constructUserInterface();
	bool isAnyWindowHovered();

private:

	//bool streamlinesAvailable = false;
	bool streamlineInitMode = false;

	int uiMode = 4;

	nk_context *ctx;

	std::map<std::string, Texture *> *textures;


	void uivec2(glm::vec2 &target);
	void uivec3(glm::vec3 &target);
	void uivec4(glm::vec4 &target);
	void uicolor(glm::vec4 &target);

	void constructLBMTab();
	void constructLightingTab();
	void constructTerrainTab();
	void constructSkyTab();
	void constructCloudVisualizationTab();
	void constructDiagramControlsTab();
	void constructLBMDebugTab();
	
	void constructDebugTab();


};

