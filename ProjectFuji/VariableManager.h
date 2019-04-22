#pragma once

#include <string>
#include "CommonEnums.h"
#include "HeightMap.h"

#include <glm\glm.hpp>
#include <GLFW\glfw3.h>
#include <filesystem>


#include "Timer.h"

class MainFramebuffer;



class VariableManager {

public:
	/*
	namespace Window {
		int screenWidth;
		int screenHeight;
	}
	*/
	Timer timer;


	int vsync = 0;
	HeightMap *heightMap = nullptr;

	int numParticles = 1000;	///< Number of particles
	int maxNumParticles = 1000000;
	
	string sceneFilename;		///< Filename of the scene


	int windowWidth = 1000;		///< Window width
	int windowHeight = 1000;	///< Window height
	bool useMonitorResolution = true;
	bool fullscreen = false;


	int screenWidth;			///< Screen width
	int screenHeight;			///< Screen height

	int latticeWidth = 100;		///< Default lattice width
	int latticeHeight = 100;	///< Default lattice height
	int latticeDepth = 100;		///< Defailt lattice depth
	float latticeScale = 100.0f;

	int terrainXOffset = 0;
	int terrainZOffset = 0;


	float tau = 0.52f;			///< Default tau value

	bool drawStreamlines = false;	///< Whether to draw streamlines - DRAWING STREAMLINES CURRENTLY NOT VIABLE
	int maxNumStreamlines = 100;
	int maxStreamlineLength = 1000;




	int paused = 0;				///< Whether the simulation is paused
	int usePointSprites = 1;	///< Whether to use point sprites for point visualization
	bool appRunning = true;		///< Helper boolean to stop the application with the exit button in the user interface
	float cameraSpeed = DEFAULT_CAMERA_SPEED;	///< Movement speed of the main camera

	int useFreeRoamCamera = 1;

	int blockDim_2D = 256;		///< Block dimension for 2D LBM
	int blockDim_3D_x = 32;		///< Block x dimension for 3D LBM
	int blockDim_3D_y = 2;		///< Block y dimension for 3D LBM

	bool measureTime = false;	///< Whether the time of simulation steps should be measured
	int avgFrameCount = 1000;	///< Number of frames for which we take time measurement average
	bool exitAfterFirstAvg = false;		///< Whether the application quits after the first average time measurement has finished

	int simulateSun = 0;

	//int stlpUseCUDA = 1;
	int applyLBM = 0;
	int applySTLP = 0;

	int stlpMaxProfiles = 100;

	float opacityMultiplier = 0.03f;

	bool consumeMouseCursor = false;
	
	string soundingFile;		///< Name of the sounding file to be loaded


	int cloudsCastShadows = 1;
	float cloudCastShadowAlphaMultiplier = 1.0f;


	int dividePrevVelocity = 0;
	float prevVelocityDivisor = 100.1f;

	int showCCLLevelLayer = 0;
	int showELLevelLayer = 0;

	int showOverlayDiagram = 0;

	glm::vec3 tintColor = glm::vec3(1.0f);

	glm::vec3 bgClearColor = glm::vec3(0.0f);


	int useSoundingWindVelocities = 0;

	float lbmVelocityMultiplier = 1.0f;
	int lbmUseCorrectInterpolation = 0;



	int useSkySunColor = 1;
	float skySunColorTintIntensity = 0.5f;

	// Particle System variables
	int maxPositionRecalculations;
	float positionRecalculationThreshold;

	// LBM variables (new)
	int useSubgridModel = 0;
	int lbmUseExtendedCollisionStep = 0;
	//int

	glm::vec3 latticePosition;

	int lbmStepFrame = 1;
	int stlpStepFrame = 1;

	int drawSkybox = 1;
	int hosekSkybox = 1;


	int terrainPickerMode = 0;
	int terrainUsesPBR = 1;



	int fogMode = eFogMode::LINEAR;
	float fogExpFalloff = 0.01f;
	float fogMinDistance = 5000.0f;
	float fogMaxDistance = 100000.0f;
	float fogIntensity = 0.25f;
	glm::vec4 fogColor = glm::vec4(0.05f, 0.05f, 0.08f, 1.0f);

	////////////////////////////////////////////////////////////////
	// Terrain
	////////////////////////////////////////////////////////////////
	int visualizeTerrainNormals = 0;
	float globalNormalMapMixingRatio = 0.2f;
	float texelWorldSize = 10.0f;
	glm::vec2 terrainHeightRange = glm::vec2(800.0f, 3700.0f);


	////////////////////////////////////////////////////////////////
	// UI helpers
	////////////////////////////////////////////////////////////////
	bool debugWindowOpened = false;
	bool aboutWindowOpened = false;

	int toolbarHeight = 20;
	int debugTabHeight = 300;
	int leftSidebarWidth = 250;
	int rightSidebarWidth = 250;
	int debugOverlayTextureRes = 250;

	int numDebugOverlayTextures = 3;

	int drawOverlayDiagramParticles = 1;


	int hideUIKey = GLFW_KEY_F;
	int hideUI = 0;

	int viewportMode = 0;


	int toggleLBMState = GLFW_KEY_L;
	int toggleSTLPState = GLFW_KEY_K;

	bool generalKeyboardInputEnabled = true;



	int projectionMode = eProjectionMode::PERSPECTIVE;
	float fov = 90.0f;

	int renderMode = 0; // disables rendering of all helper visualization structures (boxes, vectors, etc.)

	std::vector<std::string> sceneFilenames;
	std::vector<std::string> soundingDataFilenames;

	MainFramebuffer *mainFramebuffer = nullptr;


	//int show

	VariableManager();
	~VariableManager();

	//void init();
	void init(int argc, char **argv);


	/// Load configuration file and parse all correct parameters.
	void loadConfigFile();
	void loadSceneFilenames();
	void loadSoundingDataFilenames();

	static std::string getFogModeString(int fogMode);


private:
	bool ready = false;

	/// Prints simple help message for command line usage.
	void printHelpMessage(std::string errorMsg = "");

	/// Parses input arguments of the application and saves them to global variables.
	/**
	Parses input arguments of the application and saves them to global variables. Overwrites settings from config.ini if defined!
	It is important to note that boolean options such as useCUDA ("-c") must be defined using true or false argument value since
	we want to be able to rewrite the configuration values. This means that the approach of: if "-c" then use CUDA, if no argument
	"-c" then do not is not possible. This approach would mean that if "-c" is defined, then we overwrite configuration parameter
	and tell the simulator that we want to use CUDA, but if were to omit "-c" it would not set use CUDA to false, but it would use
	the config.ini value which could be both true or false.

	*/
	void parseArguments(int argc, char **argv);


	/// Parses parameter and its value from the configuration file. Assumes correct format for each parameter.
	void saveConfigParam(std::string param, std::string val);

	void saveIntParam(int &target, std::string stringVal);
	void saveFloatParam(float &target, std::string stringVal);
	void saveVec2Param(glm::vec2 &target, std::string line);
	void saveVec3Param(glm::vec3 &target, std::string line);
	void saveVec4Param(glm::vec4 &target, std::string line);
	void saveBoolParam(bool &target, std::string stringVal);
	void saveIntBoolParam(int &target, std::string stringVal);
	void saveStringParam(string &target, std::string stringVal);


};

