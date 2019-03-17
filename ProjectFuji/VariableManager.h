#pragma once

#include <string>
#include "CommonEnums.h"
#include "HeightMap.h"

#include <glm\glm.hpp>
#include <GLFW\glfw3.h>

#include "Timer.h"



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
	eLBMType lbmType;
	HeightMap *heightMap = nullptr;

	int numParticles = 1000;	///< Number of particles
	int maxNumParticles = 1000000;
	
	string sceneFilename;		///< Filename of the scene
	bool useCUDA = true;		///< Whether to use CUDA or run the CPU version of the application
	int useCUDACheckbox = 1;	///< Helper int value for the UI checkbox


	int windowWidth = 1000;		///< Window width
	int windowHeight = 1000;	///< Window height

	int screenWidth;			///< Screen width
	int screenHeight;			///< Screen height

	int latticeWidth = 100;		///< Default lattice width
	int latticeHeight = 100;	///< Default lattice height
	int latticeDepth = 100;		///< Defailt lattice depth


	float tau = 0.52f;			///< Default tau value

	bool drawStreamlines = false;	///< Whether to draw streamlines - DRAWING STREAMLINES CURRENTLY NOT VIABLE

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

	int stlpUseCUDA = 1;
	int applyLBM = 0;
	int applySTLP = 0;

	int stlpMaxProfiles = 100;

	float opacityMultiplier = 0.03f;

	bool consumeMouseCursor = false;
	
	string soundingFile;		///< Name of the sounding file to be loaded


	int showCloudShadows = 0;

	int dividePrevVelocity = 0;
	float prevVelocityDivisor = 100.1f;

	int showCCLLevelLayer = 0;
	int showELLevelLayer = 0;

	int showOverlayDiagram = 0;

	glm::vec3 tintColor = glm::vec3(1.0f);

	glm::vec3 bgClearColor = glm::vec3(0.0f);

	float terrainTextureTiling = 10.0f;

	int useSoundingWindVelocities = 0;

	float lbmVelocityMultiplier = 1.0f;
	int lbmUseCorrectInterpolation = 0;

	int fullscreen = 0;


	// Particle System variables
	int maxPositionRecalculations;
	float positionRecalculationThreshold;

	// LBM variables (new)
	int useSubgridModel = 0;




	float fogMinDistance = 0.0f;
	float fogMaxDistance = 150.0f;
	float fogIntensity = 0.8f;
	glm::vec4 fogColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

	////////////////////////////////////////////////////////////////
	// Terrain
	////////////////////////////////////////////////////////////////
	int drawGrass = 1;
	int drawTrees = 1;
	float globalNormalMapMixingRatio = 0.2f;


	////////////////////////////////////////////////////////////////
	// UI helpers
	////////////////////////////////////////////////////////////////
	bool debugWindowOpened = false;
	bool aboutWindowOpened = false;

	int toolbarHeight = 20;
	int debugTabHeight = 300;
	int leftSidebarWidth = 250;
	int rightSidebarWidth = 250;
	int debugTextureRes = leftSidebarWidth;


	int prevHideUIKeyState = GLFW_RELEASE;
	int hideUIKey = GLFW_KEY_F;
	int hideUI = 0;

	//int show

	VariableManager();
	~VariableManager();

	//void init();
	void init(int argc, char **argv);


	/// Load configuration file and parse all correct parameters.
	void loadConfigFile();


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



};

