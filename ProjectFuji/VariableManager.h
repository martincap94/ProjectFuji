///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       VariableManager.h
* \author     Martin Cap
*
*	Utility class VariableManager holds shared variables for the whole application.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include "CommonEnums.h"
#include "HeightMap.h"

#include <glm\glm.hpp>
#include <GLFW\glfw3.h>
#include <filesystem>


#include "Timer.h"

class MainFramebuffer;


//! Manages and loads variables.
/*!
	Manages and loads variables from configuration file and command line arguments.
	Single instance of this class should be used and passed around for other components
	of the framework to have access to global variables such as screen width and height
	of the window, whether vsync is enabled etc.
	Some of the variables are there just for the initialization process and the objects that
	require these variables copy them for use at runtime.
*/
class VariableManager {

public:


	Timer timer;			//!< Timer object for the application


	int vsync = 0;			//!< Whether V-Sync is turned on or off
	HeightMap *heightMap = nullptr;	//!< Heightmap for the whole scene

	int numParticles = 1000;	//!< Number of particles used (i.e. max number of variables)
	
	string sceneFilename;		//!< Filename of the scene to be loaded initiallly


	int windowWidth = 1000;		//!< Window width - used when switching back to windowed mode
	int windowHeight = 1000;	//!< Window height - used when switching back to windowed mode
	bool useMonitorResolution = true;	//!< Whether to use monitor resolution at startup (instead of window size)
	bool fullscreen = false;			//!< Whether the window is shown in fullscreen


	int screenWidth;				//!< Screen width
	int screenHeight;				//!< Screen height

	int latticeWidth = 100;			//!< Default lattice width
	int latticeHeight = 100;		//!< Default lattice height
	int latticeDepth = 100;			//!< Default lattice depth
	float latticeScale = 100.0f;	//!< Default lattice scale (how many meters is one unit cell)

	int terrainXOffset = 0;			//!< Offset of the terrain on the x axis
	int terrainZOffset = 0;			//!< Offset of the terrain on the z axis


	float tau = 0.52f;				//!< Default tau value

	bool drawStreamlines = false;	//!< --- DEPRECATED --- Whether to draw streamlines
	
	// These two below are from the new streamline system
	int maxNumStreamlines = 100;	//!< Maximum number of streamlines shown
	int maxStreamlineLength = 1000;	//!< Maximum streamline length (number of vertices of the created line)




	int paused = 0;				//!< Whether the simulation is paused
	bool appRunning = true;		//!< Helper boolean to stop the application with the exit button in the user interface
	float cameraSpeed = DEFAULT_CAMERA_SPEED;	//!< Movement speed of the main camera

	int useFreeRoamCamera = 1;

	int blockDim_2D = 256;		//!< --- DEPRECATED --- Block dimension for 2D LBM
	int blockDim_3D_x = 32;		//!< Block x dimension for 3D LBM
	int blockDim_3D_y = 2;		//!< Block y dimension for 3D LBM

	bool measureTime = false;	//!< Whether the time of simulation steps should be measured
	int avgFrameCount = 1000;	//!< Number of frames for which we take time measurement average
	bool exitAfterFirstAvg = false;		//!< Whether the application quits after the first average time measurement has finished

	int simulateSun = 0;		//!< Whether the simple sun simulation is running

	int applyLBM = 0;			//!< Whether LBM is running and applied to particles
	int applySTLP = 0;			//!< Whether STLP is running and applied to particles

	int stlpMaxProfiles = 100;	//!< Maximum number of profiles used in STLP

	float opacityMultiplier = 0.03f;	//!< Opacity multiplier of the drawn particles

	bool consumeMouseCursor = false;	//!< Whether the window consumes (hides) the mouse cursor
	
	string soundingFile;		//!< Name of the sounding file to be loaded at startup


	int cloudsCastShadows = 1;	//!< Whether clouds cast shadows
	float cloudCastShadowAlphaMultiplier = 1.0f;	//!< Multiplier of the cloud cast shadows


	int dividePrevVelocity = 1;			//!< Whether to divide velocity from previous step (used for artificial damping in STLP)
	float prevVelocityDivisor = 102.0f; //!< Velocity divisor used (x100) when artifical velocity damping is turned on in STLP simulation

	int showCCLLevelLayer = 0;			//!< Whether to visualize CCL level in the 3D viewport
	int showELLevelLayer = 0;			//!< Whether to visualize EL level in the 3D viewport

	int showOverlayDiagram = 0;			//!< Whether the overlay diagram is to be shown

	glm::vec3 tintColor = glm::vec3(1.0f);		//!< Tint color of the particles
	glm::vec3 bgClearColor = glm::vec3(0.0f);	//!< Background color (if no skybox visible)

	int useSoundingWindVelocities = 0;			//!< --- DEPRECATED --- Whether to use wind velocities from sounding file

	float lbmVelocityMultiplier = 1.0f;			//!< Multiplier of final velocity applied to particles in the LBM simulation
	int lbmUseCorrectInterpolation = 0;			//!< Whether correct (or inverse) interpolation is to be used in LBM

	int useSkySunColor = 1;						//!< Whether the sky is sampled to modify the sun's (DirectionalLight) color
	float skySunColorTintIntensity = 0.5f;		//!< Intensity of the color/tint applied to sun (from sky sample)

	// LBM variables (new)
	int useSubgridModel = 0;					//!< --- EXPERIMENTAL --- Whether to use subgrid model in LBM
	int lbmUseExtendedCollisionStep = 0;		//!< --- EXPERIMENTAL --- Whether to use extended collision step operator in LBM

	glm::vec3 latticePosition;	//!< Position of the bottom left corner of the lattice (LBM simulation area)

	int lbmStepFrame = 1;		//!< How often (in which n-th frame) the LBM is updated and applied
	int stlpStepFrame = 1;		//!< How often the STLP is applied

	int drawSkybox = 1;			//!< Whether to draw any skybox
	int hosekSkybox = 1;		//!< Whether to use Hosek-Wilkie's model when drawing skybox

	int terrainUsesPBR = 1;		//!< Whether the terrain uses PBR shaders (instead of Blinn-Phong)


	int fogMode = eFogMode::LINEAR;		//!< Fog mode used in all shaders
	float fogExpFalloff = 0.01f;		//!< Exponential fog falloff
	float fogMinDistance = 5000.0f;		//!< Linear fog minimum distance (distance at which the fog starts)
	float fogMaxDistance = 100000.0f;	//!< Linear fog maximum distance (distance to which the fog linearly increases)
	float fogIntensity = 0.25f;			//!< General intensity of the fog (how much the fog color mixes with the pixel color)
	glm::vec4 fogColor = glm::vec4(0.05f, 0.05f, 0.08f, 1.0f);	//!< General fog color

	////////////////////////////////////////////////////////////////
	// Terrain
	////////////////////////////////////////////////////////////////
	int visualizeTerrainNormals = 0;			//!< Whether to visualize normals of the terrain (using a geometry shader)
	float globalNormalMapMixingRatio = 0.2f;	//!< How much the global normal map changes normals of the whole terrain (to break repeating patterns)
	float texelWorldSize = 10.0f;				//!< World size in meters of each texel (must be set before generating the terrain)
	glm::vec2 terrainHeightRange = glm::vec2(800.0f, 3700.0f);	//!< Range of heights the terrain can have (must be set before generating the terrain)


	////////////////////////////////////////////////////////////////
	// UI helpers
	////////////////////////////////////////////////////////////////
	bool debugWindowOpened = false;			//!< Whether the debug window is opened
	bool aboutWindowOpened = false;			//!< Whether the about window is opened

	float toolbarHeight = 20.0f;					//!< Height of the toolbar
	float debugTabHeight = 300.0f;				//!< Height of the debug tab
	float leftSidebarWidth = 250.0f;				//!< Width of the left sidebar

	int rightSidebarWidth = 250;			//!< Width of the right sidebar


	int debugOverlayTextureRes = 250;		//!< Resolution of the debug overlay textures (all are square)
	int numDebugOverlayTextures = 3;		//!< Default number of debug overlay textures

	int drawOverlayDiagramParticles = 1;	//!< Draw particles inside the overlay diagram


	int hideUIKey = GLFW_KEY_F;				//!< Key used to hide the UI
	int hideUI = 0;							//!< Whether the UI is hidden

	int viewportMode = 0;					//!< Value of eViewportMode - mode of the viewport

	bool windowMinimized = false;			//!< Whether the window is minimized at the moment (width or height is equal to 0)


	int toggleLBMStateKey = GLFW_KEY_L;		//!< Key that toggles whether LBM is on/off
	int toggleSTLPStateKey = GLFW_KEY_K;	//!< Key that toggles whether STLP simulation is on/off

	bool generalKeyboardInputEnabled = true;	//!< Whether we should listen to keyboard inputs
												//!< This is useful when we are typing in UI, we do not want to listen to the keyboard inputs in our application

	int multisamplingAmount = 12;			//!< Multisampling used in MainFramebuffer for general rendering.


	int projectionMode = eProjectionMode::PERSPECTIVE;	//!< Current projection mode of the 3D viewport
	float fov = 90.0f;									//!< Field of view of the current projection
	float diagramProjectionOffset = 0.2f;				//!< Offset in the diagram projection matrix which determines zoom

	int renderMode = 0;									//!< Whether we should render all helper visualization structures (boxes, vectors, etc.)

	std::vector<std::string> sceneFilenames;			//!< List of filenames that can be scenes (terrains) in our application
	std::vector<std::string> soundingDataFilenames;		//!< List of filenames of possible sounding data

	MainFramebuffer *mainFramebuffer = nullptr;			//!< Pointer to the main framebuffer of the application

	//! Default constructor.
	VariableManager();

	//! Destroys the heightmap.
	~VariableManager();

	//! Initializes the VariableManager with command line arguments.
	void init(int argc, char **argv);


	//! Loads the configuration file and parses all valid parameters.
	void loadConfigFile();

	//! Loads the list of all scene filenames.
	void loadSceneFilenames();

	//! Loads the list of all sounding data filenames.
	void loadSoundingDataFilenames();

	static std::string getFogModeString(int fogMode);


private:
	bool ready = false;		//!< Whether the VariableManager is ready and initialized

	//! Prints simple help message for command line usage.
	void printHelpMessage(std::string errorMsg = "");

	//! Parses input arguments of the application and saves them to global variables.
	/*!
		Parses input arguments of the application and saves them to global variables.
		Overwrites settings from config.ini if defined!
		It is important to note that boolean options such as useCUDA ("-c") must be defined using true or false argument value since
		we want to be able to rewrite the configuration values. This means that the approach of: if "-c" then use CUDA, if no argument
		"-c" then do not is not possible. This approach would mean that if "-c" is defined, then we overwrite configuration parameter
		and tell the simulator that we want to use CUDA, but if were to omit "-c" it would not set use CUDA to false, but it would use
		the config.ini value which could be both true or false.

	*/
	void parseArguments(int argc, char **argv);


	//! Parses parameter and its value from the configuration file. Assumes correct format for each parameter.
	void saveConfigParam(std::string param, std::string val);

	// Private and self-explanatory functions for saving target parameter of given datatype...
	void saveIntParam(int &target, std::string stringVal);
	void saveFloatParam(float &target, std::string stringVal);
	void saveVec2Param(glm::vec2 &target, std::string line);
	void saveVec3Param(glm::vec3 &target, std::string line);
	void saveVec4Param(glm::vec4 &target, std::string line);
	void saveBoolParam(bool &target, std::string stringVal);
	void saveIntBoolParam(int &target, std::string stringVal);
	void saveStringParam(string &target, std::string stringVal);


};

