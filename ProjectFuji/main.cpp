#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include "Config.h"

//#define GLM_FORCE_CUDA // force GLM to be compatible with CUDA kernels
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <random>
#include <ctime>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip> // setprecision


#include "LBM3D_1D_indices.h"
#include "HeightMap.h"
#include "GeneralGrid.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "Camera2D.h"
#include "OrbitCamera.h"
#include "FreeRoamCamera.h"
#include "ParticleSystemLBM.h"
#include "ParticleSystem.h"
#include "DirectionalLight.h"
#include "Utils.h"
#include "Timer.h"
#include "STLPDiagram.h"
#include "STLPSimulator.h"
#include "STLPSimulatorCUDA.h"
#include "ShaderManager.h"
#include "Skybox.h"
#include "EVSMShadowMapper.h"
#include "CommonEnums.h"
#include "VariableManager.h"
#include "Model.h"
#include "CUDAUtils.cuh"
#include "Emitter.h"
#include "CircleEmitter.h"
#include "TextureManager.h"
#include "OverlayTexture.h"
#include "ParticleRenderer.h"
#include "UIConfig.h"
#include "StreamlineParticleSystem.h"
#include "TerrainPicker.h"
#include "PBRMaterial.h"
#include "MainFramebuffer.h"
#include "SceneGraph.h"
#include "PerlinNoiseSampler.h"
#include "EmitterBrushMode.h"
#include "Timer.h"
#include "TimerManager.h"



#include "HosekSkyModel.h"

//#include <omp.h>	// OpenMP for CPU parallelization

//#include <vld.h>	// Visual Leak Detector for memory leaks analysis

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "UserInterface.h"





///////////////////////////////////////////////////////////////////////////////////////////////////
///// FORWARD DECLARATIONS OF FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////

//! Run the application.
int runApp();

//! Process keyboard inputs of the window.
void processInput(GLFWwindow *window);

//! Key callback for the window.
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

//! Process continuous keyboard input (held buttons).
void processKeyboardInput(GLFWwindow *window);

//! Mouse scroll callback for the window.
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

//! Mouse button callback for the window.
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

//! Mouse position callback for the window.
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

//! Window size changed callback.
void window_size_callback(GLFWwindow* window, int width, int height);



void refreshProjectionMatrix();
void refreshDiagramProjectionMatrix();
void refreshDiagramProjectionMatrix(float aspectRatio);



///////////////////////////////////////////////////////////////////////////////////////////////////
///// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////////////////////////


VariableManager vars;		//!< VariableManager that loads and manages variables across the application

LBM3D_1D_indices *lbm;		//!< Pointer to the current LBM
Camera *camera;				//!< Pointer to the current camera
ParticleSystem *particleSystem;			//!< Particle system used throughout the application
ParticleRenderer *particleRenderer;		//!< Volumetric renderer used for particle rendering
MainFramebuffer *mainFramebuffer;		//!< Main framebuffer used by the application
StreamlineParticleSystem *streamlineParticleSystem;		//!< Debug particle system for drawing streamlines
UserInterface *ui;			//!< User interface used by the application
EmitterBrushMode *ebm;		//!< Helper object that manages brush mode (when user draws particles on the ground)

Camera *viewportCamera;				//!< Camera used in the 3D viewport
Camera *freeRoamCamera;				//!< Free roam camera used in the 3D viewport
Camera *orbitCamera;				//!< Camera that orbits around the terrain
Camera2D *diagramCamera;			//!< Camera used when visualizing the STLP diagram
Camera2D *overlayDiagramCamera;		//!< Fixed diagram camera used for overlay diagram drawing

STLPDiagram *stlpDiagram;			//!< SkewT/LogP diagram instance
STLPSimulatorCUDA *stlpSimCUDA;		//!< SkewT/LogP simulator that runs the simulation on GPU
EVSMShadowMapper *evsm;				//!< Helper class used for exponential variance shadow mapping
DirectionalLight *dirLight;			//!< The sun lighting the scene

Skybox *skybox;						//!< Simple skybox used in the scene
HosekSkyModel *hosek;				//!< Hosek-Wilkie sky model

ShaderProgram *diagramShader;				//!< Shader used for drawing diagram
ShaderProgram *visualizeNormalsShader;		//!< Shader used for visualizing normals
ShaderProgram *normalsInstancedShader;		//!< Shader for instanced models
ShaderProgram *grassShader;					//!< Shader used for grass rendering
ShaderProgram *pbrTest;						//!< PBR testing shader


float lastMouseX;	//!< Previous screen x position of the mouse cursor
float lastMouseY;	//!< Previous screen y position of the mouse cursor



// TIMERS
Timer *renderingTimer;
Timer *lbmTimer;
Timer *sortingTimer;
Timer *stlpTimer;
Timer *globalTimer;


///////////////////////////////////////////////////////////////////////////////////////////////////
///// DEFAULT VALUES THAT ARE TO BE REWRITTEN FROM THE CONFIG FILE
///////////////////////////////////////////////////////////////////////////////////////////////////
double deltaTime = 0.0;		//!< Delta time of the current frame
double lastFrameTime;		//!< Duration of the last frame

glm::mat4 view;				//!< View matrix
glm::mat4 projection;		//!< Projection matrix
glm::mat4 prevProjection;	//!< Projection matrix from previous frame
glm::mat4 viewportProjection;			//!< 3D viewport projection matrix
glm::mat4 diagramProjection;			//!< Diagram projection matrix (flipped y orthographic)
glm::mat4 overlayDiagramProjection;		//!< Overlay diagram projection matrix (flipped y orthographic, no zooming allowed)

float nearPlane = 0.1f;		//!< Near plane of the view frustum
float farPlane = 50000.0f;	//!< Far plane of the view frustum

float projWidth;			//!< Width of the orthographic projection
float projHeight;			//!< Height of the ortographic projection
float projectionRange;		//!< General projection range for 3D (largest value of lattice width, height and depth)


int mouseCursorKey = GLFW_KEY_C;		//!< Consume mouse cursor key

bool leftMouseButtonDown = false;		//!< Holds whether the left mouse button is being held down

float prevAvgFPS;			//!< Average FPS from the previous frame
float prevAvgDeltaTime;		//!< Average delta time from the previous frame



//! Main - runs the application and sets seed for the random number generator.
int main(int argc, char **argv) {
	srand((unsigned int)time(NULL));

	// Load the permutations data for Perlin Noise from file
	PerlinNoiseSampler::loadPermutationsData("resources/perlin_noise_permutations.txt");
	if (!vars.init(argc, argv)) {
		return EXIT_SUCCESS;
	}

	return runApp();
}






int runApp() {

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);


	// No need since we use auxiliary framebuffer (main framebuffer)
	//glfwWindowHint(GLFW_SAMPLES, 12); // enable MSAA with 4 samples

	// Set 
	if (vars.useMonitorResolution || vars.fullscreen) {
		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

		vars.windowWidth = mode->width;
		vars.windowHeight = mode->height;
	}
	GLFWmonitor *monitor = nullptr;
	if (vars.fullscreen) {
		monitor = glfwGetPrimaryMonitor();
	}

	GLFWwindow *window = glfwCreateWindow(vars.windowWidth, vars.windowHeight, "Project Fuji", monitor, nullptr);

	if (!window) {
		cerr << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return -1;
	}

	glfwGetFramebufferSize(window, &vars.screenWidth, &vars.screenHeight);
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		cerr << "Failed to initialize GLAD" << endl;
		return -1;
	}


	ShaderManager::init(&vars);
	CHECK_GL_ERRORS();

	TextureManager::init(&vars);
	CHECK_GL_ERRORS();

	TimerManager::init();

	mainFramebuffer = new MainFramebuffer(&vars);
	vars.mainFramebuffer = mainFramebuffer;

	dirLight = new DirectionalLight();
	evsm = new EVSMShadowMapper(&vars, dirLight);
	stlpDiagram = new STLPDiagram(&vars);
	skybox = new Skybox();
	hosek = new HosekSkyModel(dirLight);
	ui = new UserInterface(window, &vars);

	vars.heightMap = new HeightMap(&vars);
	//tPicker = new TerrainPicker(&vars);

	particleSystem = new ParticleSystem(&vars);
	particleRenderer = new ParticleRenderer(&vars, particleSystem);

	ebm = new EmitterBrushMode(&vars, particleSystem);

	stlpSimCUDA = new STLPSimulatorCUDA(&vars, stlpDiagram);

	particleSystem->stlpSim = stlpSimCUDA;
	stlpSimCUDA->particleSystem = particleSystem;


	stlpSimCUDA->initCUDAGeneral();
	stlpSimCUDA->uploadDataFromDiagramToGPU();





	//////////////////////////////////////////////////////////////////////////////////////////////////
	///// SHADERS
	//////////////////////////////////////////////////////////////////////////////////////////////////
	visualizeNormalsShader = ShaderManager::getShaderPtr("visualize_normals");
	normalsInstancedShader = ShaderManager::getShaderPtr("normals_instanced");
	grassShader = ShaderManager::getShaderPtr("grass_instanced");
	pbrTest = ShaderManager::getShaderPtr("pbr_test");


	// TIMERS
	renderingTimer = TimerManager::createTimer("Particle Rendering", true, false);
	lbmTimer = TimerManager::createTimer("LBM", false, true);
	stlpTimer = TimerManager::createTimer("STLP", false, true);
	sortingTimer = TimerManager::createTimer("Particle Sorting", false, true);
	globalTimer = TimerManager::createTimer("Global Timer", false, false);


	//////////////////////////////////////////////////////////////////////////////////////////////////
	///// CAMERAS AND PROJECTION MATRICES SETUP
	//////////////////////////////////////////////////////////////////////////////////////////////////
	float aspectRatio = (float)vars.screenWidth / (float)vars.screenHeight;

	float offset = 0.2f;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);
	overlayDiagramProjection = glm::ortho(0.0f - offset, 1.0f + offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);

	float tww = vars.heightMap->getWorldWidth();
	float twh = vars.heightMap->terrainHeightRange.y - vars.heightMap->terrainHeightRange.x;
	float twd = vars.heightMap->getWorldDepth();



	projectionRange = (tww > twd) ? tww : twd;
	projectionRange = (projectionRange > twh) ? projectionRange : twh;
	projectionRange /= 2.0f;

	projHeight = projectionRange;
	projWidth = projHeight * aspectRatio;

	viewportProjection = glm::ortho(-projWidth, projWidth, -projHeight, projHeight, nearPlane, farPlane);


	/*int maxNumTextureUnits;
	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxNumTextureUnits);
	cout << "Maximum number of texture units (combined) = " << maxNumTextureUnits << endl;*/

	float cameraRadius = sqrtf((float)(tww * tww + twd * twd)) + 10.0f;

	orbitCamera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 80.0f, glm::vec3(tww / 2.0f, (vars.heightMap->terrainHeightRange.x + vars.heightMap->terrainHeightRange.y) / 2.0f, twd / 2.0f), cameraRadius);

	viewportCamera = orbitCamera;
	camera = viewportCamera;
	diagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	diagramCamera->movementSpeed = 1.0f;
	
	overlayDiagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	
	freeRoamCamera = new FreeRoamCamera(glm::vec3(-3000.0f, 2.0f * vars.terrainHeightRange.y, -1500.0f), WORLD_UP, 35.0f, -35.0f);
	((FreeRoamCamera *)freeRoamCamera)->heightMap = vars.heightMap;
	freeRoamCamera->movementSpeed = vars.cameraSpeed;

	camera->movementSpeed = vars.cameraSpeed;



	// Create and configure the simulator
	lbm = new LBM3D_1D_indices(&vars, particleSystem, stlpDiagram);
	streamlineParticleSystem = new StreamlineParticleSystem(&vars, lbm);
	lbm->streamlineParticleSystem = streamlineParticleSystem;


	//////////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING MODELS AND INSTANCED MODELS
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//Material testMat(TextureManager::loadTexture("textures/body2.png"), TextureManager::loadTexture("textures/body2_S.png"), TextureManager::loadTexture("textures/body2_N.png"), 32.0f);
	//Model testModel("models/housewife.obj", &testMat, ShaderManager::getShaderPtr("normals"));

	Material treeMat(TextureManager::loadTextureTriplet("textures/Bark_Pine_001_COLOR.jpg", "textures/Bark_Pine_001_DISP.png", "textures/Bark_Pine_001_NORM.jpg"), 8.0f);

	Texture gdiffuse("textures/grass.png", 0);
	Texture gspecular("textures/grass_S.png", 1);
	Material gMat(&gdiffuse, &gspecular, nullptr, 32.0f);

	Model grassModel("models/grass.obj", &gMat, grassShader);
	Model treeModel("models/trees10_01.fbx", &treeMat, normalsInstancedShader);
	Model unitboxModel("models/unitbox.fbx");

	// PBR TESTING
	Model cerberus("models/Cerberus_LP.fbx");
	cerberus.transform.position = glm::vec3(4000.0f, 15000.0f, 4000.0f);
	cerberus.transform.scale = glm::vec3(100.0f);


	Texture *calbedo = TextureManager::loadTexture("textures/Cerberus/Cerberus_A.png");
	Texture *cnormalMap = TextureManager::loadTexture("textures/Cerberus/Cerberus_N.png");
	Texture *cmetallic = TextureManager::loadTexture("textures/Cerberus/Cerberus_MR.png");
	Texture *cao = TextureManager::loadTexture("textures/Cerberus/Cerberus_AO.png");
	PBRMaterial cerbmat(calbedo, cmetallic, cnormalMap, cao);
	cerberus.pbrMaterial = &cerbmat;
	cerberus.shader = pbrTest;
	cerberus.visible = 0;


	grassModel.makeInstanced(vars.heightMap, 500000, glm::vec2(0.5f, 2.0f), glm::vec2(10000.0f), glm::vec2(1000.0f));
	grassModel.castShadows = 0;
	treeModel.makeInstanced(vars.heightMap, 1000, glm::vec2(1.0f, 3.0f), glm::vec2(10000.0f), glm::vec2(1000.0f));
	treeModel.castShadows = 0;

	//testModel.transform.position = glm::vec3(1.0f, 0.0f, 5.0f);


	dirLight->position = glm::vec3(140000.0f, 70000.0f, 100000.0f);
	dirLight->focusPoint = glm::vec3(vars.heightMap->getWorldWidth() / 2.0f, 0.0f, vars.heightMap->getWorldDepth() / 2.0f);


	// Create the scene from the test models
	SceneGraph scene;
	scene.root = new Actor("Root");
	scene.root->addChild(&cerberus);
	scene.root->addChild(&treeModel);
	scene.root->addChild(&grassModel);

	grassModel.visible = 0;
	treeModel.visible = 0;

	refreshProjectionMatrix();


	GeneralGrid gGrid(20000, 1000);


	int frameCounter = 0;
	glfwSwapInterval(vars.vsync); // VSync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)

	double prevTime = glfwGetTime();
	long long int totalFrameCounter = 0;


	particleSystem->createPredefinedEmitters();
	particleSystem->refreshParticlesOnTerrain();
	if (!particleSystem->loadParticlesFromFile(vars.startupParticleSaveFile)) {
		particleSystem->formBox(glm::vec3(2000.0f), glm::vec3(2000.0f));
		particleSystem->numActiveParticles = 500000;
	}
	particleSystem->activateAllDiagramParticles();


	lbm->mapVBO(particleSystem->particleVerticesVBO);


	// Set these callbacks after nuklear initialization, otherwise they won't work!
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetKeyCallback(window, key_callback);


	double accumulatedTime = 0.0;

	//glActiveTexture(GL_TEXTURE0);



	// Preset overlay textures that are useful for debugging
	TextureManager::setOverlayTexture(TextureManager::loadTexture("lightTexture[0]"), 0);
	TextureManager::setOverlayTexture(TextureManager::loadTexture("imageTexture"), 1);

	ebm->loadBrushes();
	particleSystem->ebm = ebm;

	// Provisional settings
	ui->dirLight = dirLight;
	ui->evsm = evsm;
	ui->lbm = lbm;
	ui->particleRenderer = particleRenderer;
	ui->particleSystem = particleSystem;
	ui->stlpDiagram = stlpDiagram;
	ui->stlpSimCUDA = stlpSimCUDA;
	ui->hosek = hosek;
	ui->sps = streamlineParticleSystem;
	ui->scene = &scene;
	ui->ebm = ebm;
	ui->mainWindow = window;


	while (!glfwWindowShouldClose(window) && vars.appRunning) {

		if (vars.windowMinimized) {
			glfwPollEvents();
			continue;
		}
		globalTimer->clockAvgStart();

		// enable flags each frame because nuklear disables them when it is rendered	
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_PROGRAM_POINT_SIZE);


		double currentFrameTime = glfwGetTime();
		deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;
		frameCounter++;
		totalFrameCounter++;
		accumulatedTime += deltaTime;

		if (currentFrameTime - prevTime >= 1.0f) {
			prevAvgDeltaTime = 1000.0f * (float)(accumulatedTime / frameCounter);
			prevAvgFPS = 1000.0f / prevAvgDeltaTime;
			//printf("Avg delta time = %0.4f [ms]\n", prevAvgDeltaTime);
			prevTime += (currentFrameTime - prevTime);
			frameCounter = 0;
			accumulatedTime = 0.0;
			ui->prevAvgDeltaTime = prevAvgDeltaTime;
			ui->prevAvgFPS = prevAvgFPS;

		}

		glfwPollEvents();
		processInput(window);

		ui->camera = camera;
		ui->constructUserInterface();




		if (vars.showOverlayDiagram) {
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			view = overlayDiagramCamera->getViewMatrix();
			ShaderManager::updatePVMatrixUniforms(overlayDiagramProjection, view);


			GLint res = stlpDiagram->textureResolution;
			glViewport(0, 0, res, res);
			glBindFramebuffer(GL_FRAMEBUFFER, stlpDiagram->diagramMultisampledFramebuffer);
			glClear(GL_COLOR_BUFFER_BIT);

			stlpDiagram->draw();
			stlpDiagram->drawText();


			if (vars.drawOverlayDiagramParticles) {
				particleSystem->drawDiagramParticles();
			}



			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, stlpDiagram->diagramFramebuffer);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, stlpDiagram->diagramMultisampledFramebuffer);


			glBlitFramebuffer(0, 0, res, res, 0, 0, res, res, GL_COLOR_BUFFER_BIT, GL_NEAREST);


		}




		//cout << " Delta time = " << (deltaTime * 1000.0f) << " [ms]" << endl;
		//cout << " Framerate = " << (1.0f / deltaTime) << endl;
		glm::vec4 clearColor(0.0f, 0.0f, 0.0f, 1.0f);
		if (ui->viewportMode == eViewportMode::DIAGRAM) {
			clearColor = glm::vec4(1.0f);
			camera = diagramCamera;
			glDisable(GL_DEPTH_TEST);
		} else {
			clearColor = glm::vec4(vars.bgClearColor, 1.0f);
			camera = vars.useFreeRoamCamera ? freeRoamCamera : viewportCamera;
			glEnable(GL_DEPTH_TEST);
		}
		
		mainFramebuffer->prepareForNextFrame(clearColor);


		///////////////////////////////////////////////////////////////
		// UPDATE SHARED SHADER UNIFORMS
		///////////////////////////////////////////////////////////////
		view = camera->getViewMatrix();
		ShaderManager::updateViewMatrixUniforms(view);
		ShaderManager::updateDirectionalLightUniforms(*dirLight);
		ShaderManager::updateViewPositionUniforms(camera->position);
		ShaderManager::updateFogUniforms();


		particleSystem->update();


		// LBM simulation update
		if (vars.applyLBM) {
			if (totalFrameCounter % vars.lbmStepFrame == 0) {
				lbmTimer->clockAvgStart();
				lbm->doStepCUDA();
				lbmTimer->clockAvgEnd();
			}
			lbm->recalculateVariables(); // recalculate variables based on the updated values
		}

		// STLP simulation update
		if (vars.applySTLP) {
			if (totalFrameCounter % vars.stlpStepFrame == 0) {
				stlpTimer->clockAvgStart();
				stlpSimCUDA->doStep();
				stlpTimer->clockAvgEnd();
			}
		}

		if (ui->viewportMode == eViewportMode::DIAGRAM) {
			refreshProjectionMatrix();	// reorganize so this can be removed



			stlpDiagram->draw();
			stlpDiagram->drawText();

			if (vars.drawOverlayDiagramParticles) {
				particleSystem->drawDiagramParticles();
			}

		} else if (ui->viewportMode == eViewportMode::VIEWPORT_3D) {

			// Update Hosek's sky model parameters using current sun elevation
			hosek->update();
			


			// Naively simulate sun movement
			if (vars.simulateSun) {
				dirLight->circularMotionStep((float)deltaTime);
			}

			scene.root->update();



			refreshProjectionMatrix();	// reorganize so this can be removed


			///////////////////////////////////////////////////////////////
			// DRAW SKYBOX
			///////////////////////////////////////////////////////////////
			if (vars.drawSkybox) {
				glm::mat4 tmpView = glm::mat4(glm::mat3(view));

				if (vars.hosekSkybox) {
					hosek->draw(tmpView);
				} else {
					skybox->draw(tmpView);
				}
			}


			if (vars.useSkySunColor) {
				glm::vec3 sc = hosek->getSunColor();
				//cout << "sun color: " << sc.x << ", " << sc.y << ", " << sc.z << endl;
				float maxColor = max(max(sc.x, sc.y), sc.z);
				rangeToRange(sc, 0.0f, maxColor, 0.0f, 1.0f);
				float t = vars.skySunColorTintIntensity;

				dirLight->color = (1.0f - t) * glm::vec3(1.0f) + t * sc;
			}

			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);


			evsm->preFirstPass();

			///////////////////////////////////////////////////////////////
			// DRAW DEPTH FOR SHADOW MAPPING (light view)
			///////////////////////////////////////////////////////////////

			grassShader->use();
			grassShader->setVec3("u_CameraPos", camera->position);

			vars.heightMap->drawGeometry(evsm->firstPassShaders[0]);
			//stlpSim->heightMap->drawGeometry(evsm->firstPassShader);

			scene.root->drawShadows(evsm->firstPassShaders[0]);

			evsm->postFirstPass();
			evsm->preSecondPass();

			///////////////////////////////////////////////////////////////
			// DRAW SCENE (camera view)
			///////////////////////////////////////////////////////////////

			vars.heightMap->draw();

			/*
				Updating emitter brush mode requires drawing the terrain, hence why it is in the drawing
				section of the main loop.
			*/
			ebm->update();

			//tPicker->drawTerrain(vars.heightMap);

			if (vars.visualizeTerrainNormals && vars.projectionMode == eProjectionMode::PERSPECTIVE) {
				vars.heightMap->drawGeometry(visualizeNormalsShader);
			}

			scene.root->draw();

			evsm->postSecondPass();


			// Draw helper structures if not in render mode
			if (!vars.renderMode) {
				dirLight->draw();

				particleSystem->drawHelperStructures();

				lbm->draw();
				gGrid.draw();

				streamlineParticleSystem->draw();
			}

			///////////////////////////////////////////////////////////////
			// DRAW PARTICLES
			///////////////////////////////////////////////////////////////		
			if (particleRenderer->useVolumetricRendering) {
	

				// Recalculate half vector to be used for sorting and prepare necessary matrix uniforms
				sortingTimer->clockAvgStart();
				particleRenderer->recalcVectors(camera, dirLight);
				glm::vec3 sortVec = particleRenderer->getSortVec();

				// Check if particles are valid (valid positions, not NaN) - repair if necessary
				// This is important if LBM simulation is active which can become unstable
				particleSystem->checkParticleValidity();

				// Sort particles using the sort half vector 
				particleSystem->sortParticlesByProjection(sortVec, eSortPolicy::LEQUAL);
				sortingTimer->clockAvgEnd();


				// Move from MSAA to regular framebuffer for particle rendering
				mainFramebuffer->blitMultisampledToRegular();

				// Attach the depth texture from the regular main framebuffer to the particle renderer
				particleRenderer->preSceneRenderImage();

				// Draw particles with the half vector slicing method
				renderingTimer->clockAvgStart();
				particleRenderer->draw(particleSystem, dirLight, camera);
				renderingTimer->clockAvgEnd();
				
			} else {
				// Draw very simple particles without any sorting
				renderingTimer->clockAvgStart();
				particleSystem->draw(camera->position);
				renderingTimer->clockAvgEnd();
			}

			if (!vars.renderMode) {
				stlpSimCUDA->draw();

			}



			if (!vars.hideUI && vars.showOverlayDiagram) {
				stlpDiagram->drawOverlayDiagram();
			}

			TextureManager::drawOverlayTextures();


		}




		mainFramebuffer->drawToScreen();
		// We have the default window framebuffer bound -> draw UI as the final step


		// Render the user interface
		ui->draw();


		glfwSwapBuffers(window);

		globalTimer->clockAvgEnd();
		TimerManager::writeToBenchmarkFile();

		CHECK_GL_ERRORS();

	}



	delete particleSystem;
	delete particleRenderer;
	delete streamlineParticleSystem;
	delete lbm;
	delete evsm;

	delete freeRoamCamera;
	delete diagramCamera;
	delete overlayDiagramCamera;
	delete orbitCamera;

	delete dirLight;
	delete ebm;
	delete scene.root;
	delete mainFramebuffer;
	delete skybox;
	delete ui;
	delete hosek;

	//delete vars.heightMap;	// deleted in VariableManager
	delete stlpDiagram;
	delete stlpSimCUDA;


	ShaderManager::tearDown();
	TextureManager::tearDown();
	TimerManager::tearDown();

	//nk_glfw3_shutdown();
	glfwTerminate();

	return 0;


}



void refreshProjectionMatrix() {
	if (ui->viewportMode == eViewportMode::DIAGRAM) {
		projection = diagramProjection;
	} else {
		if (vars.projectionMode == ORTHOGRAPHIC) {
			projection = viewportProjection;
		} else {
			projection = glm::perspective(glm::radians(vars.fov), (float)vars.screenWidth / vars.screenHeight, nearPlane, farPlane);
		}
	}

	ShaderManager::updateProjectionMatrixUniforms(projection);
}


void processInput(GLFWwindow* window) {
	
	processKeyboardInput(window);

	if (leftMouseButtonDown) {


		//cout << "mouse down" << endl;
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		//cout << "Cursor Position at (" << xpos << " : " << ypos << ")" << endl;
		float x = (float)xpos;
		float y = (float)ypos;

		if (ui->viewportMode == eViewportMode::DIAGRAM && stlpDiagram->soundingCurveEditingEnabled) {
			//X_ndc = X_screen * 2.0 / VP_sizeX - 1.0;
			//Y_ndc = Y_screen * 2.0 / VP_sizeY - 1.0;
			//Z_ndc = 2.0 * depth - 1.0;
			x = x * 2.0f / (float)vars.screenWidth - 1.0f;
			y = (float)vars.screenHeight - (float)ypos;
			y = y * 2.0f / (float)vars.screenHeight - 1.0f;

			glm::vec4 mouseCoords(x, y, 0.0f, 1.0f);
			mouseCoords = glm::inverse(view) * glm::inverse(projection) * mouseCoords;
			//cout << "mouse coords = " << glm::to_string(mouseCoords) << endl;

			//stlpDiagram->findClosestSoundingPoint(mouseCoords);

			stlpDiagram->moveSelectedPoint(mouseCoords);
		} else {

			ebm->onLeftMouseButtonDown(x, y);


			/*
			glm::vec4 pos = tPicker->getPixelData(xpos, vars.screenHeight - ypos);

			if (pos.w == 1.0f) {
				((PositionalEmitter*)particleSystem->emitters[0])->position = glm::vec3(pos);
			}
			*/
		}

	}
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	if (!vars.generalKeyboardInputEnabled) {
		return;
	}

	if (action == GLFW_PRESS && mods == 0) {
		if (key == GLFW_KEY_G) {
			if (vars.useFreeRoamCamera && ui->viewportMode == eViewportMode::VIEWPORT_3D) {
				((FreeRoamCamera*)camera)->snapToGround();
				//((FreeRoamCamera*)camera)->walking = !((FreeRoamCamera*)camera)->walking;
			}
		}
		if (key == vars.hideUIKey) {
			vars.hideUI = abs(vars.hideUI - 1);
		}

		if (key == vars.toggleLBMStateKey) {
			vars.applyLBM = abs(vars.applyLBM - 1);
		}
		if (key == vars.toggleSTLPStateKey) {
			vars.applySTLP = abs(vars.applySTLP - 1);
		}

		if (key == GLFW_KEY_KP_1) {
			camera->setView(Camera::VIEW_FRONT);
		}
		if (key == GLFW_KEY_KP_3) {
			camera->setView(Camera::VIEW_SIDE);
		}
		if (key == GLFW_KEY_KP_9) {
			camera->setView(Camera::VIEW_TOP);
		}
		if (key == GLFW_KEY_KP_5) {
			vars.setProjectionMode(abs(1 - vars.projectionMode));
		}

		if (key == GLFW_KEY_1) {
			ui->viewportMode = 0;
			glfwSwapInterval(vars.vsync);
			//refreshProjectionMatrix();
		}
		if (key == GLFW_KEY_2) {
			ui->viewportMode = 1;
			glfwSwapInterval(1);
			//refreshProjectionMatrix();
		}
		if (key == mouseCursorKey) {
			vars.consumeMouseCursor = !vars.consumeMouseCursor;
			if (vars.consumeMouseCursor) {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			} else {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
		if (key == GLFW_KEY_B) {
			ebm->toggleActive();
		}
		if (key == GLFW_KEY_KP_ADD) {
			particleSystem->changeNumActiveParticles(10000);
		}
		if (key == GLFW_KEY_KP_SUBTRACT) {
			particleSystem->changeNumActiveParticles(-10000);
		}



	} else if (action == GLFW_PRESS && mods & GLFW_MOD_SHIFT) {

		if (key == GLFW_KEY_F) {
			ui->setFullscreen(!vars.fullscreen);
		}
		if (key == GLFW_KEY_G) {
			if (vars.useFreeRoamCamera && ui->viewportMode == eViewportMode::VIEWPORT_3D) {
				FreeRoamCamera *fcam = (FreeRoamCamera*)freeRoamCamera;
				int wasWalking = fcam->walking;
				fcam->walking = fcam->walking == 0;
				if (!wasWalking && fcam->walking) {
					fcam->snapToGround();
				}
			}
		}
		if (key == GLFW_KEY_B) {
			particleSystem->formBox();
		}

	} else if (action == GLFW_PRESS && mods & GLFW_MOD_CONTROL) {

		if (key == GLFW_KEY_B) {
			particleRenderer->showParticlesBelowCCL = !particleRenderer->showParticlesBelowCCL;
		}
		if (key == GLFW_KEY_KP_ADD) {
			vars.opacityMultiplier += 0.025f;
			vars.opacityMultiplier = glm::min(3.0f, vars.opacityMultiplier);
		}
		if (key == GLFW_KEY_KP_SUBTRACT) {
			vars.opacityMultiplier -= 0.025f;
			vars.opacityMultiplier = glm::max(0.0f, vars.opacityMultiplier);
		}

	}


}

void processKeyboardInput(GLFWwindow *window) {

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}

	if (!vars.generalKeyboardInputEnabled) {
		return;
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera->processKeyboardMovement(GLFW_KEY_W, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera->processKeyboardMovement(GLFW_KEY_S, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera->processKeyboardMovement(GLFW_KEY_A, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera->processKeyboardMovement(GLFW_KEY_D, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		camera->processKeyboardMovement(GLFW_KEY_E, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		camera->processKeyboardMovement(GLFW_KEY_Q, deltaTime);
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {

	if (ui->isAnyWindowHovered()) {
		return;
	}

	if (ui->viewportMode == eViewportMode::DIAGRAM) {
 		vars.diagramProjectionOffset -= (float)yoffset * 0.04f;
		if (vars.diagramProjectionOffset < -0.45f) {
			vars.diagramProjectionOffset = -0.45f;
		}
		refreshDiagramProjectionMatrix();
		camera->processMouseScroll(-(yoffset * 0.04f));



	} else {
		// no need to poll these if ebm not active
		if (ebm->isActive()) {
			int glfwMods = 0;
			if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
				glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
				glfwMods |= GLFW_MOD_CONTROL;
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
				glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
				glfwMods |= GLFW_MOD_SHIFT;
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
				glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS) {
				glfwMods |= GLFW_MOD_ALT;
			}

			ebm->processMouseWheelScroll((float)yoffset, glfwMods);

			//if (glfwMods == 0) {
			//	camera->processMouseScroll(yoffset);
			//}

		} else {
			camera->processMouseScroll(yoffset);
		}
	}
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	/*
	Use one of these functions to detect whether we want to react to the callback:
		1) nk_item_is_any_active (suggested by Vurtun)
		2) nk_window_is_any_hovered
	*/
	if (ui->isAnyWindowHovered()) {
		//cout << "Mouse callback not valid, hovering over Nuklear window/widget." << endl;
		return;
	}

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {



		if (ui->viewportMode == eViewportMode::DIAGRAM && stlpDiagram->soundingCurveEditingEnabled) {
			//cout << "Cursor Position at (" << xpos << " : " << ypos << ")" << endl;

			//X_ndc = X_screen * 2.0 / VP_sizeX - 1.0;
			//Y_ndc = Y_screen * 2.0 / VP_sizeY - 1.0;
			//Z_ndc = 2.0 * depth - 1.0;
			xpos = xpos * 2.0f / (float)vars.screenWidth - 1.0f;
			ypos = vars.screenHeight - ypos;
			ypos = ypos * 2.0f / (float)vars.screenHeight - 1.0f;

			glm::vec4 mouseCoords(xpos, ypos, 0.0f, 1.0f);
			mouseCoords = glm::inverse(view) * glm::inverse(projection) * mouseCoords;
			//cout << "mouse coords = " << glm::to_string(mouseCoords) << endl;

			stlpDiagram->findClosestSoundingPoint(mouseCoords);
		} else {
			//cout << "Cursor Position at (" << xpos << " : " << ypos << ")" << endl;

			ebm->onLeftMouseButtonPress((float)xpos, (float)ypos);

			/*
			glm::vec4 pos = tPicker->getPixelData(xpos, vars.screenHeight - ypos);

			if (pos.w == 1.0f) {
				((PositionalEmitter*)particleSystem->emitters[0])->position = glm::vec3(pos);
			}
			*/
		}

		leftMouseButtonDown = true;
	} else if (action == GLFW_RELEASE) {
		leftMouseButtonDown = false;

		if (ui->viewportMode == eViewportMode::VIEWPORT_3D) {

			ebm->onLeftMouseButtonRelease((float)xpos, (float)ypos);
		}
	}
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos) {


	static bool firstFrame = true;
	if (firstFrame) {
		lastMouseX = (float)xpos;
		lastMouseY = (float)ypos;
		firstFrame = false;
	}


	float xOffset = (float)xpos - lastMouseX;
	float yOffset = lastMouseY - (float)ypos;

	//cout << xOffset << ", " << yOffset << endl;

	lastMouseX = (float)xpos;
	lastMouseY = (float)ypos;

	if (ui->viewportMode == eViewportMode::DIAGRAM) {

		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)) {
			float offsetMultiplier = 0.001f * (vars.diagramProjectionOffset + 0.6f);
			camera->processMouseMovement(xOffset * offsetMultiplier, yOffset * offsetMultiplier, false);
		}


	} else {

		// for easier controls
		if (vars.consumeMouseCursor) {
			camera->processMouseMovement(xOffset, yOffset, false);
		}

		ebm->updateMousePosition((float)xpos, (float)ypos);
	}
}




void window_size_callback(GLFWwindow* window, int width, int height) {
	float aspectRatio = (float)width / (float)height;

	vars.windowMinimized = (width == 0 || height == 0);

	if (!vars.fullscreen) {
		vars.windowWidth = width;
		vars.windowHeight = height;
	}

	vars.screenWidth = width;
	vars.screenHeight = height;

	

	refreshDiagramProjectionMatrix(aspectRatio);

	projHeight = projectionRange;
	projWidth = projHeight * aspectRatio;
	viewportProjection = glm::ortho(-projWidth, projWidth, -projHeight, projHeight, nearPlane, farPlane);


	mainFramebuffer->refresh();
	particleRenderer->refreshImageBuffer();
	TextureManager::refreshOverlayTextures();
	stlpDiagram->refreshOverlayDiagram((float)vars.screenWidth, (float)vars.screenHeight);
	ui->refreshWidgets();
}


void refreshDiagramProjectionMatrix() {
	float aspectRatio = (float)vars.screenWidth / (float)vars.screenHeight;
	refreshDiagramProjectionMatrix(aspectRatio);
}

void refreshDiagramProjectionMatrix(float aspectRatio) {
	float offset = vars.diagramProjectionOffset;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);
}






