
#include <iostream>

#define GLFW_INCLUDE_NONE

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include "Config.h"

//#define GLM_FORCE_CUDA // force GLM to be compatible with CUDA kernels
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
//#include "glm/gtx/string_cast.hpp"

#include <random>
#include <ctime>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip> // setprecision


//#include "LBM.h"
//#include "LBM2D_1D_indices.h"
#include "LBM3D_1D_indices.h"
#include "HeightMap.h"
//#include "Grid2D.h"
//#include "Grid3D.h"
#include "GeneralGrid.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "Camera2D.h"
#include "OrbitCamera.h"
#include "FreeRoamCamera.h"
#include "ParticleSystemLBM.h"
#include "ParticleSystem.h"
#include "DirectionalLight.h"
//#include "Grid.h"
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

//#include "ArHosekSkyModel.h"
//#include "ArHosekSkyModel.c"

#include "HosekSkyModel.h"

//#include <omp.h>	// OpenMP for CPU parallelization

//#include <vld.h>	// Visual Leak Detector for memory leaks analysis

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>



#include "UserInterface.h"



///////////////////////////////////////////////////////////////////////////////////////////////////
///// FORWARD DECLARATIONS OF FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Run the application.
int runApp();

/// Process keyboard inputs of the window.
void processInput(GLFWwindow *window);

void processKeyboardInput(GLFWwindow *window);

/// Mouse scroll callback for the window.
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

/// Mouse button callback for the window.
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);


/// Window size changed callback.
void window_size_callback(GLFWwindow* window, int width, int height);

/// Constructs the user interface for the given context. Must be called in each frame!
//void constructUserInterface(nk_context *ctx/*, nk_colorf &particlesColor*/);


void refreshProjectionMatrix();




///////////////////////////////////////////////////////////////////////////////////////////////////
///// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////////////////////////


VariableManager vars;


LBM3D_1D_indices *lbm;				///< Pointer to the current LBM
//Grid *grid;				///< Pointer to the current grid
Camera *camera;			///< Pointer to the current camera
//ParticleSystemLBM *particleSystemLBM;		///< Pointer to the particle system that is to be used throughout the whole application
ParticleSystem *particleSystem;
ParticleRenderer *particleRenderer;

MainFramebuffer *mainFramebuffer;

StreamlineParticleSystem *streamlineParticleSystem;

UserInterface *ui;

//TerrainPicker *tPicker;
//HeightMap *heightMap;
EmitterBrushMode *ebm;

//Timer timer;
Camera *viewportCamera;
Camera *freeRoamCamera;
Camera *orbitCamera;
Camera2D *diagramCamera;
Camera2D *overlayDiagramCamera;

//STLPSimulator *stlpSim;
STLPSimulatorCUDA *stlpSimCUDA;

EVSMShadowMapper *evsm;
DirectionalLight *dirLight;


float lastMouseX;
float lastMouseY;



struct nk_context *ctx;

//int projectionMode = PERSPECTIVE;
//int drawSkybox = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////
///// DEFAULT VALUES THAT ARE TO BE REWRITTEN FROM THE CONFIG FILE
///////////////////////////////////////////////////////////////////////////////////////////////////
double deltaTime = 0.0;		///< Delta time of the current frame
double lastFrameTime;		///< Duration of the last frame

glm::mat4 view;				///< View matrix
glm::mat4 projection;		///< Projection matrix
glm::mat4 prevProjection; // for s
glm::mat4 viewportProjection;
glm::mat4 diagramProjection;
glm::mat4 overlayDiagramProjection;

float nearPlane = 0.1f;		///< Near plane of the view frustum
float farPlane = 50000.0f;	///< Far plane of the view frustum

float projWidth;			///< Width of the ortographic projection
float projHeight;			///< Height of the ortographic projection
float projectionRange;		///< General projection range for 3D (largest value of lattice width, height and depth)


int prevPauseKeyState = GLFW_RELEASE;	///< Pause key state from previous frame
int pauseKey = GLFW_KEY_T;				///< Pause key

int prevResetKeyState = GLFW_RELEASE;	///< Reset key state from previous frame
int resetKey = GLFW_KEY_R;				///< Reset key

int prevMouseCursorKeyState = GLFW_RELEASE;
int mouseCursorKey = GLFW_KEY_C;



bool leftMouseButtonDown = false;


float prevAvgFPS;
float prevAvgDeltaTime;

STLPDiagram *stlpDiagram;	///< SkewT/LogP diagram instance

Skybox *skybox;
HosekSkyModel *hosek;


ShaderProgram *diagramShader;
ShaderProgram *visualizeNormalsShader;
ShaderProgram *normalsInstancedShader;
ShaderProgram *grassShader;

ShaderProgram *pbrTest;


/// Main - runs the application and sets seed for the random number generator.
int main(int argc, char **argv) {
	srand(time(NULL));


	PerlinNoiseSampler::loadPermutationsData("resources/perlin_noise_permutations.txt");
	//cout << PerlinNoiseSampler::getSample(3.14f, 42.0f, 7.0f);

	vars.init(argc, argv);

	return runApp();
}






int runApp() {

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);


	//glfwWindowHint(GLFW_SAMPLES, 12); // enable MSAA with 4 samples

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



	float aspectRatio = (float)vars.screenWidth / (float)vars.screenHeight;

	float offset = 0.2f;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);
	overlayDiagramProjection = glm::ortho(0.0f - offset, 1.0f + offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);


	int maxNumTextureUnits;
	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxNumTextureUnits);
	cout << "Maximum number of texture units (combined) = " << maxNumTextureUnits << endl;

	// Create and configure the simulator
	{

		lbm = new LBM3D_1D_indices(&vars, particleSystem, stlpDiagram);


		projectionRange = (float)((vars.latticeWidth > vars.latticeHeight) ? vars.latticeWidth : vars.latticeHeight);
		projectionRange = (projectionRange > vars.latticeDepth) ? projectionRange : vars.latticeDepth;
		projectionRange /= 2.0f;

		projHeight = projectionRange;
		projWidth = projHeight * aspectRatio;

		projection = glm::ortho(-projWidth, projWidth, -projHeight, projHeight, nearPlane, farPlane);


		float cameraRadius = sqrtf((float)(vars.heightMap->getWorldWidth() * vars.heightMap->getWorldWidth() + vars.heightMap->getWorldDepth() * vars.heightMap->getWorldDepth())) + 10.0f;

		orbitCamera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 80.0f, glm::vec3(vars.heightMap->getWorldWidth() / 2.0f, (vars.heightMap->terrainHeightRange.x + vars.heightMap->terrainHeightRange.y) / 2.0f, vars.heightMap->getWorldDepth() / 2.0f), cameraRadius);
	}

	CHECK_ERROR(cudaPeekAtLastError());

	viewportCamera = orbitCamera;
	camera = viewportCamera;
	diagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	diagramCamera->movementSpeed = 1.0f;
	
	overlayDiagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	
	freeRoamCamera = new FreeRoamCamera(glm::vec3(30.0f, vars.terrainHeightRange.y, 30.0f), WORLD_UP, -35.0f, -35.0f);
	((FreeRoamCamera *)freeRoamCamera)->heightMap = vars.heightMap;
	freeRoamCamera->movementSpeed = vars.cameraSpeed;

	viewportProjection = projection;

	camera->setLatticeDimensions(vars.latticeWidth, vars.latticeHeight, vars.latticeDepth);
	camera->movementSpeed = vars.cameraSpeed;

	dirLight->focusPoint = glm::vec3(vars.heightMap->getWorldWidth() / 2.0f, 0.0f, vars.heightMap->getWorldDepth() / 2.0f);

	// TODO - cleanup this hack
	streamlineParticleSystem = new StreamlineParticleSystem(&vars, lbm);
	lbm->streamlineParticleSystem = streamlineParticleSystem;


	Material testMat(TextureManager::getTexturePtr("textures/body2.png"), TextureManager::getTexturePtr("textures/body2_S.png"), TextureManager::getTexturePtr("textures/body2_N.png"), 32.0f);

	Model testModel("models/housewife.obj", &testMat, ShaderManager::getShaderPtr("normals"));

	Material treeMat(TextureManager::getTextureTripletPtrs("textures/Bark_Pine_001_COLOR.jpg", "textures/Bark_Pine_001_DISP.png", "textures/Bark_Pine_001_NORM.jpg"), 8.0f);

	Texture adiffuse("textures/armoire/albedo.png", 0);
	Texture aspecular("textures/armoire/metallic.png", 1);
	Texture anormal("textures/armoire/normal.png", 2);
	Material aMat(adiffuse, aspecular, anormal, 32.0f);

	Texture gdiffuse("textures/grass.png", 0);
	Texture gspecular("textures/grass_S.png", 1);
	Material gMat(gdiffuse, gspecular, anormal, 32.0f);

	Model grassModel("models/grass.obj", &gMat, grassShader);

	Model treeModel("models/trees10_01.fbx", &treeMat, normalsInstancedShader);

	Model unitboxModel("models/unitbox.fbx");

	Model houseModel("models/house_model/houselow_r.fbx");

	Texture halbedo("models/house_model/unknown_Base_Color.png", 0);
	Texture hmr("models/house_model/unknown_MR.png", 1);
	Texture hnormal("models/house_model/unknown_Normal_OpenGL.png", 2);
	Texture hao("models/house_model/house_ao.jpg", 3);
	PBRMaterial hmat(&halbedo, &hmr, &hnormal, &hao);

	houseModel.pbrMaterial = &hmat;

	float hx = 10000.0f;
	float hz = 10000.0f;
	houseModel.transform.position = glm::vec3(hx, vars.heightMap->getHeight(hx, hz), hz);
	houseModel.shader = pbrTest;

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


	grassModel.makeInstanced(vars.heightMap, 500000, glm::vec2(0.5f, 2.0f), glm::vec2(10000.0f), glm::vec2(1000.0f));
	grassModel.castShadows = 0;
	treeModel.makeInstanced(vars.heightMap, 1000, glm::vec2(1.0f, 3.0f), glm::vec2(10000.0f), glm::vec2(1000.0f));
	treeModel.castShadows = 0;

	testModel.transform.position = glm::vec3(1.0f, 0.0f, 5.0f);


	dirLight->position = glm::vec3(10000.0f, 15000.0f, 20000.0f);

	//testModel.snapToGround(vars.heightMap);


	SceneGraph scene;
	scene.root = new Actor("Root");
	scene.root->addChild(&cerberus);
	scene.root->addChild(&houseModel);
	scene.root->addChild(&treeModel);
	scene.root->addChild(&grassModel);
	houseModel.addChild(&testModel);


	refreshProjectionMatrix();


	GeneralGrid gGrid(20000.0f, 1000.0f);


	int frameCounter = 0;
	glfwSwapInterval(vars.vsync); // VSync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)

	double prevTime = glfwGetTime();
	long long int totalFrameCounter = 0;


	particleSystem->createPredefinedEmitters();
	particleSystem->initParticlesOnTerrain();
	particleSystem->formBox(glm::vec3(2000.0f), glm::vec3(2000.0f));
	particleSystem->activateAllParticles();
	particleSystem->activateAllDiagramParticles();

	//particleSystem->initParticlePositions();
	CHECK_ERROR(cudaPeekAtLastError());



	glBindBuffer(GL_ARRAY_BUFFER, 0);
	lbm->mapVBOTEST(particleSystem->particleVerticesVBO, particleSystem->cudaParticleVerticesVBO);


	// Set these callbacks after nuklear initialization, otherwise they won't work!
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);



	stringstream ss;
	ss << vars.sceneFilename;
	ss << "_h=" << vars.latticeHeight;
	//ss << "_" << particleSystemLBM->numParticles;
	vars.timer.configString = ss.str();
	if (vars.measureTime) {
		vars.timer.start();
	}
	double accumulatedTime = 0.0;

	glActiveTexture(GL_TEXTURE0);

	CHECK_ERROR(cudaPeekAtLastError());


	vector<GLuint> debugTextureIds;
	debugTextureIds.push_back(evsm->getDepthMapTextureId());


	// Preset overlay textures that are useful for debugging
	TextureManager::setOverlayTexture(TextureManager::getTexturePtr("lightTexture[0]"), 0);
	TextureManager::setOverlayTexture(TextureManager::getTexturePtr("imageTexture"), 1);

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


	while (!glfwWindowShouldClose(window) && vars.appRunning) {
		// enable flags each frame because nuklear disables them when it is rendered	
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_PROGRAM_POINT_SIZE);

		CHECK_GL_ERRORS();

		double currentFrameTime = glfwGetTime();
		deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;
		frameCounter++;
		totalFrameCounter++;
		accumulatedTime += deltaTime;

		if (currentFrameTime - prevTime >= 1.0f) {
			prevAvgDeltaTime = 1000.0 * (accumulatedTime / frameCounter);
			prevAvgFPS = 1000.0 / prevAvgDeltaTime;
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


		if (vars.measureTime) {
			vars.timer.clockAvgStart();
		}



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

			CHECK_GL_ERRORS();

		}




		//cout << " Delta time = " << (deltaTime * 1000.0f) << " [ms]" << endl;
		//cout << " Framerate = " << (1.0f / deltaTime) << endl;
		glm::vec4 clearColor(0.0f, 0.0f, 0.0f, 1.0f);
		if (ui->viewportMode == eViewportMode::DIAGRAM) {
			clearColor = glm::vec4(1.0f);
			glfwSwapInterval(1);
			camera = diagramCamera;
			glDisable(GL_DEPTH_TEST);
		} else {
			clearColor = glm::vec4(vars.bgClearColor, 1.0f);
			glfwSwapInterval(0);
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




		if (ui->viewportMode == eViewportMode::DIAGRAM) {
			refreshProjectionMatrix();

			stlpDiagram->draw();
			stlpDiagram->drawText();

			if (vars.drawOverlayDiagramParticles) {
				particleSystem->drawDiagramParticles();
			}

		} else if (ui->viewportMode == eViewportMode::VIEWPORT_3D) {

			// Update Hosek's sky model parameters using current sun elevation
			hosek->update();
			

			// Update particle system (emit particles mainly)
			particleSystem->update();

			// LBM simulation update
			if (vars.applyLBM) {
				if (totalFrameCounter % vars.lbmStepFrame == 0) {
					lbm->doStepCUDA();
				}
				lbm->recalculateVariables(); // recalculate variables based on the updated values
			}

			// STLP simulation update
			if (vars.applySTLP) {
				if (totalFrameCounter % vars.stlpStepFrame == 0) {
					stlpSimCUDA->doStep();
				}
			}

			// Naively simulate sun movement
			if (vars.simulateSun) {
				dirLight->circularMotionStep(deltaTime);
			}

			scene.root->update();


			refreshProjectionMatrix();


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


			CHECK_GL_ERRORS();



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

			if (vars.visualizeTerrainNormals) {
				vars.heightMap->drawGeometry(visualizeNormalsShader);
			}

			scene.root->draw();

			evsm->postSecondPass();

			CHECK_GL_ERRORS();



			dirLight->draw();

			if (!vars.renderMode) {
				particleSystem->drawHelperStructures();

				lbm->draw();
				gGrid.draw();

				streamlineParticleSystem->draw();
			}


			///////////////////////////////////////////////////////////////
			// DRAW PARTICLES
			///////////////////////////////////////////////////////////////		
			if (particleRenderer->useVolumetricRendering) {
	

				particleRenderer->recalcVectors(camera, dirLight);
				glm::vec3 sortVec = particleRenderer->getSortVec();

				// NOW sort particles using the sort vector
				particleSystem->sortParticlesByProjection(sortVec, eSortPolicy::LEQUAL);

				mainFramebuffer->blitMultisampledToRegular();

				particleRenderer->preSceneRenderImage();
				//vars.heightMap->draw();
				//particleRenderer->postSceneRenderImage();


				particleRenderer->draw(particleSystem, dirLight, camera);
				
			} else {
				particleSystem->draw(camera->position);
			}

			if (!vars.renderMode) {
				stlpSimCUDA->draw(camera->position);

			}



			if (!vars.hideUI && vars.showOverlayDiagram) {
				stlpDiagram->drawOverlayDiagram();
			}



			TextureManager::drawOverlayTextures();


		}


		CHECK_GL_ERRORS();


		mainFramebuffer->drawToScreen();
		// We have the default window framebuffer bound -> draw UI as the final step


		// Render the user interface
		ui->draw();

		glfwSwapBuffers(window);

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
	//delete tPicker;
	delete ebm;
	delete stlpDiagram;

	delete scene.root;

	delete mainFramebuffer;

	delete skybox;



	ShaderManager::tearDown();
	TextureManager::tearDown();

	//nk_glfw3_shutdown();
	glfwTerminate();

	if (vars.measureTime) {
		vars.timer.end();
	}

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


		if (ui->viewportMode == eViewportMode::DIAGRAM) {
			//X_ndc = X_screen * 2.0 / VP_sizeX - 1.0;
			//Y_ndc = Y_screen * 2.0 / VP_sizeY - 1.0;
			//Z_ndc = 2.0 * depth - 1.0;
			xpos = xpos * 2.0f / (float)vars.screenWidth - 1.0f;
			ypos = vars.screenHeight - ypos;
			ypos = ypos * 2.0f / (float)vars.screenHeight - 1.0f;

			glm::vec4 mouseCoords(xpos, ypos, 0.0f, 1.0f);
			mouseCoords = glm::inverse(view) * glm::inverse(projection) * mouseCoords;
			//cout << "mouse coords = " << glm::to_string(mouseCoords) << endl;

			//stlpDiagram->findClosestSoundingPoint(mouseCoords);

			stlpDiagram->moveSelectedPoint(mouseCoords);
		} else {

			ebm->onLeftMouseButtonDown(xpos, ypos);


			/*
			glm::vec4 pos = tPicker->getPixelData(xpos, vars.screenHeight - ypos);

			if (pos.w == 1.0f) {
				((PositionalEmitter*)particleSystem->emitters[0])->position = glm::vec3(pos);
			}
			*/
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
		//camera->processKeyboardMovement(Camera::UP, deltaTime);
		camera->processKeyboardMovement(GLFW_KEY_W, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		//camera->processKeyboardMovement(Camera::DOWN, deltaTime);
		camera->processKeyboardMovement(GLFW_KEY_S, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		//camera->processKeyboardMovement(Camera::LEFT, deltaTime);
		camera->processKeyboardMovement(GLFW_KEY_A, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		//camera->processKeyboardMovement(Camera::RIGHT, deltaTime);
		camera->processKeyboardMovement(GLFW_KEY_D, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		//camera->processKeyboardMovement(Camera::ROTATE_LEFT, deltaTime);
		camera->processKeyboardMovement(GLFW_KEY_E, deltaTime);

	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		//camera->processKeyboardMovement(Camera::ROTATE_RIGHT, deltaTime);
		camera->processKeyboardMovement(GLFW_KEY_Q, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
		if (vars.useFreeRoamCamera) {
			((FreeRoamCamera*)camera)->snapToGround();
			//((FreeRoamCamera*)camera)->walking = !((FreeRoamCamera*)camera)->walking;
		}
	}
	if (glfwGetKey(window, vars.hideUIKey) == GLFW_PRESS) {
		if (vars.prevHideUIKeyState == GLFW_RELEASE) {
			vars.hideUI = abs(vars.hideUI - 1);
		}
		vars.prevHideUIKeyState = GLFW_PRESS;
	} else {
		vars.prevHideUIKeyState = GLFW_RELEASE;
	}

	// Toggle LBM
	if (glfwGetKey(window, vars.toggleLBMState) == GLFW_PRESS) {
		if (vars.prevToggleLBMState == GLFW_RELEASE) {
			vars.applyLBM = abs(vars.applyLBM - 1);
		}
		vars.prevToggleLBMState = GLFW_PRESS;
	} else {
		vars.prevToggleLBMState = GLFW_RELEASE;
	}


	// Toggle STLP
	if (glfwGetKey(window, vars.toggleSTLPState) == GLFW_PRESS) {
		if (vars.prevToggleSTLPState == GLFW_RELEASE) {
			vars.applySTLP = abs(vars.applySTLP - 1);
		}
		vars.prevToggleSTLPState = GLFW_PRESS;
	} else {
		vars.prevToggleSTLPState = GLFW_RELEASE;
	}



	if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_FRONT);
	}
	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_SIDE);
	}
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_TOP);
	}
	if (glfwGetKey(window, resetKey) == GLFW_PRESS) {
		if (prevResetKeyState == GLFW_RELEASE) {
			lbm->resetSimulation();
		}
		prevResetKeyState = GLFW_PRESS;
	} else {
		prevResetKeyState = GLFW_RELEASE;
	}
	if (glfwGetKey(window, pauseKey) == GLFW_PRESS) {
		if (prevPauseKeyState == GLFW_RELEASE) {
			vars.paused = !vars.paused;
		}
		prevPauseKeyState = GLFW_PRESS;
	} else {
		prevPauseKeyState = GLFW_RELEASE;
	}



	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
		ui->viewportMode = 0;
		refreshProjectionMatrix();
	}
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
		ui->viewportMode = 1;
		refreshProjectionMatrix();
	}


	if (glfwGetKey(window, mouseCursorKey) == GLFW_PRESS) {
		if (prevMouseCursorKeyState == GLFW_RELEASE) {
			//vars.paused = !vars.paused;
			// do action
			vars.consumeMouseCursor = !vars.consumeMouseCursor;
			if (vars.consumeMouseCursor) {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			} else {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
		prevMouseCursorKeyState = GLFW_PRESS;
	} else {
		prevMouseCursorKeyState = GLFW_RELEASE;
	}

}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera->processMouseScroll(yoffset);
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



		if (ui->viewportMode == eViewportMode::DIAGRAM) {
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
			cout << "Cursor Position at (" << xpos << " : " << ypos << ")" << endl;

			ebm->onLeftMouseButtonPress(xpos, ypos);

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

			ebm->onLeftMouseButtonRelease(xpos, ypos);
		}
	}
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos) {

	if (ui->viewportMode == eViewportMode::DIAGRAM) {
		return;
	}

	static bool firstFrame = true;
	if (firstFrame) {
		lastMouseX = xpos;
		lastMouseY = ypos;
		firstFrame = false;
	}


	float xOffset = xpos - lastMouseX;
	float yOffset = lastMouseY - ypos;

	//cout << xOffset << ", " << yOffset << endl;

	lastMouseX = xpos;
	lastMouseY = ypos;



	// for easier controls
	if (vars.consumeMouseCursor) {
		camera->processMouseMovement(xOffset, yOffset, false);
	}

	ebm->updateMousePosition(xpos, ypos);

}




void window_size_callback(GLFWwindow* window, int width, int height) {
	float aspectRatio = (float)width / (float)height;

	vars.screenWidth = width;
	vars.screenHeight = height;

	float offset = 0.2f;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);



	projHeight = projectionRange;
	projWidth = projHeight * aspectRatio;
	viewportProjection = glm::ortho(-projWidth, projWidth, -projHeight, projHeight, nearPlane, farPlane);


	mainFramebuffer->refresh();
	particleRenderer->refreshImageBuffer();
	TextureManager::refreshOverlayTextures();
	stlpDiagram->refreshOverlayDiagram(vars.screenWidth, vars.screenHeight);
}





