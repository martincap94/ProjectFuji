
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
#include "StaticMesh.h"
#include "Model.h"
#include "CUDAUtils.cuh"
#include "Emitter.h"
#include "CircleEmitter.h"
#include "TextureManager.h"
#include "OverlayTexture.h"
#include "ParticleRenderer.h"
#include "UIConfig.h"
#include "StreamlineParticleSystem.h"

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
void processInput(GLFWwindow* window);

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

//eLBMType lbmType;		///< The LBM type that is to be displayed

LBM3D_1D_indices *lbm;				///< Pointer to the current LBM
//Grid *grid;				///< Pointer to the current grid
Camera *camera;			///< Pointer to the current camera
//ParticleSystemLBM *particleSystemLBM;		///< Pointer to the particle system that is to be used throughout the whole application
ParticleSystem *particleSystem;
ParticleRenderer *particleRenderer;

StreamlineParticleSystem *streamlineParticleSystem;

UserInterface *ui;

//HeightMap *heightMap;

//Timer timer;
Camera *viewportCamera;
Camera *freeRoamCamera;
Camera *orbitCamera;
Camera2D *diagramCamera;
Camera2D *overlayDiagramCamera;

//STLPSimulator *stlpSim;
STLPSimulatorCUDA *stlpSimCUDA;

EVSMShadowMapper evsm;
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
float farPlane = 100000.0f;	///< Far plane of the view frustum

float projWidth;			///< Width of the ortographic projection
float projHeight;			///< Height of the ortographic projection
float projectionRange;		///< General projection range for 3D (largest value of lattice width, height and depth)


int prevPauseKeyState = GLFW_RELEASE;	///< Pause key state from previous frame
int pauseKey = GLFW_KEY_T;				///< Pause key

int prevResetKeyState = GLFW_RELEASE;	///< Reset key state from previous frame
int resetKey = GLFW_KEY_R;				///< Reset key

int prevMouseCursorKeyState = GLFW_RELEASE;
int mouseCursorKey = GLFW_KEY_C;


//string soundingFile;		///< Name of the sounding file to be loaded

bool mouseDown = false;

//bool updateFPSCounter = false;
float prevAvgFPS;
float prevAvgDeltaTime;

STLPDiagram stlpDiagram;	///< SkewT/LogP diagram instance
int mode = 3;				///< Mode: 0 - show SkewT/LogP diagram, 1 - show 3D simulator

Skybox *skybox;
HosekSkyModel *hosek;


ShaderProgram *singleColorShader;
ShaderProgram *singleColorShaderVBO;
ShaderProgram *singleColorShaderAlpha;
ShaderProgram *unlitColorShader;
ShaderProgram *dirLightOnlyShader;
ShaderProgram *textShader;
ShaderProgram *curveShader;
ShaderProgram *pointSpriteTestShader;
ShaderProgram *coloredParticleShader;
ShaderProgram *diagramShader;
ShaderProgram *skyboxShader;
ShaderProgram *hosekShader;
ShaderProgram *visualizeNormalsShader;


/// Main - runs the application and sets seed for the random number generator.
int main(int argc, char **argv) {
	srand(time(NULL));



	vars.init(argc, argv);

	runApp();

	return 0;
}

/// Runs the application including the game loop.
/**
	Creates the window, user interface and all the main parts of the simulation including the simulator itself (either
	LBM 2D or 3D), grids, particle system, and collider object (2D) or height map (3D).
	Furthermore, creates all the shaders and runs the main game loop of the application in which the simulation is updated
	and the UI is drawn (and constructed since nuklear panel needs to be constructed in each frame).
*/
int runApp() {

	/*int ompMaxThreads = omp_get_max_threads();
	printf("OpenMP max threads = %d\n", ompMaxThreads);

	omp_set_num_threads(ompMaxThreads);


	int count = 0;
#pragma omp parallel num_threads(ompMaxThreads)
	{
#pragma omp atomic
		count++;
	}
	printf_s("Number of threads: %d\n", count);*/

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);


	glfwWindowHint(GLFW_SAMPLES, 12); // enable MSAA with 4 samples

	GLFWwindow *window = glfwCreateWindow(vars.windowWidth, vars.windowHeight, "Project Fuji", nullptr, nullptr);

	if (!window) {
		cerr << "Failed to create GLFW window" << endl;
		glfwTerminate(); // maybe unnecessary according to the documentation
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

	evsm.init();
	dirLight = new DirectionalLight();
	stlpDiagram.init(vars.soundingFile);


	skybox = new Skybox();
	hosek = new HosekSkyModel();

	glViewport(0, 0, vars.screenWidth, vars.screenHeight);

	float aspectRatio = (float)vars.screenWidth / (float)vars.screenHeight;
	//cout << "Aspect ratio = " << aspectRatio << endl;

	float offset = 0.2f;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);
	overlayDiagramProjection = glm::ortho(0.0f - offset, 1.0f + offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);


	ui = new UserInterface(window, &vars);
	

	//////////////////////////////////////////////////////////////////////////////////////////////////
	///// SHADERS
	//////////////////////////////////////////////////////////////////////////////////////////////////
	singleColorShader = ShaderManager::getShaderPtr("singleColor");
	singleColorShaderAlpha = ShaderManager::getShaderPtr("singleColorAlpha");
	singleColorShaderVBO = ShaderManager::getShaderPtr("singleColor_VBO");

	unlitColorShader = ShaderManager::getShaderPtr("unlitColor");

	dirLightOnlyShader = ShaderManager::getShaderPtr("dirLightOnly");
	//dirLightOnlyShader = ShaderManager::getShaderPtr("dirLightOnly_evsm");

	pointSpriteTestShader = ShaderManager::getShaderPtr("pointSpriteTest");
	coloredParticleShader = ShaderManager::getShaderPtr("coloredParticle");
	diagramShader = ShaderManager::getShaderPtr("overlayTexture");

	textShader = ShaderManager::getShaderPtr("text");
	curveShader = ShaderManager::getShaderPtr("curve");
	skyboxShader = ShaderManager::getShaderPtr("skybox");
	hosekShader = ShaderManager::getShaderPtr("sky_hosek");
	visualizeNormalsShader = ShaderManager::getShaderPtr("visualize_normals");

	vars.heightMap = new HeightMap(&vars/*, vars.sceneFilename, vars.latticeHeight*/);
	//vars.heightMap->vars = &vars;

	//struct nk_colorf particlesColor;


	//particleSystemLBM = new ParticleSystemLBM(vars.numParticles, vars.drawStreamlines);
	particleSystem = new ParticleSystem(&vars);
	particleRenderer = new ParticleRenderer(&vars);


	glm::ivec3 latticeDim(vars.latticeWidth, vars.latticeHeight, vars.latticeDepth);

	float ratio = (float)vars.screenWidth / (float)vars.screenHeight;



	// Create and configure the simulator
	{

		dim3 blockDim(vars.blockDim_3D_x, vars.blockDim_3D_y, 1);
		CHECK_ERROR(cudaPeekAtLastError());

		lbm = new LBM3D_1D_indices(&vars, latticeDim, vars.sceneFilename, vars.tau, nullptr, particleSystem, blockDim, &stlpDiagram);
		CHECK_ERROR(cudaPeekAtLastError());


		vars.latticeWidth = lbm->latticeWidth;
		vars.latticeHeight = lbm->latticeHeight;
		vars.latticeDepth = lbm->latticeDepth;



		projectionRange = (float)((vars.latticeWidth > vars.latticeHeight) ? vars.latticeWidth : vars.latticeHeight);
		projectionRange = (projectionRange > vars.latticeDepth) ? projectionRange : vars.latticeDepth;
		projectionRange /= 2.0f;

		projHeight = projectionRange;
		projWidth = projHeight * ratio;

		projection = glm::ortho(-projWidth, projWidth, -projHeight, projHeight, nearPlane, farPlane);
		//grid = new Grid3D(vars.latticeWidth * lbm->scale, vars.latticeHeight * lbm->scale, vars.latticeDepth * lbm->scale, 6, 6, 6);
		float cameraRadius = sqrtf((float)(vars.latticeWidth * vars.latticeWidth + vars.latticeDepth * vars.latticeDepth)) + 10.0f;

		orbitCamera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 80.0f, glm::vec3(vars.latticeWidth / 2.0f, vars.latticeHeight / 2.0f, vars.latticeDepth / 2.0f), cameraRadius);
	}

	CHECK_ERROR(cudaPeekAtLastError());

	viewportCamera = orbitCamera;
	camera = viewportCamera;
	diagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	overlayDiagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	freeRoamCamera = new FreeRoamCamera(glm::vec3(30.0f, vars.terrainHeightRange.y, 30.0f), WORLD_UP, -35.0f, -35.0f);
	((FreeRoamCamera *)freeRoamCamera)->heightMap = vars.heightMap;
	freeRoamCamera->movementSpeed = vars.cameraSpeed;

	viewportProjection = projection;

	camera->setLatticeDimensions(vars.latticeWidth, vars.latticeHeight, vars.latticeDepth);
	camera->movementSpeed = vars.cameraSpeed;

	dirLight->focusPoint = glm::vec3(vars.heightMap->getWorldWidth() / 2.0f, 0.0f, vars.heightMap->getWorldDepth() / 2.0f);
	dirLight->color = glm::vec3(1.0f, 0.99f, 0.9f);
	//particleSystemLBM->lbm = lbm;

	// TO DO - cleanup this hack

	streamlineParticleSystem = new StreamlineParticleSystem(&vars, lbm);
	lbm->streamlineParticleSystem = streamlineParticleSystem;




	//StaticMesh testMesh("models/House_3.obj", ShaderManager::getShaderPtr("dirLightOnly"), nullptr);
	//Model testModel("models/House_3.obj");

	Material testMat(TextureManager::getTexturePtr("textures/body2.png"), TextureManager::getTexturePtr("textures/body2_S.png"), TextureManager::getTexturePtr("textures/body2_N.png"), 32.0f);

	Model testModel("models/housewife.obj", &testMat, ShaderManager::getShaderPtr("normals"));
	StaticMesh testMesh("models/housewife.obj", ShaderManager::getShaderPtr("normals"), &testMat);

	Material treeMat(TextureManager::getTextureTripletPtrs("textures/Bark_Pine_001_COLOR.jpg", "textures/Bark_Pine_001_DISP.png", "textures/Bark_Pine_001_NORM.jpg"), 8.0f);

	Texture adiffuse("textures/armoire/albedo.png", 0);
	Texture aspecular("textures/armoire/metallic.png", 1);
	Texture anormal("textures/armoire/normal.png", 2);
	Material aMat(adiffuse, aspecular, anormal, 32.0f);

	Texture gdiffuse("textures/grass.png", 0);
	Texture gspecular("textures/grass_S.png", 1);
	Material gMat(gdiffuse, gspecular, anormal, 32.0f);

	Model armoireModel("models/armoire.fbx", &aMat, ShaderManager::getShaderPtr("normals"));
	Model grassModel("models/grass.obj", &gMat, ShaderManager::getShaderPtr("normals_instanced"));
	//Model grassModel("models/grass.obj", &gMat, ShaderManager::getShaderPtr("normals"));

	Model treeModel("models/Tree.obj", &treeMat, ShaderManager::getShaderPtr("normals_instanced"));

	Model unitboxModel("models/unitbox.fbx");

	grassModel.makeInstanced(vars.heightMap, 1000000, glm::vec2(0.5f, 2.0f), 500.0f, 3, glm::vec2(3500.0f, 8500.0f), glm::vec2(1000.0f));
	treeModel.makeInstanced(vars.heightMap, 1000, glm::vec2(3.0f, 5.5f), 1000.0f, 20);

	testModel.transform.position = glm::vec3(3500.0f, 0.0f, 8500.0f);
	//testModel.transform.scale = glm::vec3(20.0f);

	armoireModel.transform.position = glm::vec3(3000.0f, 0.0f, 8000.0f);
	armoireModel.transform.scale = glm::vec3(1.0f);

	dirLight->position = glm::vec3(10000.0f, 15000.0f, 20000.0f);

	testModel.snapToGround(vars.heightMap);
	armoireModel.snapToGround(vars.heightMap);




	evsm.dirLight = dirLight;


	refreshProjectionMatrix();


	GeneralGrid gGrid(100, 5, (vars.lbmType == LBM3D));


	int frameCounter = 0;
	glfwSwapInterval(vars.vsync); // VSync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)

	double prevTime = glfwGetTime();
	long long int totalFrameCounter = 0;
	//int measurementFrameCounter = 0;
	//double accumulatedTime = 0.0;


	CHECK_ERROR(cudaPeekAtLastError());



	//stlpSim = new STLPSimulator(&vars, &stlpDiagram);
	stlpSimCUDA = new STLPSimulatorCUDA(&vars, &stlpDiagram);

	CHECK_ERROR(cudaPeekAtLastError());

	//stlpSim->initParticles();
	//stlpSimCUDA->initParticles();
	particleSystem->stlpSim = stlpSimCUDA;
	stlpSimCUDA->particleSystem = particleSystem;

	//stlpSimCUDA->initCUDA();
	stlpSimCUDA->initCUDAGeneral();
	stlpSimCUDA->uploadDataFromDiagramToGPU();


	//stlpDiagram.particlesVAO = stlpSimCUDA->diagramParticlesVAO; // hack

	CHECK_ERROR(cudaPeekAtLastError());



	particleSystem->initParticlesOnTerrain();
	particleSystem->formBox(glm::vec3(2000.0f), glm::vec3(2000.0f));
	particleSystem->activateAllParticles();


	//particleSystem->initParticlePositions();
	CHECK_ERROR(cudaPeekAtLastError());



	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//lbm->mapVBOTEST(stlpSimCUDA->particlesVBO, stlpSimCUDA->cudaParticleVerticesVBO);
	lbm->mapVBOTEST(particleSystem->particleVerticesVBO, particleSystem->cudaParticleVerticesVBO);


	// Set these callbacks after nuklear initialization, otherwise they won't work!
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);



	stringstream ss;
	ss << (vars.useCUDA ? "GPU" : "CPU") << "_";
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
	debugTextureIds.push_back(evsm.getDepthMapTextureId());

	//TextureManager::setOverlayTexture(TextureManager::getTexturePtr("depthMapTexture"), 0);
	//TextureManager::setOverlayTexture(TextureManager::getTexturePtr("harrisTexture"), 1);

	TextureManager::setOverlayTexture(TextureManager::getTexturePtr("lightTexture[0]"), 0);
	//TextureManager::setOverlayTexture(TextureManager::getTexturePtr("lightTexture[1]"), 1);
	TextureManager::setOverlayTexture(TextureManager::getTexturePtr("imageTexture"), 1);

	// Provisional settings
	ui->dirLight = dirLight;
	ui->evsm = &evsm;
	ui->lbm = lbm;
	ui->particleRenderer = particleRenderer;
	ui->particleSystem = particleSystem;
	ui->stlpDiagram = &stlpDiagram;
	ui->stlpSimCUDA = stlpSimCUDA;
	ui->hosek = hosek;
	ui->sps = streamlineParticleSystem;

	while (!glfwWindowShouldClose(window) && vars.appRunning) {
		// enable flags each frame because nuklear disables them when it is rendered	
		//glEnable(GL_DEPTH_TEST);

		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_PROGRAM_POINT_SIZE);
		//glEnable(GL_CULL_FACE);
		
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

		//constructUserInterface(ctx/*, particlesColor*/);


		if (vars.measureTime) {
			vars.timer.clockAvgStart();
		}



		if (vars.showOverlayDiagram) {
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			view = overlayDiagramCamera->getViewMatrix();
			ShaderManager::updatePVMatrixUniforms(overlayDiagramProjection, view);


			GLint res = stlpDiagram.textureResolution;
			glViewport(0, 0, res, res);
			glBindFramebuffer(GL_FRAMEBUFFER, stlpDiagram.diagramMultisampledFramebuffer);
			glClear(GL_COLOR_BUFFER_BIT);
			//glBindTextureUnit(0, stlpDiagram.diagramTexture);

			stlpDiagram.draw(*curveShader, *singleColorShaderVBO);

			stlpDiagram.drawText(*textShader);

			particleSystem->drawDiagramParticles(curveShader);



			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, stlpDiagram.diagramFramebuffer);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, stlpDiagram.diagramMultisampledFramebuffer);


			//glDrawBuffer(GL_BACK);


			glBlitFramebuffer(0, 0, res, res, 0, 0, res, res, GL_COLOR_BUFFER_BIT, GL_NEAREST);


			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

			CHECK_GL_ERRORS();

		}


		glViewport(0, 0, vars.screenWidth, vars.screenHeight);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		//glClear(GL_COLOR_BUFFER_BIT);



		//cout << " Delta time = " << (deltaTime * 1000.0f) << " [ms]" << endl;
		//cout << " Framerate = " << (1.0f / deltaTime) << endl;
		if (mode == 0 || mode == 1) {
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			glfwSwapInterval(1);
			camera = diagramCamera;
			glDisable(GL_DEPTH_TEST);
		} else {
			glClearColor(vars.bgClearColor.x, vars.bgClearColor.y, vars.bgClearColor.z, 1.0f);
			glfwSwapInterval(0);
			camera = vars.useFreeRoamCamera ? freeRoamCamera : viewportCamera;
			glEnable(GL_DEPTH_TEST);


		}
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// UPDATE SHADER VIEW MATRICES

		view = camera->getViewMatrix();

		ShaderManager::updateViewMatrixUniforms(view);
		ShaderManager::updateDirectionalLightUniforms(*dirLight);
		ShaderManager::updateViewPositionUniforms(camera->position);

		//dirLightOnlyShader->use();
		//dirLightOnlyShader->setVec3("v_ViewPos", camera->position);


		if (vars.drawSkybox) {
			projection = glm::perspective(glm::radians(vars.fov), (float)vars.screenWidth / vars.screenHeight, nearPlane, farPlane);

			if (mode == 2 || mode == 3) {
				if (vars.hosekSkybox) {

					hosekShader->use();
					glm::mat4 tmpView = glm::mat4(glm::mat3(view));
					hosekShader->setViewMatrix(tmpView);
					hosekShader->setProjectionMatrix(projection);
					hosekShader->setVec3("u_SunDir", -dirLight->getDirection());
					hosek->draw();


				} else {
					skyboxShader->use();
					glm::mat4 tmpView = glm::mat4(glm::mat3(view));
					skyboxShader->setMat4fv("u_View", tmpView);
					skyboxShader->setMat4fv("u_Projection", projection);
					skybox->draw(*skyboxShader);
				}
			}
		}

		refreshProjectionMatrix();

		CHECK_GL_ERRORS();




		if (mode == 0 || mode == 1) {

			/*	if (mode == 1) {
					stlpSim->doStep();
				}
	*/
			stlpDiagram.draw(*curveShader, *singleColorShaderVBO);
			stlpDiagram.drawText(*textShader);

			/*glUseProgram(diagramShader->id);
			glBindTextureUnit(0, stlpDiagram.diagramTexture);
			glUniform1i(glGetUniformLocation(diagramShader->id, "u_Texture"), 0);
			glUniform2i(glGetUniformLocation(diagramShader->id, "u_ScreenSize"), screenWidth, screenHeight);


			glBindVertexArray(stlpDiagram.overlayDiagramVAO);
			glDrawArrays(GL_TRIANGLES, 0, 6);*/



			//Show2DTexture(stlpDiagram.diagramTexture, 0, 0, 200, 200);

		} else if (mode == 2) {

			if (!vars.paused) {
				if (vars.useCUDA) {
					lbm->doStepCUDA();
				} else {
					lbm->doStep();
				}
			}

			if (vars.measureTime) {
				if (vars.useCUDA) {
					lbm->synchronize();
					//cudaDeviceSynchronize();
				}
				if (vars.timer.clockAvgEnd() && vars.exitAfterFirstAvg) {
					cout << "Exiting main loop..." << endl;
					break;
				}
			}

			// DRAW SCENE
			//grid->draw(*singleColorShader);


			lbm->draw(*singleColorShader);

			//if (vars.usePointSprites) {
			//	particleSystemLBM->draw(*pointSpriteTestShader, vars.useCUDA);
			//} else if (lbm->visualizeVelocity) {
			//	particleSystemLBM->draw(*coloredParticleShader, vars.useCUDA);
			//} else {
			//	particleSystemLBM->draw(*singleColorShader, vars.useCUDA);
			//}
			gGrid.draw(*unlitColorShader);

			/*glDisable(GL_DEPTH_TEST);
			glUseProgram(diagramShader->id);
			glBindTextureUnit(0, stlpDiagram.diagramTexture);
			glUniform1i(glGetUniformLocation(diagramShader->id, "u_Texture"), 0);
			glUniform2i(glGetUniformLocation(diagramShader->id, "u_ScreenSize"), screenWidth, screenHeight);

			glBindVertexArray(stlpDiagram.overlayDiagramVAO);
			glDrawArrays(GL_TRIANGLES, 0, 6);
			glEnable(GL_DEPTH_TEST);*/

			//display2DTexture(stlpDiagram.diagramTexture, diagramShader->id, 0, 0, 200, 200);



			//stlpDiagram.drawOverlayDiagram(diagramShader);



		} else if (mode == 3) {

			if (hosek->liveRecalc) {
				hosek->update(dirLight->getDirection());
			}

			if (vars.stlpUseCUDA) {
				if (vars.applyLBM) {
					if (totalFrameCounter % vars.lbmStepFrame == 0) {
						lbm->doStepCUDA();
					}
				}

				//particleSystem->update();
				particleSystem->doStep();

				if (vars.applySTLP) {
					if (totalFrameCounter % vars.stlpStepFrame == 0) {
						stlpSimCUDA->doStep();
					}
				}
			} else {
				//stlpSim->doStep();
			}

			//lbm->doStepCUDA();


			if (vars.simulateSun) {
				dirLight->circularMotionStep(deltaTime);
			}


			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);

			evsm.preFirstPass();
			vars.heightMap->drawGeometry(evsm.firstPassShaders[0]);
			//stlpSim->heightMap->drawGeometry(evsm.firstPassShader);

			//if (vars.cloudsCastShadows) {
			//	particleSystem->drawGeometry(evsm.firstPassShaders[0], camera->position);
			//}
			//testMesh.draw(evsm.firstPassShader);
			testModel.drawGeometry(evsm.firstPassShaders[0]);
			//grassModel.drawGeometry(evsm.firstPassShaders[0]);
			armoireModel.drawGeometry(evsm.firstPassShaders[0]);

			if (vars.drawTrees) {
				treeModel.drawGeometry(evsm.firstPassShaders[0]);
			}

			evsm.postFirstPass();
			CHECK_GL_ERRORS();


			//stlpDiagram.drawOverlayDiagram(diagramShader, evsm.depthMapTexture);

			//glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);




			evsm.preSecondPass(vars.screenWidth, vars.screenHeight);
			CHECK_GL_ERRORS();

			//stlpSim->heightMap->draw(evsm.secondPassShader);
			//vars.heightMap->draw(evsm.secondPassShader);
			vars.heightMap->draw();

			if (vars.visualizeTerrainNormals) {
				vars.heightMap->drawGeometry(visualizeNormalsShader);
			}

			//testModel.draw(*evsm.secondPassShader);
			testModel.draw();

			if (vars.drawGrass) {
				grassModel.draw();
			}

			testMesh.draw();
			armoireModel.draw();


			if (vars.drawTrees) {
				treeModel.draw();
			}

			evsm.postSecondPass();

			CHECK_GL_ERRORS();

			ShaderManager::updateFogUniforms();


			//glCullFace(GL_FRONT);
			//stlpSim->heightMap->draw();
			dirLight->draw();

			if (!vars.renderMode) {
				particleSystem->drawHelperStructures();

				lbm->draw();
				//grid->draw(*singleColorShader);
				gGrid.draw(*unlitColorShader);

				streamlineParticleSystem->draw();
			}


			if (!particleRenderer->compositeResultToFramebuffer) {
				if (vars.usePointSprites) {
					glEnable(GL_BLEND);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

					glDepthMask(GL_FALSE);
					particleSystem->draw(*pointSpriteTestShader, camera->position);
					glDepthMask(GL_TRUE);
				} else {
					particleSystem->draw(*singleColorShader, camera->position);

				}
			}

			if (!vars.renderMode) {
				stlpSimCUDA->draw(camera->position);
			}


			//if (vars.renderVolumeParticles) {
				///////////////////////////////////////////////////////////////
				//     NVIDIA VOLUMETRIC PARTICLES (HALF ANGLE SLICING)
				///////////////////////////////////////////////////////////////			


			particleRenderer->recalcVectors(camera, dirLight);
			glm::vec3 sortVec = particleRenderer->getSortVec();

			// NOW sort particles using the sort vector
			particleSystem->sortParticlesByProjection(sortVec, eSortPolicy::LEQUAL);

			particleRenderer->preSceneRenderImage();
			vars.heightMap->draw();
			particleRenderer->postSceneRenderImage();


			particleRenderer->render(particleSystem, dirLight, camera);

			/*
							glDisable(GL_BLEND);
							glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
							glEnable(GL_DEPTH_TEST);*/
							//}

							//stlpDiagram.drawOverlayDiagram(diagramShader, evsm.depthMapTexture);

			if (vars.showOverlayDiagram) {
				stlpDiagram.drawOverlayDiagram(diagramShader);
			}


			//TextureManager::drawOverlayTextures(debugTextureIds);
			TextureManager::drawOverlayTextures();


		}


		CHECK_GL_ERRORS();


		// Render the user interface

		ui->draw();
		//nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
		

		lbm->recalculateVariables(); // recalculate variables based on values set in the user interface

		glfwSwapBuffers(window);

		CHECK_GL_ERRORS();

	}


	//delete particleSystemLBM;
	delete particleSystem;
	delete particleRenderer;
	delete streamlineParticleSystem;
	delete lbm;
	//delete grid;
	//delete viewportCamera;
	delete freeRoamCamera;
	delete diagramCamera;
	delete overlayDiagramCamera;
	delete orbitCamera;
	delete dirLight;

	delete skybox;

	//delete stlpSim;


	//size_t cudaMemFree = 0;
	//size_t cudaMemTotal = 0;

	//cudaMemGetInfo(&cudaMemFree, &cudaMemTotal);

	/*
	cout << " FREE CUDA MEMORY  = " << cudaMemFree << endl;
	cout << " TOTAL CUDA MEMORY = " << cudaMemTotal << endl;
	*/

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
	if (mode == 0 || mode == 1) {
		//projection = glm::ortho(-0.2f, 1.2f, 1.2f, -0.2f, nearPlane, farPlane);
		projection = diagramProjection;
		//camera->movementSpeed = 4.0f;
	} else {
		if (vars.projectionMode == ORTHOGRAPHIC) {
			projection = viewportProjection;
		} else {
			projection = glm::perspective(glm::radians(vars.fov), (float)vars.screenWidth / vars.screenHeight, nearPlane, farPlane);
		}
		//mode = 2;
		//camera->movementSpeed = 40.0f;
	}

	ShaderManager::updateProjectionMatrixUniforms(projection);
}




void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
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
		mode = 0;
		glDisable(GL_DEPTH_TEST); // painters algorithm for now
		refreshProjectionMatrix();
	}
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
		mode = 1;
		glDisable(GL_DEPTH_TEST); // painters algorithm for now
		refreshProjectionMatrix();
	}
	if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
		mode = 2;
		glEnable(GL_DEPTH_TEST);
		refreshProjectionMatrix();
	}
	if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
		mode = 3;
		glEnable(GL_DEPTH_TEST);
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


	if (mouseDown) {
		//cout << "mouse down" << endl;
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
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

		//stlpDiagram.findClosestSoundingPoint(mouseCoords);

		stlpDiagram.moveSelectedPoint(mouseCoords);

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
	if (ui->isAnyWindowHovered() || mode >= 2) {
		//cout << "Mouse callback not valid, hovering over Nuklear window/widget." << endl;
		return;
	}

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
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

		stlpDiagram.findClosestSoundingPoint(mouseCoords);

		mouseDown = true;
	} else if (action == GLFW_RELEASE) {
		mouseDown = false;

	}
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	//if (ui->isAnyWindowHovered() || mode < 2) {
	//	return;
	//}

	if (mode < 2) {
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

	//if (camera == freeRoamCamera) {
	//	cout << "free roam camera???" << endl;
	//	freeRoamCamera->processMouseMovement(xOffset, yOffset);
	//}

	// for easier controls
	if (vars.consumeMouseCursor) {
		camera->processMouseMovement(xOffset, yOffset, false);
	}
}




void window_size_callback(GLFWwindow* window, int width, int height) {
	float aspectRatio = (float)width / (float)height;

	vars.screenWidth = width;
	vars.screenHeight = height;

	float offset = 0.2f;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);

	//cout << "Aspect ratio = " << aspectRatio << endl;


	if (vars.lbmType == LBM2D) {
		if (vars.latticeWidth >= vars.latticeHeight) {
			projWidth = (float)vars.latticeWidth;
			projHeight = projWidth / aspectRatio;
		} else {
			projHeight = (float)vars.latticeHeight;
			projWidth = projHeight * aspectRatio;
		}
		viewportProjection = glm::ortho(-1.0f, projWidth, -1.0f, projHeight, nearPlane, farPlane);
	} else {
		projHeight = projectionRange;
		projWidth = projHeight * aspectRatio;
		viewportProjection = glm::ortho(-projWidth, projWidth, -projHeight, projHeight, nearPlane, farPlane);

	}
	TextureManager::refreshOverlayTextures();
	stlpDiagram.refreshOverlayDiagram(vars.screenWidth, vars.screenHeight);
}