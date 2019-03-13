
#include <iostream>


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

//#include "LBM.h"
//#include "LBM2D_1D_indices.h"
#include "LBM3D_1D_indices.h"
#include "HeightMap.h"
#include "Grid2D.h"
#include "Grid3D.h"
#include "GeneralGrid.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "Camera2D.h"
#include "OrbitCamera.h"
#include "FreeRoamCamera.h"
#include "ParticleSystemLBM.h"
#include "ParticleSystem.h"
#include "DirectionalLight.h"
#include "Grid.h"
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

//#include <omp.h>	// OpenMP for CPU parallelization

//#include <vld.h>	// Visual Leak Detector for memory leaks analysis

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

///// NUKLEAR /////////////////////////////////////////////////////////////////
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#define NK_KEYSTATE_BASED_INPUT
#include <nuklear.h>
#include "nuklear_glfw_gl3.h"

#define INCLUDE_STYLE
#ifdef INCLUDE_STYLE
#include "nuklear/style.c"
#endif

#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024


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
void constructUserInterface(nk_context *ctx, nk_colorf &particlesColor);


void refreshProjectionMatrix();



///////////////////////////////////////////////////////////////////////////////////////////////////
///// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////////////////////////


VariableManager vars;

//eLBMType lbmType;		///< The LBM type that is to be displayed

LBM3D_1D_indices *lbm;				///< Pointer to the current LBM
Grid *grid;				///< Pointer to the current grid
Camera *camera;			///< Pointer to the current camera
ParticleSystemLBM *particleSystemLBM;		///< Pointer to the particle system that is to be used throughout the whole application
ParticleSystem *particleSystem;

//HeightMap *heightMap;

//Timer timer;
Camera *viewportCamera;
Camera *freeRoamCamera;
Camera *orbitCamera;
Camera2D *diagramCamera;
Camera2D *overlayDiagramCamera;

STLPSimulator *stlpSim;
STLPSimulatorCUDA *stlpSimCUDA;

EVSMShadowMapper evsm;
DirectionalLight dirLight;
int uiMode = 1;

float fov = 90.0f;

float lastMouseX;
float lastMouseY;



struct nk_context *ctx;

int projectionMode = PERSPECTIVE;
int drawSkybox = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////
///// DEFAULT VALUES THAT ARE TO BE REWRITTEN FROM THE CONFIG FILE
///////////////////////////////////////////////////////////////////////////////////////////////////
double deltaTime = 0.0;		///< Delta time of the current frame
double lastFrameTime;		///< Duration of the last frame

glm::mat4 view;				///< View matrix
glm::mat4 projection;		///< Projection matrix
glm::mat4 viewportProjection;
glm::mat4 diagramProjection;
glm::mat4 overlayDiagramProjection;

float nearPlane = 0.1f;		///< Near plane of the view frustum
float farPlane = 1000.0f;	///< Far plane of the view frustum

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

STLPDiagram stlpDiagram;	///< SkewT/LogP diagram instance
int mode = 3;				///< Mode: 0 - show SkewT/LogP diagram, 1 - show 3D simulator

Skybox *skybox;


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
	evsm.init();
	dirLight.init();

	skybox = new Skybox();


	glViewport(0, 0, vars.screenWidth, vars.screenHeight);

	float aspectRatio = (float)vars.screenWidth / (float)vars.screenHeight;
	cout << "Aspect ratio = " << aspectRatio << endl;

	float offset = 0.2f;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);
	overlayDiagramProjection = glm::ortho(0.0f - offset, 1.0f + offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);

	ctx = nk_glfw3_init(window, NK_GLFW3_INSTALL_CALLBACKS);


	{
		struct nk_font_atlas *atlas;
		nk_glfw3_font_stash_begin(&atlas);
		struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/Roboto-Regular.ttf", 14, 0);
		nk_glfw3_font_stash_end();
		nk_style_load_all_cursors(ctx, atlas->cursors);
		nk_style_set_font(ctx, &roboto->handle);
	}


#ifdef INCLUDE_STYLE
	set_style(ctx, THEME_MARTIN);
#endif

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
	diagramShader = ShaderManager::getShaderPtr("diagram");

	textShader = ShaderManager::getShaderPtr("text");
	curveShader = ShaderManager::getShaderPtr("curve");
	skyboxShader = ShaderManager::getShaderPtr("skybox");

	vars.heightMap = new HeightMap(vars.sceneFilename, vars.latticeHeight, dirLightOnlyShader);


	struct nk_colorf particlesColor;


	particleSystemLBM = new ParticleSystemLBM(vars.numParticles, vars.drawStreamlines);
	particleSystem = new ParticleSystem(&vars);


	//particlesColor.r = particleSystemLBM->particlesColor.r;
	//particlesColor.g = particleSystemLBM->particlesColor.g;
	//particlesColor.b = particleSystemLBM->particlesColor.b;


	glm::ivec3 latticeDim(vars.latticeWidth, vars.latticeHeight, vars.latticeDepth);

	float ratio = (float)vars.screenWidth / (float)vars.screenHeight;



	// Create and configure the simulator, select from 2D and 3D options and set parameters accordingly
	{
		printf("LBM3D SETUP...\n");

		dim3 blockDim(vars.blockDim_3D_x, vars.blockDim_3D_y, 1);
		CHECK_ERROR(cudaPeekAtLastError());

		lbm = new LBM3D_1D_indices(&vars, latticeDim, vars.sceneFilename, vars.tau, particleSystemLBM, particleSystem, blockDim);
		CHECK_ERROR(cudaPeekAtLastError());


		vars.latticeWidth = lbm->latticeWidth;
		vars.latticeHeight = lbm->latticeHeight;
		vars.latticeDepth = lbm->latticeDepth;



		projectionRange = (float)((vars.latticeWidth > vars.latticeHeight) ? vars.latticeWidth : vars.latticeHeight);
		projectionRange = (projectionRange > vars.latticeDepth) ? projectionRange : vars.latticeDepth;
		projectionRange /= 2.0f;

		projHeight = projectionRange;
		projWidth = projHeight * ratio;

		//projection = glm::ortho(-projectionRange, projectionRange, -projectionRange, projectionRange, nearPlane, farPlane);
		projection = glm::ortho(-projWidth, projWidth, -projHeight, projHeight, nearPlane, farPlane);
		grid = new Grid3D(vars.latticeWidth, vars.latticeHeight, vars.latticeDepth, 6, 6, 6);
		float cameraRadius = sqrtf((float)(vars.latticeWidth * vars.latticeWidth + vars.latticeDepth * vars.latticeDepth)) + 10.0f;
		//camera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 80.0f, glm::vec3(vars.latticeWidth / 2.0f, vars.latticeHeight / 2.0f, vars.latticeDepth / 2.0f), cameraRadius);
		orbitCamera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 80.0f, glm::vec3(vars.latticeWidth / 2.0f, vars.latticeHeight / 2.0f, vars.latticeDepth / 2.0f), cameraRadius);
	}

	CHECK_ERROR(cudaPeekAtLastError());

	viewportCamera = orbitCamera;
	camera = viewportCamera;
	diagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	overlayDiagramCamera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
	freeRoamCamera = new FreeRoamCamera(glm::vec3(30.0f, 50.0f, 30.0f), WORLD_UP, -35.0f, -35.0f);


	viewportProjection = projection;

	camera->setLatticeDimensions(vars.latticeWidth, vars.latticeHeight, vars.latticeDepth);
	camera->movementSpeed = vars.cameraSpeed;

	dirLight.focusPoint = glm::vec3(vars.latticeWidth / 2.0f, 0.0f, vars.latticeDepth / 2.0f);

	//particleSystemLBM->lbm = lbm;





	//StaticMesh testMesh("models/House_3.obj", ShaderManager::getShaderPtr("dirLightOnly"), nullptr);
	//Model testModel("models/House_3.obj");

	Texture diffuse("textures/body2.png", 0);
	Texture specular("textures/body2_S.png", 1);
	Texture normal("textures/body2_N.png", 2);
	Material testMat(diffuse, specular, normal, 32.0f);

	Model testModel("models/housewife.obj", &testMat, ShaderManager::getShaderPtr("normals"));
	StaticMesh testMesh("models/housewife.obj", ShaderManager::getShaderPtr("normals"), &testMat);

	Texture adiffuse("textures/armoire/albedo.png", 0);
	Texture aspecular("textures/armoire/metallic.png", 1);
	Texture anormal("textures/armoire/normal.png", 2);
	Material aMat(adiffuse, aspecular, anormal, 32.0f);

	Model armoireModel("models/armoire.fbx", &aMat, ShaderManager::getShaderPtr("normals"));
	armoireModel.transform.position.x += 2.0f;

	//Model testModel("models/housewife.obj", &testMat, dirLightOnlyShader);
	//StaticMesh testMesh("models/housewife.obj", dirLightOnlyShader, &testMat);
	testModel.transform.position.x += 1.0f;

	//testMesh.transform.position = glm::vec3(0.0f);




	dirLight.position = glm::vec3(100.0f, 60.0f, 60.0f);




	evsm.dirLight = &dirLight;


	refreshProjectionMatrix();


	GeneralGrid gGrid(100, 5, (vars.lbmType == LBM3D));


	int frameCounter = 0;
	glfwSwapInterval(vars.vsync); // VSync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)
	
	double prevTime = glfwGetTime();
	int totalFrameCounter = 0;
	//int measurementFrameCounter = 0;
	//double accumulatedTime = 0.0;


	stlpDiagram.init(vars.soundingFile);
	CHECK_ERROR(cudaPeekAtLastError());



	stlpSim = new STLPSimulator(&vars, &stlpDiagram);
	stlpSimCUDA = new STLPSimulatorCUDA(&vars, &stlpDiagram);

	CHECK_ERROR(cudaPeekAtLastError());

	stlpSim->initParticles();
	//stlpSimCUDA->initParticles();
	stlpSimCUDA->initCUDA();

	CHECK_ERROR(cudaPeekAtLastError());


	particleSystem->stlpSim = stlpSimCUDA;
	stlpSimCUDA->particleSystem = particleSystem;
	particleSystem->initParticlesOnTerrain();
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
	ss << "_" << particleSystemLBM->numParticles;
	vars.timer.configString = ss.str();
	if (vars.measureTime) {
		vars.timer.start();
	}
	double accumulatedTime = 0.0;

	glActiveTexture(GL_TEXTURE0);

	CHECK_ERROR(cudaPeekAtLastError());

	while (!glfwWindowShouldClose(window) && vars.appRunning) {
		// enable flags each frame because nuklear disables them when it is rendered	
		//glEnable(GL_DEPTH_TEST);

		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_PROGRAM_POINT_SIZE);
		//glEnable(GL_CULL_FACE);

		reportGLErrors("->>> LOOP START <<<-");


		double currentFrameTime = glfwGetTime();
		deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;
		frameCounter++;
		totalFrameCounter++;
		accumulatedTime += deltaTime;

		if (currentFrameTime - prevTime >= 1.0f) {
			printf("Avg delta time = %0.4f [ms]\n", 1000.0 * (accumulatedTime / frameCounter));
			prevTime += (currentFrameTime - prevTime);
			frameCounter = 0;
			accumulatedTime = 0.0;
		}

		glfwPollEvents();
		processInput(window);
		constructUserInterface(ctx, particlesColor);


		if (vars.measureTime) {
			vars.timer.clockAvgStart();
		}



		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		view = overlayDiagramCamera->getViewMatrix();
		ShaderManager::updatePVMatrixUniforms(overlayDiagramProjection, view);
		reportGLErrors("B1");


		GLint res = stlpDiagram.textureResolution;
		glViewport(0, 0, res, res);
		glBindFramebuffer(GL_FRAMEBUFFER, stlpDiagram.diagramMultisampledFramebuffer);
		glClear(GL_COLOR_BUFFER_BIT);
		//glBindTextureUnit(0, stlpDiagram.diagramTexture);
		reportGLErrors("B2");

		stlpDiagram.draw(*curveShader, *singleColorShaderVBO);
		reportGLErrors("B3");

		stlpDiagram.drawText(*textShader);
		reportGLErrors("B4");


		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, stlpDiagram.diagramFramebuffer);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, stlpDiagram.diagramMultisampledFramebuffer);

		reportGLErrors("B5");

		//glDrawBuffer(GL_BACK);
		reportGLErrors("B6");


		glBlitFramebuffer(0, 0, res, res, 0, 0, res, res, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		reportGLErrors("B7");


		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		reportGLErrors("B8");


		glViewport(0, 0, vars.screenWidth, vars.screenHeight);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT);
		reportGLErrors("B");



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

		reportGLErrors("C");

		// UPDATE SHADER VIEW MATRICES

		view = camera->getViewMatrix();

		reportGLErrors("D0");

		ShaderManager::updateViewMatrixUniforms(view);
		ShaderManager::updateDirectionalLightUniforms(dirLight);
		ShaderManager::updateViewPositionUniforms(camera->position);
		reportGLErrors("D1");

		//dirLightOnlyShader->use();
		//dirLightOnlyShader->setVec3("v_ViewPos", camera->position);
		
		reportGLErrors("D2");

		if (drawSkybox) {
			projection = glm::perspective(glm::radians(fov), (float)vars.screenWidth / vars.screenHeight, nearPlane, farPlane);

			if (mode == 2 || mode == 3) {
				skyboxShader->use();
				glm::mat4 tmpView = glm::mat4(glm::mat3(view));
				skyboxShader->setMat4fv("u_View", tmpView);
				skyboxShader->setMat4fv("u_Projection", projection);
				skybox->draw(*skyboxShader);
			}
		}

		refreshProjectionMatrix();

		reportGLErrors("D");



		if (mode == 0 || mode == 1) {

			if (mode == 1) {
				stlpSim->doStep();
			}

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
			grid->draw(*singleColorShader);

			
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


			if (vars.stlpUseCUDA) {
				if (vars.applyLBM) {
					lbm->doStepCUDA();
				}

				//particleSystem->update();
				particleSystem->doStep();

				if (vars.applySTLP) {
					stlpSimCUDA->doStep();
				}
			} else {
				stlpSim->doStep();
			}

			//lbm->doStepCUDA();


			reportGLErrors("1");


			grid->draw(*singleColorShader);

			reportGLErrors("2");

			gGrid.draw(*unlitColorShader);

			reportGLErrors("3");

			if (vars.simulateSun) {
				dirLight.circularMotionStep(deltaTime);
			}

			
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
			//glDepthFunc(GL_LEQUAL);

			glDisable(GL_CULL_FACE);
			evsm.preFirstPass();
			stlpSim->heightMap->drawGeometry(evsm.firstPassShader);
			//testMesh.draw(evsm.firstPassShader);

			evsm.postFirstPass();

			glViewport(0, 0, vars.screenWidth, vars.screenHeight);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			//stlpDiagram.drawOverlayDiagram(diagramShader, evsm.depthMapTexture);

			//glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);


			evsm.preSecondPass(vars.screenWidth, vars.screenHeight);

			stlpSim->heightMap->draw(evsm.secondPassShader);


			//testModel.draw(*evsm.secondPassShader);
			testModel.draw();
			testMesh.draw();
			armoireModel.draw();

			evsm.postSecondPass();
			
			ShaderManager::updateFogUniforms();


			//glCullFace(GL_FRONT);
			//stlpSim->heightMap->draw();
			dirLight.draw();


			glEnable(GL_BLEND);
			glDepthMask(GL_FALSE);

			if (vars.stlpUseCUDA) {
				if (vars.usePointSprites) {
					stlpSimCUDA->draw(*pointSpriteTestShader, camera->position);
				} else {
					stlpSimCUDA->draw(*singleColorShader, camera->position);
				}
			} else {
				if (vars.usePointSprites) {
					stlpSim->draw(*pointSpriteTestShader, camera->position);
				} else {
					stlpSim->draw(*singleColorShader, camera->position);
				}
			}
			particleSystem->draw(*pointSpriteTestShader, camera->position);
			glDepthMask(GL_TRUE);

			//stlpDiagram.drawOverlayDiagram(diagramShader, evsm.depthMapTexture);


			stlpDiagram.drawOverlayDiagram(diagramShader);
			

			

		}
		reportGLErrors("E");


		// DRAW SCENE
		/*grid->draw(*singleColorShader);
		lbm->draw(*singleColorShader);

		if (usePointSprites) {
			particleSystem->draw(*pointSpriteTestShader, useCUDA);
		} else if (lbm->visualizeVelocity) {
			particleSystem->draw(*coloredParticleShader, useCUDA);
		} else {
			particleSystem->draw(*singleColorShader, useCUDA);
		}
		gGrid.draw(*unlitColorShader);*/

		// Render the user interface
		nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);

		lbm->recalculateVariables(); // recalculate variables based on values set in the user interface

		glfwSwapBuffers(window);

		reportGLErrors("->>> LOOP END <<<-");

	}


	//delete particleSystemLBM;
	delete particleSystem;
	delete lbm;
	delete grid;
	//delete viewportCamera;
	delete freeRoamCamera;
	delete diagramCamera;
	delete overlayDiagramCamera;
	delete orbitCamera;

	delete skybox;

	delete stlpSim;


	//size_t cudaMemFree = 0;
	//size_t cudaMemTotal = 0;

	//cudaMemGetInfo(&cudaMemFree, &cudaMemTotal);

	/*
	cout << " FREE CUDA MEMORY  = " << cudaMemFree << endl;
	cout << " TOTAL CUDA MEMORY = " << cudaMemTotal << endl;
	*/

	ShaderManager::tearDown();


	nk_glfw3_shutdown();
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
		camera->movementSpeed = 4.0f;
	} else {
		if (projectionMode == ORTHOGRAPHIC) {
			projection = viewportProjection;
		} else {
			projection = glm::perspective(glm::radians(fov), (float)vars.screenWidth / vars.screenHeight, nearPlane, farPlane);
		}
		//mode = 2;
		camera->movementSpeed = 40.0f;
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
	if (nk_window_is_any_hovered(ctx) || mode >= 2) {
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
	//if (nk_window_is_any_hovered(ctx) || mode < 2) {
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



void constructUserInterface(nk_context *ctx, nk_colorf &particlesColor) {
	nk_glfw3_new_frame();

	//ctx->style.window.padding = nk_vec2(10.0f, 10.0f);
	ctx->style.window.padding = nk_vec2(0.0f, 0.0f);


	/* GUI */
	if (nk_begin(ctx, "Control Panel", nk_rect(50, 50, 275, 700),
				 NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
				 NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {
		enum { EASY, HARD };
		//static int op = EASY;
		//static int property = 20;

		nk_layout_row_dynamic(ctx, 30, 2);
		if (nk_button_label(ctx, "LBM")) {
			uiMode = 0;
		}
		if (nk_button_label(ctx, "Shadows")) {
			uiMode = 1;
		}


		if (uiMode == 0) {
			nk_layout_row_dynamic(ctx, 30, 1);
			nk_label(ctx, "LBM Controls", NK_TEXT_CENTERED);

			nk_layout_row_static(ctx, 30, 80, 3);
			if (nk_button_label(ctx, "Reset")) {
				//fprintf(stdout, "button pressed\n");
				lbm->resetSimulation();
			}
			const char *buttonDescription = vars.paused ? "Play" : "Pause";
			if (nk_button_label(ctx, buttonDescription)) {
				vars.paused = !vars.paused;
			}
			if (nk_button_label(ctx, "EXIT")) {
				vars.appRunning = false;
			}


#ifdef LBM_EXPERIMENTAL
			nk_layout_row_dynamic(ctx, 30, 1);
			nk_label_colored_wrap(ctx, "Enabling or disabling CUDA at runtime is highly unstable at the moment, use at your own discretion", nk_rgba_f(1.0f, 0.5f, 0.5f, 1.0f));

			bool useCUDAPrev = useCUDA;
			nk_checkbox_label(ctx, "Use CUDA", &useCUDACheckbox);
			useCUDA = useCUDACheckbox;
			if (useCUDAPrev != useCUDA && useCUDA == false) {
				lbm->switchToCPU();
			}
#endif

			nk_layout_row_dynamic(ctx, 15, 1);

			nk_property_float(ctx, "Tau:", 0.5005f, &lbm->tau, 10.0f, 0.005f, 0.005f);

			int mirrorSidesPrev = lbm->mirrorSides;
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_checkbox_label(ctx, "Mirror sides", &lbm->mirrorSides);
			if (mirrorSidesPrev != lbm->mirrorSides) {
				cout << "Mirror sides value changed!" << endl;
				lbm->updateControlProperty(LBM3D_1D_indices::MIRROR_SIDES_PROP);
			}

#ifdef LBM_EXPERIMENTAL
			if (lbmType == LBM3D) {
				nk_layout_row_dynamic(ctx, 15, 1);
				nk_checkbox_label(ctx, "Use subgrid model", &lbm->useSubgridModel);
			}
#endif



			nk_layout_row_dynamic(ctx, 15, 1);
			//nk_label(ctx, "Use point sprites", NK_TEXT_LEFT);
			int prevVsync = vars.vsync;
			nk_checkbox_label(ctx, "VSync", &vars.vsync);
			if (prevVsync != vars.vsync) {
				glfwSwapInterval(vars.vsync);
			}

			nk_label(ctx, "Inlet velocity:", NK_TEXT_LEFT);

			nk_layout_row_dynamic(ctx, 15, (vars.lbmType == LBM2D) ? 2 : 3);
			nk_property_float(ctx, "x:", -1.0f, &lbm->inletVelocity.x, 1.0f, 0.01f, 0.005f);
			nk_property_float(ctx, "y:", -1.0f, &lbm->inletVelocity.y, 1.0f, 0.01f, 0.005f);
			if (vars.lbmType == LBM3D) {
				nk_property_float(ctx, "z:", -1.0f, &lbm->inletVelocity.z, 1.0f, 0.01f, 0.005f);
			}


			nk_layout_row_dynamic(ctx, 15, 1);
			//nk_label(ctx, "Use point sprites", NK_TEXT_LEFT);
			nk_checkbox_label(ctx, "Use point sprites", &vars.usePointSprites);

			if (/*lbmType == LBM2D &&*/ vars.useCUDA && !vars.usePointSprites) {
				nk_layout_row_dynamic(ctx, 15, 1);
				nk_checkbox_label(ctx, "Visualize velocity", &lbm->visualizeVelocity);
			}

			if (!vars.useCUDA) {
				nk_layout_row_dynamic(ctx, 15, 1);
				nk_checkbox_label(ctx, "Respawn linearly", &lbm->respawnLinearly);
			}
/*
			nk_layout_row_dynamic(ctx, 10, 1);
			nk_labelf(ctx, NK_TEXT_LEFT, "Point size");
			nk_slider_float(ctx, 1.0f, &particleSystemLBM->pointSize, 100.0f, 0.5f);

			if (!vars.usePointSprites && !lbm->visualizeVelocity) {
				nk_layout_row_dynamic(ctx, 20, 1);
				nk_label(ctx, "Particles Color:", NK_TEXT_LEFT);
				nk_layout_row_dynamic(ctx, 25, 1);
				if (nk_combo_begin_color(ctx, nk_rgb_cf(particlesColor), nk_vec2(nk_widget_width(ctx), 400))) {
					nk_layout_row_dynamic(ctx, 120, 1);
					particlesColor = nk_color_picker(ctx, particlesColor, NK_RGBA);
					nk_layout_row_dynamic(ctx, 25, 1);
					particlesColor.r = nk_propertyf(ctx, "#R:", 0, particlesColor.r, 1.0f, 0.01f, 0.005f);
					particlesColor.g = nk_propertyf(ctx, "#G:", 0, particlesColor.g, 1.0f, 0.01f, 0.005f);
					particlesColor.b = nk_propertyf(ctx, "#B:", 0, particlesColor.b, 1.0f, 0.01f, 0.005f);
					particlesColor.a = nk_propertyf(ctx, "#A:", 0, particlesColor.a, 1.0f, 0.01f, 0.005f);
					particleSystemLBM->particlesColor = glm::vec3(particlesColor.r, particlesColor.g, particlesColor.b);
					nk_combo_end(ctx);
				}
			}*/
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_label(ctx, "Camera movement speed", NK_TEXT_LEFT);
			nk_slider_float(ctx, 1.0f, &camera->movementSpeed, 400.0f, 1.0f);


			nk_layout_row_dynamic(ctx, 15, 2);
			if (nk_option_label(ctx, "Orthographic", projectionMode == ORTHOGRAPHIC)) {
				projectionMode = ORTHOGRAPHIC;
			}
			if (nk_option_label(ctx, "Perspective", projectionMode == PERSPECTIVE)) {
				projectionMode = PERSPECTIVE;
			}
			if (projectionMode == PERSPECTIVE) {
				nk_slider_float(ctx, 30.0f, &fov, 120.0f, 1.0f);
			}

			nk_layout_row_dynamic(ctx, 15, 2);
			nk_checkbox_label(ctx, "Skybox", &drawSkybox);

			//int useFreeRoamCameraPrev = vars.useFreeRoamCamera;
			nk_checkbox_label(ctx, "Use freeroam camera", &vars.useFreeRoamCamera);
			//if (useFreeRoamCameraPrev != vars.useFreeRoamCamera) {
			//	if (mode >= 2) {
			//		camera = (vars.useFreeRoamCamera) ? freeRoamCamera : viewportCamera;
			//		if (vars.useFreeRoamCamera) {
			//			cout << "using freeRoamCamera from now on" << endl;
			//		}
			//	}
			//}
			//nk_colorf()
			struct nk_colorf tmpColor;
			tmpColor.r = dirLight.color.x;
			tmpColor.g = dirLight.color.y;
			tmpColor.b = dirLight.color.z;

			if (nk_combo_begin_color(ctx, nk_rgb_cf(tmpColor), nk_vec2(nk_widget_width(ctx), 400))) {
				nk_layout_row_dynamic(ctx, 120, 1);
				tmpColor = nk_color_picker(ctx, tmpColor, NK_RGBA);
				nk_layout_row_dynamic(ctx, 25, 1);
				tmpColor.r = nk_propertyf(ctx, "#R:", 0, tmpColor.r, 1.0f, 0.01f, 0.005f);
				tmpColor.g = nk_propertyf(ctx, "#G:", 0, tmpColor.g, 1.0f, 0.01f, 0.005f);
				tmpColor.b = nk_propertyf(ctx, "#B:", 0, tmpColor.b, 1.0f, 0.01f, 0.005f);
				tmpColor.a = nk_propertyf(ctx, "#A:", 0, tmpColor.a, 1.0f, 0.01f, 0.005f);
				dirLight.color = glm::vec3(tmpColor.r, tmpColor.g, tmpColor.b);
				nk_combo_end(ctx);
			}

			nk_layout_row_dynamic(ctx, 15, 1);
			nk_label(ctx, "LBM Respawn Mode", NK_TEXT_CENTERED);
			nk_layout_row_dynamic(ctx, 15, 2);
			if (nk_option_label(ctx, "Keep Position", lbm->respawnMode == LBM3D_1D_indices::KEEP_POSITION)) {
				lbm->respawnMode = LBM3D_1D_indices::KEEP_POSITION;
			}
			if (nk_option_label(ctx, "Random (Uniform)", lbm->respawnMode == LBM3D_1D_indices::RANDOM_UNIFORM)) {
				lbm->respawnMode = LBM3D_1D_indices::RANDOM_UNIFORM;
			}

			nk_layout_row_dynamic(ctx, 15, 1);
			nk_label(ctx, "LBM Out of Bounds Mode", NK_TEXT_CENTERED);
			nk_layout_row_dynamic(ctx, 15, 2);
			if (nk_option_label(ctx, "Ignore Particles", lbm->outOfBoundsMode == LBM3D_1D_indices::KEEP_POSITION)) {
				lbm->outOfBoundsMode = LBM3D_1D_indices::IGNORE_PARTICLES;
			}
			if (nk_option_label(ctx, "Deactivate Particles", lbm->outOfBoundsMode == LBM3D_1D_indices::DEACTIVATE_PARTICLES)) {
				lbm->outOfBoundsMode = LBM3D_1D_indices::DEACTIVATE_PARTICLES;
			}
			if (nk_option_label(ctx, "Respawn Particles in Inlet", lbm->outOfBoundsMode == LBM3D_1D_indices::RESPAWN_PARTICLES_INLET)) {
				lbm->outOfBoundsMode = LBM3D_1D_indices::RESPAWN_PARTICLES_INLET;
			}
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_checkbox_label(ctx, "x left inlet", &lbm->xLeftInlet);
			nk_checkbox_label(ctx, "x right inlet", &lbm->xRightInlet);
			nk_checkbox_label(ctx, "y bottom inlet", &lbm->yBottomInlet);
			nk_checkbox_label(ctx, "y top inlet", &lbm->yTopInlet);
			nk_checkbox_label(ctx, "z left inlet", &lbm->zLeftInlet);
			nk_checkbox_label(ctx, "z right inlet", &lbm->zRightInlet);

			nk_checkbox_label(ctx, "Use subgrid model (experimental)", &vars.useSubgridModel);






		} else if (uiMode == 1) {



			nk_layout_row_dynamic(ctx, 30, 1);

			nk_label(ctx, "Shadow/Light Controls", NK_TEXT_CENTERED);


			nk_layout_row_dynamic(ctx, 15, 1);
			nk_property_float(ctx, "#x:", -1000.0f, &dirLight.position.x, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "#y:", -1000.0f, &dirLight.position.y, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "#z:", -1000.0f, &dirLight.position.z, 1000.0f, 1.0f, 1.0f);


			nk_layout_row_dynamic(ctx, 15, 1);
			nk_property_float(ctx, "focus x:", -1000.0f, &dirLight.focusPoint.x, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "focus y:", -1000.0f, &dirLight.focusPoint.y, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "focus z:", -1000.0f, &dirLight.focusPoint.z, 1000.0f, 1.0f, 1.0f);


			nk_layout_row_dynamic(ctx, 15, 1);
			nk_property_float(ctx, "left:", -1000.0f, &dirLight.pLeft, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "right:", -1000.0f, &dirLight.pRight, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "bottom:", -1000.0f, &dirLight.pBottom, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "top:", -1000.0f, &dirLight.pTop, 1000.0f, 1.0f, 1.0f);

			nk_checkbox_label(ctx, "Simulate sun", &vars.simulateSun);
			nk_property_float(ctx, "Sun speed", 0.1f, &dirLight.circularMotionSpeed, 1000.0f, 0.1f, 0.1f);
			if (nk_option_label(ctx, "y axis", dirLight.rotationAxis == DirectionalLight::Y_AXIS)) {
				dirLight.rotationAxis = DirectionalLight::Y_AXIS;
			}
			if (nk_option_label(ctx, "z axis", dirLight.rotationAxis == DirectionalLight::Z_AXIS)) {
				dirLight.rotationAxis = DirectionalLight::Z_AXIS;
			}
			nk_property_float(ctx, "rotation radius:", 0.0f, &dirLight.radius, 10000.0f, 1.0f, 1.0f);

			nk_label(ctx, "EVSM", NK_TEXT_CENTERED);

			nk_checkbox_label(ctx, "use blur pass:", (int *)&evsm.useBlurPass);
			nk_property_float(ctx, "shadowBias:", 0.0f, &evsm.shadowBias, 1.0f, 0.0001f, 0.0001f);
			nk_property_float(ctx, "light bleed reduction:", 0.0f, &evsm.lightBleedReduction, 1.0f, 0.0001f, 0.0001f);
			//nk_property_float(ctx, "variance min limit:", 0.0f, &evsm.varianceMinLimit, 1.0f, 0.0001f, 0.0001f);
			nk_property_float(ctx, "exponent:", 1.0f, &evsm.exponent, 42.0f, 0.1f, 0.1f);

			nk_checkbox_label(ctx, "shadow only", &evsm.shadowOnly);


			nk_property_float(ctx, "Fog intensity", 0.0f, &vars.fogIntensity, 1.0f, 0.01f, 0.01f);

			nk_property_float(ctx, "Fog min distance", 0.0f, &vars.fogMinDistance, 1000.0f, 0.1f, 0.1f);
			nk_property_float(ctx, "Fog max distance", 0.0f, &vars.fogMaxDistance, 1000.0f, 0.1f, 0.1f);


			struct nk_colorf tmpColor;
			tmpColor.r = vars.fogColor.x;
			tmpColor.g = vars.fogColor.y;
			tmpColor.b = vars.fogColor.z;
			tmpColor.a = vars.fogColor.w;

			if (nk_combo_begin_color(ctx, nk_rgb_cf(tmpColor), nk_vec2(nk_widget_width(ctx), 400))) {
				nk_layout_row_dynamic(ctx, 120, 1);
				tmpColor = nk_color_picker(ctx, tmpColor, NK_RGBA);
				nk_layout_row_dynamic(ctx, 25, 1);
				tmpColor.r = nk_propertyf(ctx, "#R:", 0, tmpColor.r, 1.0f, 0.01f, 0.005f);
				tmpColor.g = nk_propertyf(ctx, "#G:", 0, tmpColor.g, 1.0f, 0.01f, 0.005f);
				tmpColor.b = nk_propertyf(ctx, "#B:", 0, tmpColor.b, 1.0f, 0.01f, 0.005f);
				tmpColor.a = nk_propertyf(ctx, "#A:", 0, tmpColor.a, 1.0f, 0.01f, 0.005f);
				vars.fogColor = glm::vec4(tmpColor.r, tmpColor.g, tmpColor.b, tmpColor.a);
				nk_combo_end(ctx);
			}

		}


	}
	nk_end(ctx);



	// if NK_WINDOW_MOVABLE or NK_WINDOW_SCALABLE -> does not change rectange when window size (screen size) changes
	if (nk_begin(ctx, "Diagram", nk_rect(vars.screenWidth - 200, 32, 200, vars.screenHeight - 32),
				 NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR /*| NK_WINDOW_MOVABLE*/ /*| NK_WINDOW_SCALABLE*/ /*|
				 NK_WINDOW_MINIMIZABLE*/ /*| NK_WINDOW_TITLE*/)) {

		nk_layout_row_static(ctx, 15, 200, 1);
		if (nk_button_label(ctx, "Recalculate Params")) {
			//lbm->resetSimulation();
			stlpDiagram.recalculateParameters();
		}

		

		if (nk_tree_push(ctx, NK_TREE_TAB, "Diagram controls", NK_MINIMIZED)) {
			nk_layout_row_static(ctx, 15, 200, 1);

			nk_checkbox_label(ctx, "Show isobars", &stlpDiagram.showIsobars);
			nk_checkbox_label(ctx, "Show isotherms", &stlpDiagram.showIsotherms);
			nk_checkbox_label(ctx, "Show isohumes", &stlpDiagram.showIsohumes);
			nk_checkbox_label(ctx, "Show dry adiabats", &stlpDiagram.showDryAdiabats);
			nk_checkbox_label(ctx, "Show moist adiabats", &stlpDiagram.showMoistAdiabats);
			nk_checkbox_label(ctx, "Show dewpoint curve", &stlpDiagram.showDewpointCurve);
			nk_checkbox_label(ctx, "Show ambient temp. curve", &stlpDiagram.showAmbientTemperatureCurve);
			nk_checkbox_label(ctx, "Crop Bounds", &stlpDiagram.cropBounds);

			nk_tree_pop(ctx);

		}

		nk_layout_row_static(ctx, 15, 200, 1);

		int tmp = stlpDiagram.overlayDiagramWidth;
		int maxDiagramWidth = (vars.screenWidth < vars.screenHeight) ? vars.screenWidth : vars.screenHeight;
		nk_slider_int(ctx, 10, (int *)&stlpDiagram.overlayDiagramWidth, maxDiagramWidth, 1);
		if (tmp != stlpDiagram.overlayDiagramWidth) {
			stlpDiagram.overlayDiagramHeight = stlpDiagram.overlayDiagramWidth;
			stlpDiagram.refreshOverlayDiagram(vars.screenWidth, vars.screenHeight);
		}

		if (nk_button_label(ctx, "Reset to default")) {
			stlpDiagram.resetToDefault();
		}

		if (nk_button_label(ctx, "Reset simulation")) {
			stlpSim->resetSimulation();
		}

		nk_slider_float(ctx, 0.01f, &stlpSim->simulationSpeedMultiplier, 1.0f, 0.01f);

		float delta_t_prev = stlpSim->delta_t;
		nk_property_float(ctx, "delta t", 0.0001f, &stlpSim->delta_t, 100.0f, 0.0001f, 1.0f);
		if (stlpSim->delta_t != delta_t_prev) {
			stlpSimCUDA->delta_t = stlpSim->delta_t;
			stlpSimCUDA->updateGPU_delta_t();
		}

		nk_property_int(ctx, "number of profiles", 2, &stlpDiagram.numProfiles, 100, 1, 1.0f); // somewhere bug when only one profile -> FIX!

		nk_property_float(ctx, "profile range", -10.0f, &stlpDiagram.convectiveTempRange, 10.0f, 0.01f, 0.01f);

		nk_property_int(ctx, "max particles", 1, &stlpSim->maxNumParticles, 100000, 1, 10.0f);

		nk_checkbox_label(ctx, "Simulate wind", &stlpSim->simulateWind);

		nk_checkbox_label(ctx, "use prev velocity", &stlpSim->usePrevVelocity);

		nk_checkbox_label(ctx, "Divide Previous Velocity", &vars.dividePrevVelocity);
		if (vars.dividePrevVelocity) {
			nk_property_float(ctx, "Divisor (x100)", 100.0f, &vars.prevVelocityDivisor, 1000.0f, 0.1f, 0.1f); // [1.0, 10.0]
		}

		nk_checkbox_label(ctx, "Show CCL Level", &vars.showCCLLevelLayer);
		nk_checkbox_label(ctx, "Show EL Level", &vars.showELLevelLayer);


		nk_checkbox_label(ctx, "Use CUDA", &vars.stlpUseCUDA);

		nk_checkbox_label(ctx, "Apply LBM", &vars.applyLBM);

		nk_checkbox_label(ctx, "Apply STLP", &vars.applySTLP);

		nk_property_float(ctx, "Point size", 0.1f, &stlpSim->pointSize, 100.0f, 0.1f, 0.1f);
		stlpSimCUDA->pointSize = stlpSim->pointSize;
		particleSystem->pointSize = stlpSim->pointSize;
		//nk_property_float(ctx, "Point size (CUDA)", 0.1f, &stlpSimCUDA->pointSize, 100.0f, 0.1f, 0.1f);

		nk_property_float(ctx, "Opacity multiplier", 0.01f, &vars.opacityMultiplier, 10.0f, 0.01f, 0.01f);

		struct nk_colorf tintColor;
		tintColor.r = vars.tintColor.x;
		tintColor.g = vars.tintColor.y;
		tintColor.b = vars.tintColor.z;

		if (nk_combo_begin_color(ctx, nk_rgb_cf(tintColor), nk_vec2(nk_widget_width(ctx), 400))) {
			nk_layout_row_dynamic(ctx, 120, 1);
			tintColor = nk_color_picker(ctx, tintColor, NK_RGBA);
			nk_layout_row_dynamic(ctx, 10, 1);
			tintColor.r = nk_propertyf(ctx, "#R:", 0, tintColor.r, 1.0f, 0.01f, 0.005f);
			tintColor.g = nk_propertyf(ctx, "#G:", 0, tintColor.g, 1.0f, 0.01f, 0.005f);
			tintColor.b = nk_propertyf(ctx, "#B:", 0, tintColor.b, 1.0f, 0.01f, 0.005f);
			tintColor.a = nk_propertyf(ctx, "#A:", 0, tintColor.a, 1.0f, 0.01f, 0.005f);
			vars.tintColor = glm::vec3(tintColor.r, tintColor.g, tintColor.b);
			nk_combo_end(ctx);
		}

		nk_property_int(ctx, "Opacity Blend Mode", 0, &particleSystem->opacityBlendMode, 1, 1, 1);
		nk_property_float(ctx, "Opacity Blend Range", 0.0f, &particleSystem->opacityBlendRange, 20.0f, 0.1f, 0.1f);
		nk_checkbox_label(ctx, "Show Hidden Particles", &particleSystem->showHiddenParticles);

		for (int i = 0; i < particleSystem->emitters.size(); i++) {
			if (nk_tree_push(ctx, NK_TREE_TAB, ("#Emitter " + to_string(i)).c_str(), NK_MAXIMIZED)) {
				Emitter *e = particleSystem->emitters[i];

				nk_layout_row_static(ctx, 15, 200, 1);
				nk_checkbox_label(ctx, "#enabled", &e->enabled);
				nk_checkbox_label(ctx, "#visible", &e->visible);
				nk_checkbox_label(ctx, "#wiggle", &e->wiggle);
				nk_property_float(ctx, "#x wiggle", 0.1f, &e->xWiggleRange, 10.0f, 0.1f, 0.1f);
				nk_property_float(ctx, "#z wiggle", 0.1f, &e->zWiggleRange, 10.0f, 0.1f, 0.1f);


				nk_property_float(ctx, "#x", -1000.0f, &e->position.x, 1000.0f, 1.0f, 1.0f);
				//nk_property_float(ctx, "#y", -1000.0f, &e->position.y, 1000.0f, 1.0f, 1.0f);
				nk_property_float(ctx, "#z", -1000.0f, &e->position.z, 1000.0f, 1.0f, 1.0f);

				//nk_property_variant_int()
				nk_property_int(ctx, "#emit per step", 0, &e->numParticlesToEmitPerStep, 10000, 10, 10);

				CircleEmitter *ce = dynamic_cast<CircleEmitter *>(e);
				if (ce) {
					nk_property_float(ctx, "#radius", 1.0f, &ce->radius, 1000.0f, 1.0f, 1.0f);
				}


				nk_tree_pop(ctx);
				//particleSystem->emitters[i]
			}
		}
		nk_layout_row_static(ctx, 15, 200, 1);
		if (nk_button_label(ctx, "Activate All Particles")) {
			particleSystem->activateAllParticles();
		}
		if (nk_button_label(ctx, "Deactivate All Particles")) {
			particleSystem->deactivateAllParticles();
		}
		if (nk_button_label(ctx, "Enable All Emitters")) {
			particleSystem->enableAllEmitters();
		}
		if (nk_button_label(ctx, "Disable All Emitters")) {
			particleSystem->disableAllEmitters();
		}

		nk_property_int(ctx, "Active Particles", 0, &particleSystem->numActiveParticles, particleSystem->numParticles, 1000, 100);



		tintColor.r = vars.bgClearColor.x;
		tintColor.g = vars.bgClearColor.y;
		tintColor.b = vars.bgClearColor.z;

		if (nk_combo_begin_color(ctx, nk_rgb_cf(tintColor), nk_vec2(nk_widget_width(ctx), 400))) {
			nk_layout_row_dynamic(ctx, 120, 1);
			tintColor = nk_color_picker(ctx, tintColor, NK_RGBA);
			nk_layout_row_dynamic(ctx, 10, 1);
			tintColor.r = nk_propertyf(ctx, "#R:", 0, tintColor.r, 1.0f, 0.01f, 0.005f);
			tintColor.g = nk_propertyf(ctx, "#G:", 0, tintColor.g, 1.0f, 0.01f, 0.005f);
			tintColor.b = nk_propertyf(ctx, "#B:", 0, tintColor.b, 1.0f, 0.01f, 0.005f);
			//tintColor.a = nk_propertyf(ctx, "#A:", 0, tintColor.a, 1.0f, 0.01f, 0.005f);
			vars.bgClearColor = glm::vec3(tintColor.r, tintColor.g, tintColor.b);
			nk_combo_end(ctx);
		}


	}
	nk_end(ctx);





	ctx->style.window.padding = nk_vec2(0, 0);

	static int toolbarHeight = 32;
	if (nk_begin(ctx, "test", nk_rect(0, 0, vars.screenWidth, toolbarHeight), NK_WINDOW_NO_SCROLLBAR)) {

		int numToolbarItems = 3;

		//nk_layout_row_static(ctx, 32, screenWidth, numToolbarItems);
		//nk_menu

		/* menubar */
		enum menu_states { MENU_DEFAULT, MENU_WINDOWS };
		static nk_size mprog = 60;
		static int mslider = 10;
		static int mcheck = nk_true;
		nk_menubar_begin(ctx);

		/* menu #1 */
		nk_layout_row_begin(ctx, NK_STATIC, toolbarHeight, 5);
		nk_layout_row_push(ctx, 120);
		if (nk_menu_begin_label(ctx, "File", NK_TEXT_CENTERED, nk_vec2(120, 200))) {
			static size_t prog = 40;
			static int slider = 10;
			static int check = nk_true;
			nk_layout_row_dynamic(ctx, 25, 1);
			nk_menu_item_label(ctx, "Hide", NK_TEXT_LEFT);
			nk_menu_item_label(ctx, "About", NK_TEXT_LEFT);
			nk_progress(ctx, &prog, 100, NK_MODIFIABLE);
			nk_slider_int(ctx, 0, &slider, 16, 1);
			nk_checkbox_label(ctx, "check", &check);
			nk_menu_end(ctx);
		}
		nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);
		if (nk_menu_begin_label(ctx, "View", NK_TEXT_CENTERED, nk_vec2(120, 200))) {
			nk_layout_row_dynamic(ctx, 25, 1);
			//nk_button_label(ctx, "Debug Window");
			if (nk_menu_item_label(ctx, "Debug Window", NK_TEXT_CENTERED)) {
				cout << "opening debug window" << endl;
			}
			nk_label(ctx, "Camera Settings", NK_TEXT_CENTERED);
			if (nk_menu_item_label(ctx, "Front View (I)", NK_TEXT_CENTERED)) {
				camera->setView(Camera::VIEW_FRONT);
			}
			if (nk_menu_item_label(ctx, "Side View (O)", NK_TEXT_CENTERED)) {
				camera->setView(Camera::VIEW_SIDE);
			}
			if (nk_menu_item_label(ctx, "Top View (P)", NK_TEXT_CENTERED)) {
				camera->setView(Camera::VIEW_TOP);
			}
			if (drawSkybox) {
				if (nk_menu_item_label(ctx, "Hide Skybox", NK_TEXT_CENTERED)) {
					drawSkybox = false;
				}
			} else {
				if (nk_menu_item_label(ctx, "Show Skybox", NK_TEXT_CENTERED)) {
					drawSkybox = true;
				}
			}
			nk_menu_end(ctx);

		}
		nk_layout_row_push(ctx, 120);
		if (nk_menu_begin_label(ctx, "About", NK_TEXT_CENTERED, nk_vec2(120, 200))) {
			nk_layout_row_dynamic(ctx, 25, 1);
			if (nk_menu_item_label(ctx, "Show About", NK_TEXT_CENTERED)) {
				vars.aboutWindowOpened = true;
			}

			nk_menu_end(ctx);
		}

		//nk_label(ctx, "About", NK_TEXT_CENTERED);

		//nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);

		//nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);
	}
	nk_end(ctx);


	/*
	
			static int show_group =  1;
		if (show_group) {
			nk_layout_row_dynamic(ctx, 100, 1);
			int res = nk_group_begin(ctx, "Node", NK_WINDOW_CLOSABLE|NK_WINDOW_BORDER);
			show_group = res != NK_WINDOW_CLOSED;
			if (res && show_group) {
				 ...
			nk_group_end(ctx);
			}
		}
	*/

	if (vars.aboutWindowOpened) {


		if (nk_begin(ctx, "About Window", nk_rect(vars.screenWidth / 2 - 250, vars.screenHeight / 2 - 250, 500, 500), NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_CLOSABLE)) {
			nk_layout_row_dynamic(ctx, 20.0f, 1);

			nk_label(ctx, "Orographic Cloud Simulator", NK_TEXT_CENTERED);
			nk_label(ctx, "Author: Martin Cap", NK_TEXT_CENTERED);
			nk_label(ctx, "Email: martincap94@gmail.com", NK_TEXT_CENTERED);


		} else {
			vars.aboutWindowOpened = false;
		}
		nk_end(ctx);


	}

	//nk_end(ctx);



}


void window_size_callback(GLFWwindow* window, int width, int height) {
	float aspectRatio = (float)width / (float)height;

	vars.screenWidth = width;
	vars.screenHeight = height;

	float offset = 0.2f;
	diagramProjection = glm::ortho(-aspectRatio / 2.0f + 0.5f - aspectRatio * offset, aspectRatio / 2.0f + 0.5f + aspectRatio * offset, 1.0f + offset, 0.0f - offset, nearPlane, farPlane);

	cout << "Aspect ratio = " << aspectRatio << endl;


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
	stlpDiagram.refreshOverlayDiagram(vars.screenWidth, vars.screenHeight);
}