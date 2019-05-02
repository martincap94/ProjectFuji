///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       STLPDiagram.h
* \author     Martin Cap
* \date       2019/01/18
*
*	The SkewT/LogP diagram (STLPDiagram class) is the heart of SkewT/LogP cloud simulator by Duarte.
*	Available here: https://www.researchgate.net/publication/318444032_Real-Time_Simulation_of_Cumulus_Clouds_through_SkewTLogP_Diagrams
*	It generates the SkewT/LogP diagram on the CPU. The data upload to GPU is processed by the
*	STLPSimulatorCUDA that accesses this diagram's data.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <vector>
#include <glad\glad.h>

#include "DataStructures.h"
#include "ShaderProgram.h"
#include "Curve.h"
#include "TextRenderer.h"
#include "VariableManager.h"

#include "UserInterface.h"
#include <nuklear.h>


using namespace std;

class STLPDiagram {
public:

	struct WindDataItem {
		//float z; // altitude in this case
		//float delta_x;
		//float delta_y;
	
		// OpenGL coordinate system
		float y;
		float delta_x;
		float delta_z;
	};
	vector<WindDataItem> windData;

	TextRenderer *textRend;			///< Text renderer engine

	vector<SoundingDataItem> soundingData;	///< Sounding data loaded from file


	glm::vec2 CCLNormalized = glm::vec2(0.0f);		///< Normalized convective condensation level
	glm::vec2 TcNormalized = glm::vec2(0.0f);			///< Normalized convective temperature
	glm::vec2 ELNormalized = glm::vec2(0.0f);			///< Normalized equilibrium level
	glm::vec2 LCLNormalized = glm::vec2(0.0f);
	glm::vec2 LFCNormalized = glm::vec2(0.0f);
	glm::vec2 orographicELNormalized = glm::vec2(0.0f);

	glm::vec2 CCL = glm::vec2(0.0f);					///< Convective condensation level
	glm::vec2 Tc = glm::vec2(0.0f);					///< Convective temperature
	glm::vec2 EL = glm::vec2(0.0f);					///< Equilibrium level
	glm::vec2 LCL = glm::vec2(0.0f);
	glm::vec2 LFC = glm::vec2(0.0f);
	glm::vec2 orographicEL = glm::vec2(0.0f);

	bool dewpointFound = false;
	bool CCLFound = false;
	bool TcFound = false;
	bool ELFound = false;
	bool LCLFound = false;
	bool LFCFound = false;
	bool orographicELFound = false;



	int soundingCurveEditingEnabled = 0;

	int useOrographicParameters = 0;

	// helper curves
	Curve xaxis;					///< x axis curve (single line)
	Curve yaxis;					///< y axis curve (single line)

	Curve groundIsobar;

	Curve ambientCurve;				///< Sounding curve with ambient temperatures
	Curve dewpointCurve;			///< Sounding curve with dewpoint temperatures
	Curve TcDryAdiabat;				///< Dry adiabat that has convective temperature at ground level
	Curve moistAdiabat_CCL_EL;
	Curve moistAdiabat_LCL_EL;

	Curve mixingCCL;
	Curve LCLDryAdiabatCurve;

	// use int values for user interface (easier to use than temporary int values)
	int showIsobars = 1;
	int showIsotherms = 1;
	int showIsohumes = 1;
	int showDryAdiabats[2] = { 1, 1 };
	int showMoistAdiabats[2] = { 1, 1 };
	int showDewpointCurve = 1;
	int showAmbientCurve = 1;

	glm::vec3 isobarsColor = glm::vec3(0.8f, 0.8f, 0.8f);
	glm::vec3 isothermsColor = glm::vec3(0.8f, 0.8f, 0.8f);
	glm::vec3 ambientCurveColor = glm::vec3(0.7f, 0.1f, 0.15f);
	glm::vec3 dewpointCurveColor = glm::vec3(0.1f, 0.7f, 0.15f);
	glm::vec3 isohumesColor = glm::vec3(0.1f, 0.15f, 0.7f);
	glm::vec3 dryAdiabatsColor[2] = { glm::vec3(0.6f, 0.6f, 0.6f), glm::vec3(1.0f, 0.6f, 0.6f) };
	glm::vec3 moistAdiabatsColor[2] = { glm::vec3(0.2f, 0.6f, 0.8f), glm::vec3(0.2f, 0.8f, 0.9f) };


	float minP;						//!< Minimum pressure in normalized (diagram) coordinates 
	float maxP;						///< Maximum pressure in normalized (diagram) coordinates
	float minT;						///< Minimum temperature in normalized (diagram) coordinates
	float maxT;						///< Maximum temperature in normalized (diagram) coordinates
	int maxVerticesPerCurve;


	string soundingFilename;			///< Filename of the sounding file that will be loaded and displayed

	pair<Curve *, int> selectedPoint;

	int numProfiles = 100;
	float convectiveTempRange = 2.0f;
	float profileDelta;

	vector<glm::vec2> TcProfiles;
	vector<glm::vec2> CCLProfiles;
	vector<glm::vec2> ELProfiles;
	vector<Curve> dryAdiabatProfiles;
	vector<Curve> moistAdiabatProfiles;



	GLuint diagramTexture;
	GLuint diagramFramebuffer;
	GLuint diagramMultisampledFramebuffer;
	GLuint diagramMultisampledTexture;
	GLint textureResolution = 1024;
	GLuint overlayDiagramVAO;
	GLuint overlayDiagramVBO;

	float overlayDiagramX = 270.0f;
	float overlayDiagramY = 250.0f;
	float overlayDiagramResolution = 500.0f;
	//GLuint overlayDiagramHeight = 256;

	GLuint particlesVAO;
	GLuint particlesVBO;


	float dryAdiabatDeltaT = 10.0f;
	float moistAdiabatDeltaT = 10.0f;

	float P0;
	float groundAltitude;

	int cropBounds = 1;

	vector<glm::vec2> particlePoints; // deprecated!



	STLPDiagram(VariableManager *vars);

	/// Deallocates data - destroys the text renderer engine in particular.
	~STLPDiagram();

	/// Initializes the diagram from given sounding data.
	void init();

	/// Loads the sounding data to the soundingData vector.
	void loadSoundingData();


	/**
		Generates isobars and uploads their data to the VBO for drawing.
		This function does not save any data on CPU, only OpenGL VBO is updated.
		The VBO is assumed to be ready for data upload.
	*/
	void generateIsobars();

	/**
		Generates isotherms and uploads their data to the VBO for drawing.
		This function does not save any data on CPU, only OpenGL VBO is updated.
		The VBO is assumed to be ready for data upload.
	*/
	void generateIsotherms();

	/**
		Initializes the dewpoint sounding curve and uploads it to the VBO for rendering.
	*/
	void initDewpointCurve();

	/**
	Initializes the ambient temperature sounding curve and uploads it to the VBO for rendering.
	*/
	void initAmbientTemperatureCurve();


	void generateMixingRatioLine();
	void generateMixingRatioLineExperimental();



	/*
		Dry adiabats feature the thermodynamic behaviour of unsaturated air parcels moving upwards (or downwards).
		They represent the dry adiabatic lapse rate (DALR).
		This thermodynamic behaviour is valid for all air parcels moving between the ground and the convective
		condensation level (CCL).

		T(P) = theta / ((P0 / P)^(Rd / cp))
		where
		P0 is the initial value of pressure (profileIndex.e. ground pressure)
		cp is the heat capacity of dry air at constant pressure
		(cv is the heat capacity of dry air at constant volume)
		Rd is the gas constant for dry air [J kg^-1 K^-1]
		k = Rd / cp = (cp - cv) / cp =(approx)= 0.286
	*/
	void generateDryAdiabat(float theta, vector<glm::vec2> &vertices, int mode, float P0 = 1000.0f, vector<int> *edgeCounter = nullptr, bool incrementCounter = true, float deltaP = 25.0f, Curve *curve = nullptr);


	void generateMoistAdiabat(float theta, float startP, vector<glm::vec2> &vertices, int mode, float P0 = 1000.0f, vector<int> *edgeCounter = nullptr, bool incrementCounter = true, float deltaP = 25.0f, Curve *curve = nullptr, float smallDeltaP = 1.0f);


	void recalculateAll();


	void initBuffers();
	void initCurves();
	void initOverlayDiagram();

	void recalculateParameters();
	void recalculateProfileDelta();

	/// Initializes all buffers and therefore curves of the diagram.
	/**
		Initializes all buffers and therefore curves of the diagram.
		Prototype function that does all the work - should be separated into multiple
		methods and utilize Curve class that will be extended.
	*/
	void initBuffersOld();

	glm::vec2 getWindDeltasFromAltitude(float altitude);
	void getWindDeltasForLattice(int latticeHeight, std::vector<glm::vec3> &outWindDeltas);

	///////////////////////////////////////////////////////////////////////////
	// Normalized coordinates helper functions
	///////////////////////////////////////////////////////////////////////////
	glm::vec2 getNormalizedCoords(glm::vec2 coords);
	glm::vec2 getDenormalizedCoords(glm::vec2 coords);

	glm::vec2 getNormalizedCoords(float T, float P);
	glm::vec2 getDenormalizedCoords(float x, float y);

	float getNormalizedTemp(float T, float y);
	float getNormalizedPres(float P);

	float getDenormalizedTemp(float x, float y);
	float getDenormalizedPres(float y);
	///////////////////////////////////////////////////////////////////////////

	/// Initializes the FreeType TextRenderer.
	void initFreetype();

	/// Draws the diagram (without text).
	void draw();

	/// Draws the text label for the diagram.
	void drawText();


	void drawOverlayDiagram(GLuint textureId = -1);
	void refreshOverlayDiagram(GLuint viewportWidth, GLuint viewportHeight, GLuint viewport_x = 0, GLuint viewport_y = 0);


	/// Only sets existing point!
	void setVisualizationPoint(glm::vec3 position, glm::vec3 color, int index, bool positionIsNormalized);

	void findClosestSoundingPoint(glm::vec2 queryPoint);
	void moveSelectedPoint(glm::vec2 mouseCoords);


	void constructDiagramCurvesToolbar(struct nk_context *ctx, UserInterface *ui);
	//void constructParametersFoundTab(struct nk_context *ctx, UserInterface *ui);




private:

	const float temperatureNotchSize = 0.01f;

	const float maxTextScale = 0.0005f;


	VariableManager *vars = nullptr;
	
	ShaderProgram *curveShader = nullptr;
	ShaderProgram *singleColorShaderVBO = nullptr;
	ShaderProgram *overlayDiagramShader = nullptr;

	vector<glm::vec3> visualizationPoints; // helper points, buffered in such a way that it is point, color, point, color, etc. (stride = 3 * sizeof(float))
	GLuint visPointsVAO;
	GLuint visPointsVBO;

	vector<glm::vec3> mainParameterPoints;
	GLuint mainParameterPointsVAO;
	GLuint mainParameterPointsVBO;


	vector<glm::vec2> temperaturePoints;

	GLuint isobarsVAO;
	GLuint isobarsVBO;
	int numIsobars;

	GLuint temperaturePointsVAO;
	GLuint temperaturePointsVBO;
	int temperaturePointsCount;

	GLuint isothermsVAO;
	GLuint isothermsVBO;
	int isothermsCount;

	GLuint ambientTemperatureVAO;
	GLuint ambientTemperatureVBO;

	GLuint dewTemperatureVAO;
	GLuint dewTemperatureVBO;


	GLuint isohumesVAO;
	GLuint isohumesVBO;


	GLuint dryAdiabatsVAO[2];
	GLuint dryAdiabatsVBO[2];
	int numDryAdiabats[2];
	vector<int> dryAdiabatEdgeCount[2];

	GLuint moistAdiabatsVAO[2];
	GLuint moistAdiabatsVBO[2];
	int numMoistAdiabats[2];
	vector<int> moistAdiabatEdgeCount[2];



	// deprecated
	GLuint CCLVAO;
	GLuint CCLVBO;

	// deprecated
	GLuint TcVAO;
	GLuint TcVBO;


	const float xmin = 0.0f;
	const float xmax = 1.0f;
	float ymin;
	float ymax;

	//bool initialized = false; //!< Whether the diagram has already been initialized



	void generateTemperatureNotches();

	void generateDryAdiabats();
	void generateMoistAdiabats();


	void uploadMainParameterPointsToBuffer();


};

