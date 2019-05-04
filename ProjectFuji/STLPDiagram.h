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

//! STLPDiagram is an important meteorological diagram that describes air parcel motion in the atmosphere.
/*!
	SkewT/LogP diagrams are created from real sounding data obtained by radiosondes.
	On the x axis we plot a 45 degree skewed temperature (hence SkewT) and on the y axis the logarithm of base 10 
	of pressure is plotted (hence LogP). Note that pressure decreases the higher altitude in the atmosphere.
	It describes properties such as ambient temperature, dewpoint temperature, wind speeds, mixing ratio, and others.
	It is the main tool of Duarte's thesis where he describes its usage in cloud formation simulation.
	Our implementation of the diagram runs on the CPU and is then uploaded to GPU by STLPSimulatorCUDA object.

*/
class STLPDiagram {
public:

	//! Describes a wind data item from the sounding data.
	struct WindDataItem {
		// OpenGL coordinate system
		float y;		//!< Altitude for which the wind deltas are measured
		float delta_x;	//!< Speed of wind on the x axis
		float delta_z;	//!< Speed of wind on the z axis
	};
	vector<WindDataItem> windData;			//!< List of all wind data items for the loaded sounding file

	TextRenderer *textRend;					//!< Text renderer engine

	vector<SoundingDataItem> soundingData;	//!< Sounding data loaded from file

	// Note that normalized means within range [0,1] and in the coordinate system of the diagram visualization

	glm::vec2 CCLNormalized = glm::vec2(0.0f);				//!< Normalized convective condensation level
	glm::vec2 TcNormalized = glm::vec2(0.0f);				//!< Normalized convective temperature
	glm::vec2 ELNormalized = glm::vec2(0.0f);				//!< Normalized equilibrium level
	glm::vec2 LCLNormalized = glm::vec2(0.0f);				//!< Normalized lifting condensation level 
	glm::vec2 LFCNormalized = glm::vec2(0.0f);				//!< Normalized level of free convection
	glm::vec2 orographicELNormalized = glm::vec2(0.0f);		//!< Normalized equilibrium level for the orographic parameters

	glm::vec2 CCL = glm::vec2(0.0f);				//!< Convective condensation level in world coordinates
	glm::vec2 Tc = glm::vec2(0.0f);					//!< Convective temperature in world coordinates
	glm::vec2 EL = glm::vec2(0.0f);					//!< Equilibrium level in world coordinates
	glm::vec2 LCL = glm::vec2(0.0f);				//!< Lifting condensation level in world coordinates
	glm::vec2 LFC = glm::vec2(0.0f);				//!< Level of free convection in world coordinates
	glm::vec2 orographicEL = glm::vec2(0.0f);		//!< Equilibrium level for orographic parameters in world coordinates

	bool CCLFound = false;				//!< Whether the CCL was found (intersection exists)
	bool TcFound = false;				//!< Whether the Tc was found (intersection exists)
	bool ELFound = false;				//!< Whether the EL was found (intersection exists)
	bool LCLFound = false;				//!< Whether the LCL was found (intersection exists)
	bool LFCFound = false;				//!< Whether the LFC was found (intersection exists)
	bool orographicELFound = false;		//!< Whether the orographic EL was found (intersection exists)


	int soundingCurveEditingEnabled = 0;	//!< Whether we are currently editing sounding curves
	int useOrographicParameters = 0;		//!< Whether to use orographic parameter set to create curve profiles

	Curve xaxis;					//!< x axis curve (single line)
	Curve yaxis;					//!< y axis curve (single line)

	Curve groundIsobar;				//!< Isobar plotted at the ground level from the sounding data

	Curve ambientCurve;				//!< Sounding curve with ambient temperatures
	Curve dewpointCurve;			//!< Sounding curve with dewpoint temperatures
	Curve TcDryAdiabat;				//!< Dry adiabat that has convective temperature at ground level
	Curve moistAdiabat_CCL_EL;		//!< Moist adiabat running through CCL
	Curve moistAdiabat_LCL_EL;		//!< Moist adiabat running through LCL

	Curve mixingCCL;				//!< Mixing ratio line starting in the dewpoint on ground
	Curve LCLDryAdiabatCurve;		//!< Dry adiabat curve going through LCL

	// use int values for user interface (easier to use than temporary int values)
	int showIsobars = 1;					//!< Whether isobars are visible in the diagram
	int showIsotherms = 1;					//!< Whether isotherms are visible in the diagram
	int showIsohumes = 1;					//!< Whether isohumes (mixing ration lines) are visible in the diagram
	int showDryAdiabats[2] = { 1, 1 };		//!< Whether dry adiabats (general, profiles) are visible in the diagram
	int showMoistAdiabats[2] = { 1, 1 };	//!< Whether moist adiabats (general, profiles) are visible in the diagram
	int showDewpointCurve = 1;				//!< Whether dewpoint curve is visible in the diagram
	int showAmbientCurve = 1;				//!< Whether ambient temperature curve is visible in the diagram

	glm::vec3 isobarsColor = glm::vec3(0.8f, 0.8f, 0.8f);
	glm::vec3 isothermsColor = glm::vec3(0.8f, 0.8f, 0.8f);
	glm::vec3 ambientCurveColor = glm::vec3(0.7f, 0.1f, 0.15f);
	glm::vec3 dewpointCurveColor = glm::vec3(0.1f, 0.7f, 0.15f);
	glm::vec3 isohumesColor = glm::vec3(0.1f, 0.15f, 0.7f);
	glm::vec3 dryAdiabatsColor[2] = { glm::vec3(0.6f, 0.6f, 0.6f), glm::vec3(1.0f, 0.6f, 0.6f) };
	glm::vec3 moistAdiabatsColor[2] = { glm::vec3(0.2f, 0.6f, 0.8f), glm::vec3(0.2f, 0.8f, 0.9f) };


	float minP;						//!< Minimum pressure in normalized (diagram) coordinates 
	float maxP;						//!< Maximum pressure in normalized (diagram) coordinates
	float minT;						//!< Minimum temperature in normalized (diagram) coordinates
	float maxT;						//!< Maximum temperature in normalized (diagram) coordinates
	int maxVerticesPerCurve;		//!< Maximum number of vertices per curve

	string soundingFilename;			//!< Filename of the sounding file that will be loaded and displayed at startup

	pair<Curve *, int> selectedPoint;	//!< Selected point during diagram editing

	int numProfiles = 100;				//!< Number of convective temperature profiles to be used
	float convectiveTempRange = 2.0f;	//!< Range of the profiles (may be negative)
	float profileDelta;					//!< Delta temperature between two profiles (spacing between curves)

	vector<glm::vec2> TcProfiles;		//!< List of all Tc temperature/pressure profiles
	vector<glm::vec2> CCLProfiles;		//!< List of all CCL temperature/pressure profiles
	vector<glm::vec2> ELProfiles;		//!< List of all EL temperature/pressure profiles
	vector<Curve> dryAdiabatProfiles;	//!< List of all dry adiabat curve profiles
	vector<Curve> moistAdiabatProfiles;	//!< List of all moist adiabat curve profiles



	GLuint diagramFramebuffer;				//!< Framebuffer for the overlay diagram
	GLuint diagramTexture;					//!< Texture color attachment for the overlay diagram
	GLuint diagramMultisampledFramebuffer;	//!< Multisampled framebuffer for the overlay diagram
	GLuint diagramMultisampledTexture;		//!< Multisampled texture color attachment for the overlay diagram
	GLint textureResolution = 1024;			//!< Resolution of the overlay diagram
	GLuint overlayDiagramVAO;				//!< VAO for the overlay diagram
	GLuint overlayDiagramVBO;				//!< VBO for the overlay diagram

	float overlayDiagramX = 270.0f;				//!< Default overlay diagram screen x offset
	float overlayDiagramY = 250.0f;				//!< Default overlay diagram screen y offset
	float overlayDiagramResolution = 500.0f;	//!< Default overlay diagram screen resolution/size

	std::vector<glm::vec2> particlePoints;		//!< Old CPU particle points.


	float dryAdiabatDeltaT = 10.0f;			//!< Default delta T for general dry adiabats
	float moistAdiabatDeltaT = 10.0f;		//!< Default delta T for general moist adiabats

	float P0;								//!< Ground pressure for the current sounding file
	float groundAltitude;					//!< Ground altitude for the current sounding file

	int cropBounds = 1;						//!< Crop out of bounds curves and other diagram elements (discard in fragment shader)

	//! Initializes the STLPDiagram.
	/*!
		Calls the init() function which initializes the whole diagram and all members that are needed.
		\see init()
		\param[in] vars		VariableManager to be used by the diagram.
	*/
	STLPDiagram(VariableManager *vars);

	//! Deallocates data - destroys the text renderer engine in particular.
	~STLPDiagram();

	//! Initializes the diagram from given sounding data.
	/*!
		Loads the sounding file, initializes the shaders for drawing the diagram, overlay diagram.
		Initializes the FreeType library. Initializes buffers, curves and the overlay diagram.
	*/
	void init();

	//! Loads the sounding data to the soundingData vector.
	void loadSoundingData();


	//! Generates isobars and uploads the data to their VBO.
	/*!
		Generates isobars and uploads their data to the VBO for drawing.
		This function does not save any data on CPU, only OpenGL VBO is updated.
		The VBO is assumed to be ready for data upload.
	*/
	void generateIsobars();

	//! Generates isotherms and uploads the data to their VBO.
	/*!
		Generates isotherms and uploads their data to the VBO for drawing.
		This function does not save any data on CPU, only OpenGL VBO is updated.
		The VBO is assumed to be ready for data upload.
	*/
	void generateIsotherms();

	//! Initializes the dewpoint sounding curve and uploads it to the VBO for rendering.
	void initDewpointCurve();

	//! Initializes the ambient temperature sounding curve and uploads it to the VBO for rendering.
	void initAmbientTemperatureCurve();

	//! Generates the mixing ratio line that starts in the dewpoint on ground level.
	void generateMixingRatioLine();

	//! --- for testing purposes only --- Generates the mixing ratio line.
	void generateMixingRatioLineExperimental();



	//! Generate a dry adiabat for a constant potential temperature.
	/*!
		Dry adiabats feature the thermodynamic behaviour of unsaturated air parcels moving upwards (or downwards).
		They represent the dry adiabatic lapse rate (DALR).
		This thermodynamic behaviour is valid for all air parcels moving between the ground and the convective
		condensation level (CCL).

		T(P) = theta / ((P0 / P)^(R_d / c_pd))
		where
		 -	P0 is the initial value of pressure (profileIndex.e. ground pressure)
		 -	c_pd is the heat capacity of dry air at constant pressure, c_pd = 1005.7
		 -	(cv is the heat capacity of dry air at constant volume)
		 -	R_d is the gas constant for dry air [J kg^-1 K^-1], R_d = 287.05307
		 - 	k = R_d / c_pd =(approx)= 0.285

		\param[in] theta				Potential temperature for which we want to plot the dry adiabat curve (in celsius [C]).
		\param[in,out] vertices			Reference to the vertex vector to be filled with the generated curve vertices.
		\param[in] mode					Mode of dry adiabats (general or profiles).
		\param[in] P0					Ground pressure to be used in computation in hectopascals [hPa].
		\param[in,out] edgeCounter		Edge counter into which we push this adiabat's edge count (for drawing purposes).
		\param[in] incrementCounter		Whether to increment the global counter for dry adiabats.
		\param[in] deltaP				Distance on the y axis between vertices of the generated curve (in hectopascals [hPa]).
		\param[in,out] curve			The curve to be set as the generated adiabat.
	*/
	void generateDryAdiabat(float theta, vector<glm::vec2> &vertices, int mode, float P0 = 1000.0f, vector<int> *edgeCounter = nullptr, bool incrementCounter = true, float deltaP = 25.0f, Curve *curve = nullptr);

	//! Generate a moist adiabat that starts in the given point.
	/*!
		Moist adiabats are generated iteratively using the saturated (moist) adiabatic lapse rate.
		\see dTdP_moist_degK_Bakhshaii()

		\param[in] startT				Starting absolute temperature in celsius [C].
		\param[in] startP				Starting pressure in hectopascals [hPa].
		\param[in,out] vertices			Reference to the vertex vector to be filled with the generated curve vertices.
		\param[in] mode					Mode of moist adiabats (general or profiles).
		\param[in] P0					Ground pressure to be used in computation in hectopascals [hPa].
		\param[in,out] edgeCounter		Edge counter into which we push this adiabat's edge count (for drawing purposes).
		\param[in] incrementCounter		Whether to increment the global counter for moist adiabats.
		\param[in] deltaP				Distance on the y axis between vertices of the generated curve (in hectopascals [hPa]).
		\param[in,out] curve			The curve to be set as the generated adiabat.
		\param[in] smallDeltaP			Change of pressure between iterations in hectopascals [hPa].
	*/
	void generateMoistAdiabat(float startT, float startP, vector<glm::vec2> &vertices, int mode, float P0 = 1000.0f, vector<int> *edgeCounter = nullptr, bool incrementCounter = true, float deltaP = 25.0f, Curve *curve = nullptr, float smallDeltaP = 1.0f);

	//! Recalculates all the data of the diagram.
	void recalculateAll();

	//! Initializes the necessary OpenGL buffers for diagram drawing.
	void initBuffers();

	//! Initializes all the curves and important parameter points for STLP simulation.
	void initCurves();

	//! Initializes the necessary buffers (framebuffers, textures, VBOs, VAOs) for overlay diagram drawing.
	void initOverlayDiagram();

	//! Recalculates the simulation parameters (such as CCL, EL, and so on). Useful when editing diagram curves.
	void recalculateParameters();

	//! Recalculate the distances on the x axis (temperatures) between individual profiles with equal spacing.
	void recalculateProfileDelta();

	//! Initializes all buffers and therefore curves of the diagram.
	/*!
		Initializes all buffers and therefore curves of the diagram.
		Prototype function that does all the work - should be separated into multiple
		methods and utilize Curve class that will be extended.
	*/
	//void initBuffersOld();

	//! Returns loaded wind deltas (velocities in x and z axes).
	/*!
		Uses linear interpolation to compute precise velocities.
		\param[in] altitude		Given altitude in meters [m].
		\return					Vector of the x and z wind velocities.
	*/
	glm::vec2 getWindDeltasFromAltitude(float altitude);

	//! Generates wind deltas list custom created for lattice height (for each lattice cell basically).
	/*!
		\param[in] latticeHeight	Height of the lattice.
		\param[out] outWindDeltas	Vector populated with the wind deltas for each cell.
	*/
	void getWindDeltasForLattice(int latticeHeight, std::vector<glm::vec3> &outWindDeltas);

	///////////////////////////////////////////////////////////////////////////
	// Normalized coordinates helper functions
	///////////////////////////////////////////////////////////////////////////
	//! Returns normalized diagram coordinates for the given (temperature [C], pressure [hPa]) pair.
	/*!
		\param[in] coords	Pair of temperature [C] and pressure [hPa].
		\return				Normalized diagram coordinates (x, y).
	*/
	glm::vec2 getNormalizedCoords(glm::vec2 coords);

	//! Returns denormalized pair (temperature [C], pressure [hPa]) for the given diagram coordinates.
	/*!
		\param[in] coords	Normalized diagram coordinates (x, y).
		\return				Pair of temperature [C] and pressure [hPa].
	*/
	glm::vec2 getDenormalizedCoords(glm::vec2 coords);

	//! Returns normalized diagram coordinates for the given temperature [C] and pressure [hPa].
	/*!
		\param[in] T		Temperature in celsius [C].
		\param[in] P		Pressure in hectopascals [hPa].
		\return				Normalized diagram coordinates (x, y).
	*/
	glm::vec2 getNormalizedCoords(float T, float P);

	//! Returns denormalized pair (temperature [C], pressure [hPa]) for the given diagram coordinates.
	/*!
		\param[in] x		Normalized diagram x coordinate.
		\param[in] y		Normalized diagram y coordinate.
		\return				Pair of temperature [C] and pressure [hPa].
	*/
	glm::vec2 getDenormalizedCoords(float x, float y);

	//! Returns normalized diagram x value for the given temperature [C] and y axis value.
	/*!
		\param[in] T		Temperature in celsius [C].
		\param[in] y		Normalized diagram y coordinate.
		\return				Normalized diagram x coordinate.
	*/
	float getNormalizedTemp(float T, float y);

	//! Returns normalized diagram y value for the given pressure [hPa].
	/*!
		\param[in] P		Pressure in hectopascals [hPa].
		\return				Normalized diagram y coordinate.
	*/
	float getNormalizedPres(float P);

	//! Return temperature for the given x and y normalized diagram coordinates.
	/*!
		\param[in] x		Normalized diagram x coordinate.
		\param[in] y		Normalized diagram y coordinate.
		\return				Temperature in celsius [C] for the given coordinates.
	*/
	float getDenormalizedTemp(float x, float y);

	//! Return pressure for the given y normalized diagram coordinate.
	/*!
		\param[in] y		Normalized diagram y coordinate.
		\return				Pressure in hectopascals [hPa] for the given y coordinate.
	*/
	float getDenormalizedPres(float y);
	///////////////////////////////////////////////////////////////////////////

	//! Initializes the FreeType TextRenderer.
	void initFreetype();

	//! Draws the diagram (without text).
	void draw();

	//! Draws the text labels for the diagram.
	void drawText();

	//! Draws the overlay diagram.
	/*!
		\param[in] textureId		Texture to be used if diagramTexture not set.
	*/
	void drawOverlayDiagram(GLuint textureId = -1);

	//! Refreshes position of the overlay diagram and updates its VBO.
	/*!
		\param[in] viewportWidth	Width of the current viewport.
		\param[in] viewportHeight	Height of the current viewport.
		\param[in] viewport_x		x position of the viewport (should be 0).
		\param[in] viewport_y		y position of the viewport (should be 0).
	*/
	void refreshOverlayDiagram(GLuint viewportWidth, GLuint viewportHeight, GLuint viewport_x = 0, GLuint viewport_y = 0);


	//! Sets position for a visualization point.
	/*!
		Warning: this only works on valid indices!
		\param[in] position					New position to be used.
		\param[in] color					New color for the visualization point.
		\param[in] index					Index of the visualization point.
		\param[in] positionIsNormalized		Whether the given position is in normalized diagram coordinate system or not.
	*/
	void setVisualizationPoint(glm::vec3 position, glm::vec3 color, int index, bool positionIsNormalized);

	//! Finds closest point to the query point and sets it as selected.
	/*!
		Iterates only editable curves (ambient temperature curve and dewpoint temperature curve).
		\param[in] queryPoint		Point for which we are finding its closest curve point.
	*/
	void findClosestSoundingPoint(glm::vec2 queryPoint);

	//! Moves the selected point (if it exists) to new coordinates.
	/*!
		\param[in] mouseCoords		Coordinates of the mouse.
	*/
	void moveSelectedPoint(glm::vec2 mouseCoords);

	//! Constructs a toolbar for diagram controls.
	/*!
		\param[in] ctx		Nuklear context to be used.
		\param[in] ui		User interface for which we are creating this toolbar.
	*/
	void constructDiagramCurvesToolbar(struct nk_context *ctx, UserInterface *ui);




private:

	const float temperatureNotchSize = 0.01f;		//!< Size of the temperature notches in the diagram

	const float maxTextScale = 0.0005f;				//!< Maximum text scale when zoomed in


	VariableManager *vars = nullptr;				//!< VariableManager for this diagram
	
	ShaderProgram *curveShader = nullptr;			//!< Shader to be used when drawing curves
	ShaderProgram *singleColorShaderVBO = nullptr;	//!< Shader that can color individual vertices
	ShaderProgram *overlayDiagramShader = nullptr;	//!< Shader for the overlay diagram

	vector<glm::vec3> visualizationPoints; //!< Helper points, buffered in such a way that it is point, color, point, color, etc. (stride = 3 * sizeof(float))

	GLuint visPointsVAO;						//!< VAO for the visualization points
	GLuint visPointsVBO;						//!< VBO for the visualization points

	vector<glm::vec3> mainParameterPoints;		//!< List of the main parameter points for visualization usage
	GLuint mainParameterPointsVAO;				//!< VAO for the main parameter points
	GLuint mainParameterPointsVBO;				//!< VBO for the main parameter points


	vector<glm::vec2> temperaturePoints;		//!< List of temperature points (where notches are to be displayed)

	GLuint isobarsVAO;					//!< VAO for the isobars
	GLuint isobarsVBO;					//!< VBO for the isobars
	int numIsobars;						//!< Number of isobars created

	GLuint temperaturePointsVAO;		//!< VAO for the temperature points
	GLuint temperaturePointsVBO;		//!< VBO for the temperature points
	int temperaturePointsCount;			//!< Number of temperature points created

	GLuint isothermsVAO;				//!< VAO for the isotherms
	GLuint isothermsVBO;				//!< VBO for the isotherms
	int isothermsCount;					//!< Number of the isotherms created

	GLuint ambientTemperatureVAO;		//!< VAO for the ambient temperature curve
	GLuint ambientTemperatureVBO;		//!< VBO for the ambient temperature curve

	GLuint dewTemperatureVAO;			//!< VAO for the dewpoint temperature curve
	GLuint dewTemperatureVBO;			//!< VBO for the dewpoint temperature curve

	GLuint isohumesVAO;					//!< VAO for the isohumes (mixing ratio lines)
	GLuint isohumesVBO;					//!< VBO for the isohumes (mixing ratio lines)

	GLuint dryAdiabatsVAO[2];			//!< Two VAOs for the general and profile dry adiabats
	GLuint dryAdiabatsVBO[2];			//!< Two VBOs for the general and profile dry adiabats
	int numDryAdiabats[2];				//!< Number of general and profile dry adiabats
	vector<int> dryAdiabatEdgeCount[2];	//!< Number of edges (in total) for general and profile dry adiabats

	GLuint moistAdiabatsVAO[2];				//!< Two VAOs for the general and profile moist adiabats
	GLuint moistAdiabatsVBO[2];				//!< Two VBOs for the general and profile moist adiabats
	int numMoistAdiabats[2];				//!< Number of general and profile moist adiabats
	vector<int> moistAdiabatEdgeCount[2];	//!< Number of edges (in total) for general and profile moist adiabats

	const float xmin = 0.0f;				//!< Minimum normalized diagram x coordinate
	const float xmax = 1.0f;				//!< Maximum normalized diagram x coordinate
	float ymin;								//!< Minimum normalized diagram y coordinate
	float ymax;								//!< Maximum normalized diagram y coordinate

	//! Generates temperature notches for the diagram visualization.
	void generateTemperatureNotches();

	//! Uploads the main parameters to their shared VBO.
	void uploadMainParameterPointsToBuffer();


};

