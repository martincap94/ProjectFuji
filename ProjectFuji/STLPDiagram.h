///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       STLPDiagram.h
* \author     Martin Cap
* \date       2019/01/18
* \brief	  Describes the SkewT/LogP class.
*
*  Describes the SkewT/LogP class that is drawn on the screen.
*
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


using namespace std;

class STLPDiagram {
public:



	TextRenderer *textRend;			///< Text renderer engine

	vector<SoundingDataItem> soundingData;	///< Sounding data loaded from file

	glm::vec2 DewpointNormalized;	///< Normalized dewpoint temperature and pressure
	glm::vec2 CCLNormalized;		///< Normalized convective condensation level
	glm::vec2 TcNormalized;			///< Normalized convective temperature
	glm::vec2 ELNormalized;			///< Normalized equilibrium level
	glm::vec2 LCLNormalized;
	glm::vec2 LFCNormalized;
	glm::vec2 OrographicELNormalized;

	glm::vec2 Dewpoint;				///< Dewpoint temperature and pressure
	glm::vec2 CCL;					///< Convective condensation level
	glm::vec2 Tc;					///< Convective temperature
	glm::vec2 EL;					///< Equilibrium level
	glm::vec2 LCL;
	glm::vec2 LFC;
	glm::vec2 OrographicEL;

	// helper curves
	Curve xaxis;					///< x axis curve (single line)
	Curve yaxis;					///< y axis curve (single line)

	Curve ambientCurve;				///< Sounding curve with ambient temperatures
	Curve dewpointCurve;			///< Sounding curve with dewpoint temperatures
	Curve TcDryAdiabat;				///< Dry adiabat that has convective temperature at ground level
	Curve moistAdiabat_CCL_EL;
	Curve moistAdiabat_LCL_EL;

	float minP;						///< Minimum pressure in normalized (diagram) coordinates 
	float maxP;						///< Maximum pressure in normalized (diagram) coordinates
	float minT;						///< Minimum temperature in normalized (diagram) coordinates
	float maxT;						///< Maximum temperature in normalized (diagram) coordinates

	string soundingFile;			///< Filename of the sounding file that will be loaded and displayed

	pair<Curve *, int> selectedPoint;

	int numProfiles = 10;
	float convectiveTempRange = 1.0f;
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
	GLuint quadVAO;
	GLuint quadVBO;


	/// Constructs the diagram instance without loading any sounding data.
	STLPDiagram();

	/// Constructs the diagram and loads the data from the given file.
	/**
		Constructs the diagram and loads the data from the given file.
		\param[in] filename		Name of file that contains the sounding data in appropriate form.
	*/
	STLPDiagram(string filename);

	/// Deallocates data - destroys the text renderer engine in particular.
	~STLPDiagram();

	/// Initializes the diagram from given sounding data.
	void init(string filename);

	/// Loads the sounding data to the soundingData vector.
	void loadSoundingData(string filename);

	/// Initializes all buffers and therefore curves of the diagram.
	/**
		Initializes all buffers and therefore curves of the diagram.
		Prototype function that does all the work - should be separated into multiple
		methods and utilize Curve class that will be extended.
	*/
	void initBuffersNormalized();


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
	void draw(ShaderProgram &shader, ShaderProgram &altShader);

	/// Draws the text label for the diagram.
	void drawText(ShaderProgram &shader);

	/// Only sets existing point!
	void setVisualizationPoint(glm::vec3 position, glm::vec3 color, int index, bool positionIsNormalized);

	void findClosestSoundingPoint(glm::vec2 queryPoint);
	void moveSelectedPoint(glm::vec2 mouseCoords);


private:
	
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


	GLuint dryAdiabatsVAO;
	GLuint dryAdiabatsVBO;
	int numDryAdiabats;
	vector<int> dryAdiabatEdgeCount; // quick fix

	GLuint moistAdiabatsVAO;
	GLuint moistAdiabatsVBO;
	int numMoistAdiabats;
	int numMoistAdiabatEdges;
	vector<int> moistAdiabatEdgeCount; // quick fix


	GLuint CCLVAO;
	GLuint CCLVBO;

	GLuint TcVAO;
	GLuint TcVBO;


};

