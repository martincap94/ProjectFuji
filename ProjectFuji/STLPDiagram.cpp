#include "STLPDiagram.h"

#include <iostream>
#include <fstream>
#include "Config.h"

#include "STLPUtils.h"
#include "Utils.h"
#include "MainFramebuffer.h"
#include "ShaderManager.h"



STLPDiagram::STLPDiagram(VariableManager *vars) : vars(vars) {
	init();
}


STLPDiagram::~STLPDiagram() {
	delete textRend;
}

void STLPDiagram::init() {

	soundingFilename = string(SOUNDING_DATA_DIR) + vars->soundingFile;
	tmpSoundingFilename = soundingFilename;

	curveShader = ShaderManager::getShaderPtr("curve");
	singleColorShaderVBO = ShaderManager::getShaderPtr("singleColor_VBO");
	overlayDiagramShader = ShaderManager::getShaderPtr("overlayTexture");


	loadSoundingData();

	initFreetype();

	initBuffers();
	CHECK_GL_ERRORS();
	initCurves();
	CHECK_GL_ERRORS();
	initOverlayDiagram();
	CHECK_GL_ERRORS();
}

void STLPDiagram::loadSoundingData() {

	soundingFilename = tmpSoundingFilename; // not important at initialization; used when user edits the diagram
	soundingFilenameChanged = false;

	//cout << "Sounding filename = " << filename << endl;
	ifstream infile(soundingFilename);

	if (!infile.is_open()) {
		cerr << soundingFilename << " could not be opened!" << endl;
		exit(EXIT_FAILURE);
	}
	// assume the file is in correct format!
	string line;

	// read first line (the header)
	getline(infile, line);

	if (soundingData.size() > 0) {
		soundingData.clear();
	}
	if (windData.size() > 0) {
		windData.clear();
	}

	SoundingDataItem tmp;
	while (infile.good()) {
		infile >> tmp.data[PRES];
		infile >> tmp.data[HGHT];
		infile >> tmp.data[TEMP];
		infile >> tmp.data[DWPT];
		infile >> tmp.data[RELH];
		infile >> tmp.data[MIXR];
		infile >> tmp.data[DRCT];
		infile >> tmp.data[SKNT];
		infile >> tmp.data[TWTB];
		infile >> tmp.data[TVRT];
		infile >> tmp.data[THTA];
		infile >> tmp.data[THTE];
		infile >> tmp.data[THTV];
		soundingData.push_back(tmp);
		//soundingData.back().print();

		WindDataItem wdi;
		wdi.delta_x = 0.514444f * tmp.data[SKNT] * cos(tmp.data[DRCT]); // do not forget to convert from knots to meters per second
		wdi.delta_z = 0.514444f * tmp.data[SKNT] * sin(tmp.data[DRCT]); // do not forget to convert from knots to meters per second
		//wdi.y = tmp.data[HGHT];
		wdi.y = getAltitudeFromPressure(tmp.data[PRES]);

		windData.push_back(wdi);

	}

	minT = MIN_TEMP;
	maxT = MAX_TEMP;
	minP = MIN_P;
	maxP = MAX_P;
	P0 = soundingData[0].data[PRES];
	maxVerticesPerCurve = (int)((maxP - minP) / CURVE_DELTA_P + 1.0f);

	groundAltitude = getAltitudeFromPressure(P0);
	//cout << "ground altitude (computed) = " << groundAltitude << endl;
	//cout << "ground altitude (sounding) = " << soundingData[0].data[HGHT] << endl;


}



void STLPDiagram::generateIsobars() {
	vector<glm::vec2> vertices;

	numIsobars = 0;
	float P;
	for (P = MAX_P; P >= MIN_P; P -= CURVE_DELTA_P) {
		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
		//P = soundingData[profileIndex].data[PRES];
		float y = getNormalizedPres(P);
		vertices.push_back(glm::vec2(xmin, y));
		vertices.push_back(glm::vec2(xmax, y));
		numIsobars++;
	}
	//float y;
	//P = soundingData[0].data[PRES];
	//y = getNormalizedPres(P);
	//vertices.push_back(glm::vec2(xmin, y));
	//vertices.push_back(glm::vec2(xmax, y));
	numIsobars++;


	glBindBuffer(GL_ARRAY_BUFFER, isobarsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
}

void STLPDiagram::generateIsotherms() {

	vector<glm::vec2> vertices;
	float x;
	float y;

	isothermsCount = 0;
	for (int i = MIN_TEMP - 80.0f; i <= MAX_TEMP; i += 10) {

		y = ymax;
		x = getNormalizedTemp(i, y);
		vertices.push_back(glm::vec2(x, y));

		y = ymin;
		x = getNormalizedTemp(i, y);
		vertices.push_back(glm::vec2(x, y));

		isothermsCount++;
	}
	
	
	glBindBuffer(GL_ARRAY_BUFFER, isothermsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

}

void STLPDiagram::initDewpointCurve() {
	vector<glm::vec2> vertices;
	float x;
	float y;

	for (int i = 0; i < soundingData.size(); i++) {

		float P = soundingData[i].data[PRES];
		float T = soundingData[i].data[DWPT];

		y = getNormalizedPres(P);
		x = getNormalizedTemp(T, y);

		vertices.push_back(glm::vec2(x, y));
	}
	dewpointCurve.vertices = vertices;


	glBindBuffer(GL_ARRAY_BUFFER, dewTemperatureVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

}

void STLPDiagram::initAmbientTemperatureCurve() {
	vector<glm::vec2> vertices;
	float x;
	float y;
	for (int i = 0; i < soundingData.size(); i++) {

		float P = soundingData[i].data[PRES];
		float T = soundingData[i].data[TEMP];

		y = getNormalizedPres(P);
		x = getNormalizedTemp(T, y);

		vertices.push_back(glm::vec2(x, y));
	}

	ambientCurve.vertices = vertices;
	//ambientCurve.printVertices();

	glBindBuffer(GL_ARRAY_BUFFER, ambientTemperatureVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
}

void STLPDiagram::generateMixingRatioLine() {
	vector<glm::vec2> vertices;


	float Rd = 287.05307f;	// gas constant for dry air [J kg^-1 K^-1]
	float Rm = 461.5f;		// gas constant for moist air [J kg^-1 K^-1]

							// w(T,P) = (eps * e(T)) / (P - e(T))
							// where
							//		eps = Rd / Rm
							//		e(T) ... saturation vapor pressure (can be approx'd by August-Roche-Magnus formula
							//		e(T) =(approx)= C exp( (A*T) / (T + B))
							//		where
							//				A = 17.625
							//				B = 243.04
							//				C = 610.94

	float A = 17.625f;
	float B = 243.04f;
	float C = 610.94f;

	// given that w(T,P) is const., let W = w(T,P), we can express T in terms of P (see Equation 3.13)

	// to determine the mixing ratio line that passes through (T,P), we calculate the value of the temperature
	// T(P + delta) where delta is a small integer
	// -> the points (T,P) nad (T(P + delta), P + delta) define a mixing ratio line, whose points all have the same mixing ratio


	// Compute CCL using a mixing ratio line
	float w0 = soundingData[0].data[MIXR];

	float P = soundingData[0].data[PRES];


	//float T = soundingData[0].data[DWPT]; // default computation from initial sounding data, does not take changes to curves into consideration
	float T = getDenormalizedTemp(findIntersection(groundIsobar, dewpointCurve).x, getNormalizedPres(P));	// this is more general (when user changes dewpoint curve for example)



	float eps = Rd / Rm;
	//float satVP = C * exp((A * T) / (T + B));	// saturation vapor pressure: e(T)
	//float satVP = e_s_degC(T) / 100.0f;
	//float W = (eps * satVP) / (P - satVP);
	float W = w_degC(T, P * 100.0f);

	//cout << " -> Computed W = " << W << endl;

	float deltaP = 20.0f;


	float x, y;
	while (P >= MIN_P) {


		float fracPart = log((W * P * 100.0f) / (C * (W + eps)));
		float computedT = (B * fracPart) / (A - fracPart);

		//cout << "P = " << P << "[hPa], T = " << computedT << "[deg C]" << endl;

		y = getNormalizedPres(P);
		x = getNormalizedTemp(computedT, y);

		//if (x < xmin || x > xmax || y < 0.0f || y > 1.0f) {
		//	break;
		//}

		vertices.push_back(glm::vec2(x, y));


		float delta = 10.0f; // should be a small integer - produces dashed line (logarithmic)
							 //float offsetP = P - delta;
		float offsetP = P - deltaP; // produces continuous line
		fracPart = log((W * offsetP * 100.0f) / (C * (W + eps)));
		computedT = (B * fracPart) / (A - fracPart);

		y = getNormalizedPres(offsetP);
		x = getNormalizedTemp(computedT, y);
		vertices.push_back(glm::vec2(x, y));

		P -= deltaP;

	}
	mixingCCL.vertices = vertices;

	glBindBuffer(GL_ARRAY_BUFFER, isohumesVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
}





void STLPDiagram::generateMixingRatioLineExperimental() {
	vector<glm::vec2> vertices;


	float Rd = 287.05307f;	// gas constant for dry air [J kg^-1 K^-1]
	float Rm = 461.5f;		// gas constant for moist air [J kg^-1 K^-1]

							// w(T,P) = (eps * e(T)) / (P - e(T))
							// where
							//		eps = Rd / Rm
							//		e(T) ... saturation vapor pressure (can be approx'd by August-Roche-Magnus formula
							//		e(T) =(approx)= C exp( (A*T) / (T + B))
							//		where
							//				A = 17.625
							//				B = 243.04
							//				C = 610.94

	float A = 17.625f;
	float B = 243.04f;
	float C = 610.94f;

	// given that w(T,P) is const., let W = w(T,P), we can express T in terms of P (see Equation 3.13)

	// to determine the mixing ratio line that passes through (T,P), we calculate the value of the temperature
	// T(P + delta) where delta is a small integer
	// -> the points (T,P) nad (T(P + delta), P + delta) define a mixing ratio line, whose points all have the same mixing ratio


	// Compute CCL using a mixing ratio line
	float w0 = soundingData[0].data[MIXR];

	float P = P0;


	//float T = soundingData[0].data[DWPT]; // default computation from initial sounding data, does not take changes to curves into consideration
	float T = getDenormalizedTemp(findIntersection(groundIsobar, dewpointCurve).x, getNormalizedPres(P));	// this is more general (when user changes dewpoint curve for example)



	float eps = Rd / Rm;
	//float satVP = C * exp((A * T) / (T + B));	// saturation vapor pressure: e(T)
	//float satVP = e_s_degC(T) / 100.0f;
	//float W = (eps * satVP) / (P - satVP);
	float W = w_degC(T, P * 100.0f);

	//cout << " -> Computed W = " << W << endl;

	float deltaP = 20.0f;

	float segmentDeltaPVis = 10.0f; // should be a small integer - produces dashed line (logarithmic)
	float segmentDeltaP = deltaP; // for intersection search, we need a continuous line

	float x, y;

	float fracPart = log((W * P * 100.0f) / (C * (W + eps)));
	float computedT = (B * fracPart) / (A - fracPart);

	y = getNormalizedPres(P);
	x = getNormalizedTemp(computedT, y);

	mixingCCL.vertices.push_back(glm::vec2(x, y));
	vertices.push_back(glm::vec2(x, y));
	
	while (P >= MIN_P) {

		float offsetP = P - segmentDeltaP; // produces continuous line
		fracPart = log((W * offsetP * 100.0f) / (C * (W + eps)));
		computedT = (B * fracPart) / (A - fracPart);

		y = getNormalizedPres(offsetP);
		x = getNormalizedTemp(computedT, y);

		mixingCCL.vertices.push_back(glm::vec2(x, y));
		vertices.push_back(glm::vec2(x, y));

		P -= deltaP;

	}

	
	glBindBuffer(GL_ARRAY_BUFFER, isohumesVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
}

void STLPDiagram::generateDryAdiabat(float theta, vector<glm::vec2> &vertices, int mode, float P0, vector<int> *edgeCounter, bool incrementCounter, float deltaP, Curve *curve) {

	int vertexCounter = 0;

	if (curve != nullptr) {
		curve->vertices.clear();
	}

	float x, y, T, P;

	/*
		We want the dry adiabat to adhere to the given range (P0 to MIN_P) for calculation purposes.
		This is only meant for extreme cases when the particles are out of bounds. In these cases, we clamp
		the adiabat minus ambient temperature computation to their last viable values. For this we need these
		last values to be in the same pressure level (on the same isobar).
	*/

	// We need to compute the initial step to snap to the isobars.


	//printf(" P0 = %0.1f\n deltaP = %0.1f\n ", P0, deltaP);

	float nextMultipleP = P0 + (deltaP - fmodf(P0, deltaP)) - deltaP;
	float firstDeltaP = P0 - nextMultipleP;

	//printf(" next multiple P = %0.1f\n first delta P = %0.1f\n ", nextMultipleP, firstDeltaP);

	P = P0;
	T = computeAbsoluteFromThetaC(theta, P, this->P0);

	y = getNormalizedPres(P);
	x = getNormalizedTemp(T, y);

	vertices.push_back(glm::vec2(x, y));
	if (curve != nullptr) {
		curve->vertices.push_back(glm::vec2(x, y));
	}
	vertexCounter++;


	for (P = nextMultipleP; P >= MIN_P; P -= deltaP) {

		T = computeAbsoluteFromThetaC(theta, P, this->P0);

		y = getNormalizedPres(P);
		x = getNormalizedTemp(T, y);

		vertices.push_back(glm::vec2(x, y));
		if (curve != nullptr) {
			curve->vertices.push_back(glm::vec2(x, y));
		}
		vertexCounter++;
	}
	//cout << vertexCounter << endl;
	//cout << ((MAX_P - MIN_P) / CURVE_DELTA_P + 1) << endl;

	if (incrementCounter && edgeCounter != nullptr) {
		numDryAdiabats[mode]++;
		edgeCounter->push_back(vertexCounter);
		//dryAdiabatEdgeCount.push_back(vertexCounter);
	}


}

void STLPDiagram::generateMoistAdiabat(float startT, float startP, vector<glm::vec2>& vertices, int mode, float P0, vector<int>* edgeCounter, bool incrementCounter, float deltaP, Curve * curve, float smallDeltaP) {

	float T = getKelvin(startT);

	int vertexCounter = 0;
	float x, y;

	if (curve != nullptr) {
		curve->vertices.clear();
	}

	//cout << "Generating moist adiabat..." << endl;
	//printf(" | delta P = %0.2f, small delta P = %0.2f\n", deltaP, smallDeltaP);
	float P_Pa;
	float accumulatedP = 0.0f;
	const float smallDeltaP_Pa = smallDeltaP * 100.0f;
	for (float P = startP; P >= MIN_P - smallDeltaP; P -= smallDeltaP) {


		//printf("   | pressure = %0.2f, (int)pressure = %d, (int)pressure % (int)deltaP = %d\n", p, (int)p, (int)p % (int)deltaP);

		//if ((int)p % (int)deltaP == 0 || p == startP || p <= MIN_P) {

		if (accumulatedP >= deltaP || accumulatedP == 0.0f || P <= MIN_P) {

			accumulatedP = 0.0f;
			//cout << "---- ADDING VERTEX ----" << endl;

			y = getNormalizedPres(P);
			x = getNormalizedTemp(getCelsius(T), y);

			vertices.push_back(glm::vec2(x, y));
			if (curve != nullptr) {
				curve->vertices.push_back(glm::vec2(x, y));
			}
			vertexCounter++;
		}

		P_Pa = P * 100.0f;
		//T -= dTdP_moist_degK_Bakhshaii(T, P_Pa) * smallDeltaP_Pa;
		T -= getMoistAdiabatIntegralVal(T, P_Pa) * smallDeltaP_Pa;

		accumulatedP += smallDeltaP;

	}
	//cout << endl;
	//cout << "Vertex counter = " << vertexCounter << endl;
	if (incrementCounter && edgeCounter != nullptr) {
		numMoistAdiabats[mode]++;
		edgeCounter->push_back(vertexCounter);
	}





}



void STLPDiagram::recalculateAll() {



	// Show diagram between minimal & maximal pressure values
	ymin = getNormalizedPres(MIN_P);
	ymax = getNormalizedPres(MAX_P);


	P0 = soundingData[0].data[PRES];
	float y0 = getNormalizedPres(P0);


	float T;
	float P;

	xaxis.vertices.clear();
	yaxis.vertices.clear();

	xaxis.vertices.push_back(glm::vec2(xmin, ymax));
	xaxis.vertices.push_back(glm::vec2(xmax, ymax));
	yaxis.vertices.push_back(glm::vec2(xmin, ymin));
	yaxis.vertices.push_back(glm::vec2(xmin, ymax));

	xaxis.uploadToBuffers();
	yaxis.uploadToBuffers();

	groundIsobar.vertices.clear();
	groundIsobar.vertices.push_back(glm::vec2(xmin, y0));
	groundIsobar.vertices.push_back(glm::vec2(xmax * 10.0f, y0)); // we want it to be long for Tc computation (intersection beyond xmax)

	groundIsobar.uploadToBuffers();




	///////////////////////////////////////////////////////////////////////////////////////
	// ISOBARS
	///////////////////////////////////////////////////////////////////////////////////////
	generateIsobars();

	///////////////////////////////////////////////////////////////////////////////////////
	// TEMPERATURE POINTS
	///////////////////////////////////////////////////////////////////////////////////////
	generateTemperatureNotches();

	///////////////////////////////////////////////////////////////////////////////////////
	// ISOTHERMS
	///////////////////////////////////////////////////////////////////////////////////////
	generateIsotherms();

	///////////////////////////////////////////////////////////////////////////////////////
	// AMBIENT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////
	initAmbientTemperatureCurve();

	///////////////////////////////////////////////////////////////////////////////////////
	// DEWPOINT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////
	initDewpointCurve();

	recalculateParameters();

}

void STLPDiagram::initBuffers() {

	xaxis.initBuffers();
	yaxis.initBuffers();
	groundIsobar.initBuffers();


	///////////////////////////////////////////////////////////////////////////////////////
	// ISOBARS
	///////////////////////////////////////////////////////////////////////////////////////

	glGenVertexArrays(1, &isobarsVAO);
	glBindVertexArray(isobarsVAO);
	glGenBuffers(1, &isobarsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, isobarsVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);

	///////////////////////////////////////////////////////////////////////////////////////
	// TEMPERATURE POINTS
	///////////////////////////////////////////////////////////////////////////////////////

	glGenVertexArrays(1, &temperaturePointsVAO);
	glBindVertexArray(temperaturePointsVAO);
	glGenBuffers(1, &temperaturePointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, temperaturePointsVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// ISOTHERMS
	///////////////////////////////////////////////////////////////////////////////////////
	glGenVertexArrays(1, &isothermsVAO);
	glBindVertexArray(isothermsVAO);
	glGenBuffers(1, &isothermsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, isothermsVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// AMBIENT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////

	glGenVertexArrays(1, &ambientTemperatureVAO);
	glBindVertexArray(ambientTemperatureVAO);
	glGenBuffers(1, &ambientTemperatureVBO);
	glBindBuffer(GL_ARRAY_BUFFER, ambientTemperatureVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// DEWPOINT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////

	glGenVertexArrays(1, &dewTemperatureVAO);
	glBindVertexArray(dewTemperatureVAO);
	glGenBuffers(1, &dewTemperatureVBO);
	glBindBuffer(GL_ARRAY_BUFFER, dewTemperatureVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// ISOHUMES (MIXING RATIO LINES)
	///////////////////////////////////////////////////////////////////////////////////////

	glGenVertexArrays(1, &isohumesVAO);
	glBindVertexArray(isohumesVAO);
	glGenBuffers(1, &isohumesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, isohumesVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < 2; i++) {
		glGenVertexArrays(1, &dryAdiabatsVAO[i]);
		glBindVertexArray(dryAdiabatsVAO[i]);
		glGenBuffers(1, &dryAdiabatsVBO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, dryAdiabatsVBO[i]);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

		glBindVertexArray(0);
	}

	///////////////////////////////////////////////////////////////////////////////////////
	// MOIST ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < 2; i++) {
		glGenVertexArrays(1, &moistAdiabatsVAO[i]);
		glBindVertexArray(moistAdiabatsVAO[i]);
		glGenBuffers(1, &moistAdiabatsVBO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, moistAdiabatsVBO[i]);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

		glBindVertexArray(0);
	}




	///////////////////////////////////////////////////////////////////////////////////////
	// MAIN PARAMETER POINTS
	///////////////////////////////////////////////////////////////////////////////////////

	glGenVertexArrays(1, &mainParameterPointsVAO);
	glBindVertexArray(mainParameterPointsVAO);
	glGenBuffers(1, &mainParameterPointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, mainParameterPointsVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);

}




void STLPDiagram::initCurves() {


	recalculateProfileDelta();

	// Initialize main variables


	// Show diagram between minimal & maximal pressure values
	ymin = getNormalizedPres(MIN_P);
	ymax = getNormalizedPres(MAX_P);


	P0 = soundingData[0].data[PRES];
	float y0 = getNormalizedPres(P0);


	float T;
	float P;

	xaxis.vertices.clear();
	yaxis.vertices.clear();

	xaxis.vertices.push_back(glm::vec2(xmin, ymax));
	xaxis.vertices.push_back(glm::vec2(xmax, ymax));
	yaxis.vertices.push_back(glm::vec2(xmin, ymin));
	yaxis.vertices.push_back(glm::vec2(xmin, ymax));

	xaxis.uploadToBuffers();
	yaxis.uploadToBuffers();

	groundIsobar.vertices.clear();
	groundIsobar.vertices.push_back(glm::vec2(xmin, y0));
	groundIsobar.vertices.push_back(glm::vec2(xmax * 10.0f, y0)); // we want it to be long for Tc computation (intersection beyond xmax)

	groundIsobar.uploadToBuffers();



	///////////////////////////////////////////////////////////////////////////////////////
	// ISOBARS
	///////////////////////////////////////////////////////////////////////////////////////
	generateIsobars();

	///////////////////////////////////////////////////////////////////////////////////////
	// TEMPERATURE POINTS
	///////////////////////////////////////////////////////////////////////////////////////
	generateTemperatureNotches();

	///////////////////////////////////////////////////////////////////////////////////////
	// ISOTHERMS
	///////////////////////////////////////////////////////////////////////////////////////
	generateIsotherms();

	///////////////////////////////////////////////////////////////////////////////////////
	// AMBIENT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////
	initAmbientTemperatureCurve();

	///////////////////////////////////////////////////////////////////////////////////////
	// DEWPOINT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////
	initDewpointCurve();

	///////////////////////////////////////////////////////////////////////////////////////
	// ISOHUMES (MIXING RATIO LINES)
	///////////////////////////////////////////////////////////////////////////////////////

	generateMixingRatioLine();

	CCLFound = findIntersectionNew(mixingCCL, ambientCurve, CCLNormalized);
	if (CCLFound) {
		CCL = getDenormalizedCoords(CCLNormalized);
	}

	///////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	vector<glm::vec2> vertices;

	vertices.clear();
	numDryAdiabats[0] = 0;
	int counter;

	for (float theta = MIN_TEMP; theta <= MAX_TEMP * 5; theta += dryAdiabatDeltaT) {
		generateDryAdiabat(theta, vertices, 0, P0, &dryAdiabatEdgeCount[0]);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: Tc <-> CCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	float theta = computeThetaFromAbsoluteC(CCL.x, CCL.y, P0);
	generateDryAdiabat(theta, vertices, 0, P0, &dryAdiabatEdgeCount[0], true, CURVE_DELTA_P, &TcDryAdiabat);

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: Ground ambient temperature <-> LCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	theta = computeThetaFromAbsoluteC(soundingData[0].data[TEMP], P0, P0);
	generateDryAdiabat(theta, vertices, 0, P0, &dryAdiabatEdgeCount[0], true, CURVE_DELTA_P, &LCLDryAdiabatCurve);


	///////////////////////////////////////////////////////////////////////////////////////////////
	// LCL and Tc
	///////////////////////////////////////////////////////////////////////////////////////////////


	// Check correctness by computing thetaCCL == Tc
	float thetaCCL = (CCL.x + 273.15f) * pow((P0 / CCL.y), k_ratio);
	thetaCCL -= 273.15f;

	LCLFound = findIntersectionNew(LCLDryAdiabatCurve, mixingCCL, LCLNormalized);
	if (LCLFound) {
		LCL = getDenormalizedCoords(LCLNormalized);
	}

	if (useOrographicParameters) {
		TcFound = true;
		TcNormalized = ambientCurve.vertices[0];
	} else {
		TcFound = findIntersectionNew(groundIsobar, TcDryAdiabat, TcNormalized);
	}
	if (TcFound) {
		Tc = getDenormalizedCoords(TcNormalized);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////



	// PROFILES
	for (int i = 0; i < numProfiles; i++) {
		TcProfiles.push_back(Tc + glm::vec2(i * profileDelta, 0.0f));
		//visualizationPoints.push_back(glm::vec3(getNormalizedCoords(TcProfiles.back()), -2.0f)); // point
		//float tint = (i + 1) * profileDelta;
		//rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		//visualizationPoints.push_back(glm::vec3(tint, 0.0f, 0.0f)); // color	
	}



	glBindBuffer(GL_ARRAY_BUFFER, dryAdiabatsVBO[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	vertices.clear();
	numDryAdiabats[1] = 0;

	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {

		dryAdiabatProfiles.push_back(Curve());

		float theta = computeAbsoluteFromThetaC(TcProfiles[profileIndex].x, P0, P0);
		generateDryAdiabat(theta, vertices, 1, P0, &dryAdiabatEdgeCount[1], true, CURVE_DELTA_P, &dryAdiabatProfiles[profileIndex]);

		if (useOrographicParameters) {
			CCLProfiles.push_back(getDenormalizedCoords(findIntersection(dryAdiabatProfiles[profileIndex], mixingCCL)));
		} else {
			CCLProfiles.push_back(getDenormalizedCoords(findIntersection(dryAdiabatProfiles[profileIndex], ambientCurve)));
		}

		//visualizationPoints.push_back(glm::vec3(getNormalizedCoords(CCLProfiles.back()), -2.0f)); // point
		//float tint = (profileIndex + 1) * profileDelta;
		//rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		//visualizationPoints.push_back(glm::vec3(tint, 0.0f, 1.0f)); // color	

	}


	glBindBuffer(GL_ARRAY_BUFFER, dryAdiabatsVBO[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);



	///////////////////////////////////////////////////////////////////////////////////////
	// MOIST ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	/*cout << "////////////////////////////////////////////////////" << endl;
	cout << "// MOIST ADIABATS" << endl;
	cout << "////////////////////////////////////////////////////" << endl;*/


	vertices.clear();

	numMoistAdiabats[0] = 0;


	/*float a = -6.14342f * 0.00001f;
	float b = 1.58927 * 0.001f;
	float c = -2.36418f;
	float d = 2500.79f;

	float g = -9.81f;*/
	// Lv(T) = (aT^3 + bT^2 + cT + d) * 1000
	// Lv(T)	... latent heat of vaporisation/condensation
	//for (float T = MIN_TEMP; T <= MAX_TEMP; T++) {
	//float Lv = (a*T*T*T + b*T*T + c*T + d) * 1000.0f;

	float currP;

	// General moist adiabats (not part of the parameter calculations)
	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += moistAdiabatDeltaT) {
		generateMoistAdiabat(currT, MAX_P, vertices, 0, P0, &moistAdiabatEdgeCount[0]);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// EL (regular) computation - special moist adiabat (goes through CCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(CCL.x, CCL.y, vertices, 0, P0, &moistAdiabatEdgeCount[0], true, CURVE_DELTA_P, &moistAdiabat_CCL_EL);

	{
		///////////////////////////////////////////////////////////////////////////////////////////////
		// Find EL 
		///////////////////////////////////////////////////////////////////////////////////////////////

		ELFound = findIntersectionNew(moistAdiabat_CCL_EL, ambientCurve, ELNormalized, 1, true);
		//cout << "EL (normalized): x = " << ELNormalized.x << ", y = " << ELNormalized.y << endl;
		if (ELFound) {
			EL = getDenormalizedCoords(ELNormalized);
		}
		//cout << "EL: T = " << EL.x << ", P = " << EL.y << endl;

		//visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
		//visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	

		//visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
		//visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	
	}


	///////////////////////////////////////////////////////////////////////////////////////////////
	// EL (orographic) computation - special moist adiabat (goes through LCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(LCL.x, LCL.y, vertices, 0, P0, &moistAdiabatEdgeCount[0], true, CURVE_DELTA_P, &moistAdiabat_LCL_EL);
	{

		LFCFound = findIntersectionNew(moistAdiabat_LCL_EL, ambientCurve, LFCNormalized);
		if (LFCFound) {
			LFC = getDenormalizedCoords(LFCNormalized);
		}

	    orographicELFound = findIntersectionNew(moistAdiabat_LCL_EL, ambientCurve, orographicELNormalized, 2, true);
		if (orographicELFound) {
			orographicEL = getDenormalizedCoords(orographicELNormalized);
		}

	}

	glBindBuffer(GL_ARRAY_BUFFER, moistAdiabatsVBO[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);



	///////////////////////////////////////////////////////////////////////////////////////////////
	// Moist adiabat profiles - used in simulation
	///////////////////////////////////////////////////////////////////////////////////////////////
	vertices.clear();
	numMoistAdiabats[1] = 0;

	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
		//int counter = 0;
		moistAdiabatProfiles.push_back(Curve());
		generateMoistAdiabat(CCLProfiles[profileIndex].x, CCLProfiles[profileIndex].y, vertices, 1, P0, &moistAdiabatEdgeCount[1], true, CURVE_DELTA_P, &moistAdiabatProfiles[profileIndex], 0.2f);

		glm::vec2 tmp = findIntersection(moistAdiabatProfiles[profileIndex], ambientCurve, true);
		ELProfiles.push_back(getDenormalizedCoords(tmp));

		//visualizationPoints.push_back(glm::vec3(getNormalizedCoords(ELProfiles.back()), -2.0f)); // point
		//float tint = (profileIndex + 1) * profileDelta;
		//rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		//visualizationPoints.push_back(glm::vec3(tint, 1.0f, 1.0f)); // color	
	}



	//numMoistAdiabats++;
	glBindBuffer(GL_ARRAY_BUFFER, moistAdiabatsVBO[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);





	// trying out stuff
	P = 432.2f;
	float normP = getNormalizedPres(P);
	////cout << "Pressure = " << P << ", normalized pressure = " << normP << endl;
	//visualizationPoints.push_back(glm::vec3(ambientCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
	//visualizationPoints.push_back(glm::vec3(1.0f, 0.0f, 0.0f)); // color

	//visualizationPoints.push_back(glm::vec3(dewpointCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
	//visualizationPoints.push_back(glm::vec3(0.0f, 0.0f, 1.0f)); // color


	glGenVertexArrays(1, &visPointsVAO);
	glBindVertexArray(visPointsVAO);
	glGenBuffers(1, &visPointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, visPointsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * visualizationPoints.size(), &visualizationPoints[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);


	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)(sizeof(glm::vec3)));

	glBindVertexArray(0);




	// Main parameters visualization
	uploadMainParameterPointsToBuffer();


}






void STLPDiagram::recalculateParameters() {

	recalculateProfileDelta();


	diagramChanged = false;
	useOrographicParametersChanged = false;
	useOrographicParameters = useOrographicParametersEdit;

	generateMixingRatioLine();

	CCLNormalized = findIntersection(mixingCCL, ambientCurve);
	CCL = getDenormalizedCoords(CCLNormalized);


	vector<glm::vec2> vertices;

	for (int i = 0; i < 2; i++) {
		numDryAdiabats[i] = 0;
		dryAdiabatEdgeCount[i].clear();
	}

	for (float theta = MIN_TEMP; theta <= MAX_TEMP * 5; theta += dryAdiabatDeltaT) {
		generateDryAdiabat(theta, vertices, 0, P0, &dryAdiabatEdgeCount[0]);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: T_c <-> CCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	float theta = computeThetaFromAbsoluteC(CCL.x, CCL.y, P0);
	TcDryAdiabat.vertices.clear(); // hack
	generateDryAdiabat(theta, vertices, 0, P0, &dryAdiabatEdgeCount[0], true, CURVE_DELTA_P, &TcDryAdiabat);

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: Ground ambient temperature <-> LCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	theta = computeThetaFromAbsoluteC(soundingData[0].data[TEMP], P0, P0); // will need line-to-line intersection for the ground temperature
	LCLDryAdiabatCurve.vertices.clear(); // hack
	generateDryAdiabat(theta, vertices, 0, P0, &dryAdiabatEdgeCount[0], true, CURVE_DELTA_P, &LCLDryAdiabatCurve);



	///////////////////////////////////////////////////////////////////////////////////////////////
	// Tc and LCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	LCLFound = findIntersectionNew(LCLDryAdiabatCurve, mixingCCL, LCLNormalized);
	if (LCLFound) {
		LCL = getDenormalizedCoords(LCLNormalized);
	}

	if (useOrographicParameters) {
		TcFound = true;
		TcNormalized = ambientCurve.vertices[0];
	} else {
		TcFound = findIntersectionNew(groundIsobar, TcDryAdiabat, TcNormalized);
	}
	if (TcFound) {
		Tc = getDenormalizedCoords(TcNormalized);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////

	glNamedBufferData(dryAdiabatsVBO[0], sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	vertices.clear();

	//// Profiles
	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
		TcProfiles[profileIndex] = Tc + glm::vec2((profileIndex + 1) * profileDelta, 0.0f);

		float theta = computeAbsoluteFromThetaC(TcProfiles[profileIndex].x, P0, P0);
		dryAdiabatProfiles[profileIndex].vertices.clear();
		generateDryAdiabat(theta, vertices, 1, P0, &dryAdiabatEdgeCount[1], true, CURVE_DELTA_P, &dryAdiabatProfiles[profileIndex]);

		//CCLProfiles[profileIndex] = getDenormalizedCoords(findIntersection(dryAdiabatProfiles[profileIndex], ambientCurve));

		if (useOrographicParameters) {
			CCLProfiles[profileIndex] = getDenormalizedCoords(findIntersection(dryAdiabatProfiles[profileIndex], mixingCCL));
		} else {
			CCLProfiles[profileIndex] = getDenormalizedCoords(findIntersection(dryAdiabatProfiles[profileIndex], ambientCurve));
		}
	}

	//int sum = 0;
	//for (int i = 0; i < numDryAdiabats - numProfiles - 2; i++) {
	//	sum += dryAdiabatEdgeCount[i];
	//}

	glNamedBufferData(dryAdiabatsVBO[1], sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);




	vertices.clear();
	for (int i = 0; i < 2; i++) {
		numMoistAdiabats[i] = 0;
		moistAdiabatEdgeCount[i].clear();
	}

	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += moistAdiabatDeltaT) {
		generateMoistAdiabat(currT, MAX_P, vertices, 0, P0, &moistAdiabatEdgeCount[0]);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// EL (regular) computation - special moist adiabat (goes through CCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(CCL.x, CCL.y, vertices, 0, P0, &moistAdiabatEdgeCount[0], true, CURVE_DELTA_P, &moistAdiabat_CCL_EL);


	ELFound = findIntersectionNew(moistAdiabat_CCL_EL, ambientCurve, ELNormalized, 1, true);
	//cout << "EL (normalized): x = " << ELNormalized.x << ", y = " << ELNormalized.y << endl;
	if (ELFound) {
		EL = getDenormalizedCoords(ELNormalized);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// EL (orographic) computation - special moist adiabat (goes through LCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(LCL.x, LCL.y, vertices, 0, P0, &moistAdiabatEdgeCount[0], true, CURVE_DELTA_P, &moistAdiabat_LCL_EL);

	LFCFound = findIntersectionNew(moistAdiabat_LCL_EL, ambientCurve, LFCNormalized);
	if (LFCFound) {
		LFC = getDenormalizedCoords(LFCNormalized);
	}

	orographicELFound = findIntersectionNew(moistAdiabat_LCL_EL, ambientCurve, orographicELNormalized, 2, true);
	if (orographicELFound) {
		orographicEL = getDenormalizedCoords(orographicELNormalized);
	}

	glNamedBufferData(moistAdiabatsVBO[0], sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	vertices.clear();

	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
		generateMoistAdiabat(CCLProfiles[profileIndex].x, CCLProfiles[profileIndex].y, vertices, 1, P0, &moistAdiabatEdgeCount[1], true, CURVE_DELTA_P, &moistAdiabatProfiles[profileIndex]);

		glm::vec2 tmp = findIntersection(moistAdiabatProfiles[profileIndex], ambientCurve, true);
		ELProfiles[profileIndex] = getDenormalizedCoords(tmp);
	}

	glNamedBufferData(moistAdiabatsVBO[1], sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);


	uploadMainParameterPointsToBuffer();

}

void STLPDiagram::recalculateProfileDelta() {
	profileDelta = convectiveTempRange / (float)numProfiles;
}



glm::vec2 STLPDiagram::getNormalizedCoords(glm::vec2 coords) {
	glm::vec2 res;
	res.y = getNormalizedPres(coords.y);
	res.x = getNormalizedTemp(coords.x, res.y);
	return res;
}

glm::vec2 STLPDiagram::getDenormalizedCoords(glm::vec2 coords) {
	glm::vec2 res;
	res.x = getDenormalizedTemp(coords.x, coords.y);
	res.y = getDenormalizedPres(coords.y);
	return res;
}

glm::vec2 STLPDiagram::getNormalizedCoords(float T, float P) {
	return getNormalizedCoords(glm::vec2(T, P));
}

glm::vec2 STLPDiagram::getDenormalizedCoords(float x, float y) {
	return getDenormalizedCoords(glm::vec2(x, y));
}

float STLPDiagram::getNormalizedTemp(float T, float y) {
	return (T - MIN_TEMP) / (MAX_TEMP - MIN_TEMP) + (1.0f - y);
}

float STLPDiagram::getNormalizedPres(float P) {
	return ((log10f(P) - log10f(MIN_P)) / (log10f(P0) - log10f(MIN_P)));
}

float STLPDiagram::getDenormalizedTemp(float x, float y) {
	return (x + y - 1.0f) * (MAX_TEMP - MIN_TEMP) + MIN_TEMP;
}

float STLPDiagram::getDenormalizedPres(float y) {
	return powf(10.0f, y * (log10f(P0) - log10f(MIN_P)) + log10f(MIN_P));
}




void STLPDiagram::initFreetype() {
	textRend = new TextRenderer();

}

void STLPDiagram::draw() {

	int counter;
	glLineWidth(1.0f);

	curveShader->use();

	curveShader->setBool("u_CropBounds", (bool)cropBounds);

	if (showIsobars) {
		curveShader->setColor(isobarsColor);
		glBindVertexArray(isobarsVAO);
		glDrawArrays(GL_LINES, 0, numIsobars * 2);
	}


	glPointSize(8.0f);
	curveShader->setVec3("u_Color", glm::vec3(0.0f));
	glBindVertexArray(temperaturePointsVAO);
	glDrawArrays(GL_LINES, 0, temperaturePointsCount * 2);


	if (showIsotherms) {
		glPointSize(8.0f);
		curveShader->setColor(isothermsColor);
		glBindVertexArray(isothermsVAO);
		glDrawArrays(GL_LINES, 0, isothermsCount * 2);
	}


	if (showAmbientCurve) {
		curveShader->setColor(ambientCurveColor);
		glBindVertexArray(ambientTemperatureVAO);
		glDrawArrays(GL_LINE_STRIP, 0, soundingData.size());
	}


	if (showDewpointCurve) {
		curveShader->setColor(dewpointCurveColor);
		glBindVertexArray(dewTemperatureVAO);
		glDrawArrays(GL_LINE_STRIP, 0, soundingData.size());
	}


	if (showIsohumes) {
		curveShader->setColor(isohumesColor);
		glBindVertexArray(isohumesVAO);
		glDrawArrays(GL_LINES, 0, mixingCCL.vertices.size());
	}

	for (int j = 0; j < 2; j++) {
		if (showDryAdiabats[j]) {
			curveShader->setColor(dryAdiabatsColor[j]);
			glBindVertexArray(dryAdiabatsVAO[j]);

			glLineWidth(0.01f);

			counter = 0;
			for (int i = 0; i < numDryAdiabats[j]; i++) {
				//glDrawArrays(GL_LINE_STRIP, (numIsobars-1) * i, numIsobars - 1);

				glDrawArrays(GL_LINE_STRIP, counter, dryAdiabatEdgeCount[j][i]);
				counter += dryAdiabatEdgeCount[j][i];
			}
		}
	}

	for (int j = 0; j < 2; j++) {
		if (showMoistAdiabats[j]) {
			glPointSize(2.0f);
			curveShader->setColor(moistAdiabatsColor[j]);
			glBindVertexArray(moistAdiabatsVAO[j]);
			//glDrawArrays(GL_LINE_STRIP, 0, 1000000);
			//glDrawArrays(GL_POINTS, 0, 100000);

			counter = 0;
			for (int i = 0; i < numMoistAdiabats[j]; i++) {

				glDrawArrays(GL_LINE_STRIP, counter, moistAdiabatEdgeCount[j][i]);
				counter += moistAdiabatEdgeCount[j][i];
			}
		}
	}


	xaxis.draw(curveShader);
	yaxis.draw(curveShader);
	groundIsobar.draw(curveShader);



	glPointSize(6.0f);
	curveShader->setColor(glm::vec3(0.0f));
	glBindVertexArray(mainParameterPointsVAO);
	glDrawArrays(GL_POINTS, 0, mainParameterPoints.size());



	
	glPointSize(3.0f);
	singleColorShaderVBO->use();
	glBindVertexArray(visPointsVAO);
	glDrawArrays(GL_POINTS, 0, visualizationPoints.size() / 2);
	



	CHECK_GL_ERRORS();


}

void STLPDiagram::drawText() {

	float scaleZoomModifier = vars->diagramProjectionOffset + 0.5f;
	float textScale = 0.0006f * powf(scaleZoomModifier, 0.8f);
	textScale = glm::min(textScale, maxTextScale);

	int i = 0;
	for (int temp = MIN_TEMP; temp <= MAX_TEMP; temp += 10) {
		if (temperaturePoints[i].x < 0.0f || temperaturePoints[i].x > 1.0f) {
			i++;
			continue;
		}
		textRend->renderText(to_string(temp), temperaturePoints[i].x, temperaturePoints[i].y + 0.02f, textScale);
		i++;
	}

	if (CCLFound) {
		textRend->renderText("CCL", CCLNormalized.x, CCLNormalized.y, textScale);
	}
	if (TcFound) {
		textRend->renderText("Tc", TcNormalized.x, TcNormalized.y, textScale);
	}
	if (ELFound) {
		textRend->renderText("EL", ELNormalized.x, ELNormalized.y, textScale);
	}
	if (LCLFound) {
		textRend->renderText("LCL", LCLNormalized.x, LCLNormalized.y, textScale);
	}
	if (LFCFound) {
		textRend->renderText("LFC", LFCNormalized.x, LFCNormalized.y, textScale);
	}
	if (orographicELFound) {
		textRend->renderText("OEL", orographicELNormalized.x, orographicELNormalized.y, textScale);
	}

	textRend->renderText("Ground Level at " + to_string((int)getAltitudeFromPressure(P0)) + "[m]", 1.0f + 0.01f * scaleZoomModifier, getNormalizedPres(P0), textScale);

	for (i = 1000.0f; i >= MIN_P; i -= 100) {
		textRend->renderText(to_string(i), 0.0f - 0.04f - 0.02f * scaleZoomModifier, getNormalizedPres(i), textScale);
		textRend->renderText(to_string((int)getAltitudeFromPressure(i)) + "[m]", 0.0f + 0.01f * scaleZoomModifier, getNormalizedPres(i), textScale);
	}

	textRend->renderText("Temperature [degree C]", 0.35f, ymax + 0.08f, textScale);
	textRend->renderText("P [hPa]", -0.15f, 0.5f, textScale);

	textRend->renderText("SkewT/LogP (" + soundingFilename + ")", 0.2f, -0.05f, textScale);


}

void STLPDiagram::drawOverlayDiagram(GLuint textureId) {
	//GLint current_program_id = 0;
	//glGetIntegerv(GL_CURRENT_PROGRAM, &current_program_id);
	GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);

	glDisable(GL_DEPTH_TEST);
	overlayDiagramShader->use();
	glActiveTexture(GL_TEXTURE0);

	if (textureId == -1) {
		glBindTextureUnit(0, diagramTexture);
	} else {
		glBindTextureUnit(0, textureId);
	}
	overlayDiagramShader->setInt("u_Texture", 0);

	glBindVertexArray(overlayDiagramVAO);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	if (depth_test_enabled) {
		glEnable(GL_DEPTH_TEST);
	}

}

void STLPDiagram::refreshOverlayDiagram(GLuint viewportWidth, GLuint viewportHeight, GLuint viewport_x, GLuint viewport_y) {

	glm::vec4 vp;
	//glGetFloatv(GL_VIEWPORT, &vp.x);
	vp.x = viewport_x;
	vp.y = viewport_y;
	vp.z = viewportWidth;
	vp.w = viewportHeight;
	float width = overlayDiagramResolution;
	float height = overlayDiagramResolution;
	float x = overlayDiagramX;
	float y = overlayDiagramY;

	const GLfloat normalized_coords_with_tex_coords[] = {
		(x - vp.x) / (vp.z - vp.x)*2.0f - 1.0f,          (y - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 0.0f, 0.0f,
		(x + width - vp.x) / (vp.z - vp.x)*2.0f - 1.0f,          (y - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 1.0f, 0.0f,
		(x + width - vp.x) / (vp.z - vp.x)*2.0f - 1.0f, (y + height - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 1.0f, 1.0f,
		(x - vp.x) / (vp.z - vp.x)*2.0f - 1.0f, (y + height - vp.y) / (vp.w - vp.y)*2.0f - 1.0f, 0.0f, 1.0f,
	};

	glBindBuffer(GL_ARRAY_BUFFER, overlayDiagramVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(normalized_coords_with_tex_coords), &normalized_coords_with_tex_coords, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);


}

void STLPDiagram::setVisualizationPoint(glm::vec3 position, glm::vec3 color, int index, bool positionIsNormalized) {

	if (!positionIsNormalized) {
		position.y = getNormalizedPres(position.y);
		position.x = getNormalizedTemp(position.x, position.y);
	}

	index *= 2;
	if (index + 1 < visualizationPoints.size()) {

		visualizationPoints[index] = position;
		visualizationPoints[index + 1] = color;

		//glBindBuffer(GL_ARRAY_BUFFER, visPointsVBO);
		//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * visualizationPoints.size(), &visualizationPoints[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, visPointsVBO);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * index, 2 * sizeof(glm::vec3), &visualizationPoints[index]);


		//glNamedBufferSubData(visPointsVBO, sizeof(glm::vec3) * index, 2 * sizeof(glm::vec3), &visualizationPoints[index]);

	}

}

void STLPDiagram::findClosestSoundingPoint(glm::vec2 queryPoint) {

	float minDist = glm::distance(ambientCurve.vertices[0], queryPoint);
	int pointIndex = 0;
	int curveIndex = 0;

	for (int i = 0; i < ambientCurve.vertices.size(); i++) {
		float currDist = glm::distance(ambientCurve.vertices[i], queryPoint);
		if (currDist < minDist) {
			minDist = currDist;
			pointIndex = i;
		}
	}
	for (int i = 0; i < dewpointCurve.vertices.size(); i++) {
		float currDist = glm::distance(dewpointCurve.vertices[i], queryPoint);
		if (currDist < minDist) {
			minDist = currDist;
			pointIndex = i;
			curveIndex = 1;
		}
	}

	if (curveIndex == 0) {
		setVisualizationPoint(glm::vec3(ambientCurve.vertices[pointIndex], 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 3, true);
		selectedPoint = pair<Curve *, int>(&ambientCurve, pointIndex);

	} else {
		setVisualizationPoint(glm::vec3(dewpointCurve.vertices[pointIndex], 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 3, true);
		selectedPoint = pair<Curve *, int>(&dewpointCurve, pointIndex);

	}

}

void STLPDiagram::moveSelectedPoint(glm::vec2 mouseCoords) {
	Curve *curvePtr = selectedPoint.first;
	int pointIndex = selectedPoint.second;
	curvePtr->vertices[pointIndex].x = mouseCoords.x;
	if (curvePtr == &ambientCurve) {
		glBindBuffer(GL_ARRAY_BUFFER, ambientTemperatureVBO);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * pointIndex, sizeof(glm::vec2), &curvePtr->vertices[pointIndex]);
	} else if (curvePtr == &dewpointCurve) {
		glBindBuffer(GL_ARRAY_BUFFER, dewTemperatureVBO);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * pointIndex, sizeof(glm::vec2), &curvePtr->vertices[pointIndex]);
	}
	setVisualizationPoint(glm::vec3(curvePtr->vertices[pointIndex], 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 3, true);
	diagramChanged = true;


}

void STLPDiagram::constructDiagramCurvesToolbar(nk_context *ctx, UserInterface *ui) {

	nk_checkbox_label(ctx, "Show Isobars", &showIsobars);
	if (showIsobars) {
		ui->nk_property_color_rgb(ctx, isobarsColor);
	}

	nk_checkbox_label(ctx, "Show Isotherms", &showIsotherms);
	if (showIsotherms) {
		ui->nk_property_color_rgb(ctx, isothermsColor);
	}

	nk_checkbox_label(ctx, "Show Isohumes", &showIsohumes);
	if (showIsohumes) {
		ui->nk_property_color_rgb(ctx, isohumesColor);
	}

	nk_checkbox_label(ctx, "Show Dry Adiabats (General)", &showDryAdiabats[0]);
	if (showDryAdiabats[0]) {
		ui->nk_property_color_rgb(ctx, dryAdiabatsColor[0]);
	}

	nk_checkbox_label(ctx, "Show Moist Adiabats (General)", &showMoistAdiabats[0]);
	if (showMoistAdiabats[0]) {
		ui->nk_property_color_rgb(ctx, moistAdiabatsColor[0]);
	}

	nk_checkbox_label(ctx, "Show Dry Adiabat Profiles", &showDryAdiabats[1]);
	if (showDryAdiabats[1]) {
		ui->nk_property_color_rgb(ctx, dryAdiabatsColor[1]);
	}

	nk_checkbox_label(ctx, "Show Moist Adiabat Profiles", &showMoistAdiabats[1]);
	if (showMoistAdiabats[1]) {
		ui->nk_property_color_rgb(ctx, moistAdiabatsColor[1]);
	}

	nk_checkbox_label(ctx, "Show Dewpoint Curve", &showDewpointCurve);
	if (showDewpointCurve) {
		ui->nk_property_color_rgb(ctx, dewpointCurveColor);
	}

	nk_checkbox_label(ctx, "Show Ambient Curve", &showAmbientCurve);
	if (showAmbientCurve) {
		ui->nk_property_color_rgb(ctx, ambientCurveColor);
	}

	nk_checkbox_label(ctx, "Crop Bounds", &cropBounds);



}

void STLPDiagram::setTmpSoundingFilename(string tmpSoundingFilename) {
	this->tmpSoundingFilename = tmpSoundingFilename;
	soundingFilenameChanged = (soundingFilename != this->tmpSoundingFilename);
}

string STLPDiagram::getTmpSoundingFilename() {
	return tmpSoundingFilename;
}

bool STLPDiagram::wasSoundingFilenameChanged() {
	return soundingFilenameChanged;
}

bool STLPDiagram::wasDiagramChanged() {
	return diagramChanged;
}

void STLPDiagram::setDiagramChanged(bool diagramChanged) {
	this->diagramChanged = diagramChanged;
}



glm::vec2 STLPDiagram::getWindDeltasFromAltitude(float altitude) {
	// naive solution -> linearly find the correct altitude pair

	for (int i = 0; i < windData.size() - 1; i++) {
		if (altitude >= windData[i].y && altitude < windData[i + 1].y) {
			if (altitude == windData[i].y) {
				return glm::vec2(windData[i].delta_x, windData[i].delta_z);
			}
			float t = (altitude - windData[i + 1].y) / (windData[i].y - windData[i + 1].y);
			glm::vec2 res;
			res.x = t * windData[i].delta_x + (1.0f - t) * windData[i + 1].delta_x;
			res.y = t * windData[i].delta_z + (1.0f - t) * windData[i + 1].delta_z;
			return res / 100.0f;
		}
	}

	return glm::vec2(0.0f);
}

void STLPDiagram::getWindDeltasForLattice(int latticeHeight, std::vector<glm::vec3>& outWindDeltas) {

	outWindDeltas.clear();
	float step = (float)(soundingData.size() - 1) / (float)latticeHeight;

	float idx = 0.0f;
	for (int i = 0; i < latticeHeight; i++) {

		int idxBottom = (int)idx;
		int idxTop = idxBottom + 1;
		glm::vec3 deltaBottom = glm::vec3(windData[idxBottom].delta_x, 0.0f, windData[idxBottom].delta_z);
		glm::vec3 deltaTop = glm::vec3(windData[idxTop].delta_x, 0.0f, windData[idxTop].delta_z);

		float t = idx - idxBottom;

		glm::vec3 delta = t * deltaTop + (1.0f - t) * deltaBottom;
		outWindDeltas.push_back(glm::normalize(delta));

		idx += step;

	}

}





void STLPDiagram::initOverlayDiagram() {

	// Overlay DIAGRAM

	glm::vec4 vp;
	glGetFloatv(GL_VIEWPORT, &vp.x);

	glGenVertexArrays(1, &overlayDiagramVAO);
	glGenBuffers(1, &overlayDiagramVBO);
	glBindVertexArray(overlayDiagramVAO);
	glBindBuffer(GL_ARRAY_BUFFER, overlayDiagramVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glBindVertexArray(0);

	refreshOverlayDiagram(vp.z, vp.w, vp.x, vp.y);



	// TEXTURE AND FRAMEBUFFER

	glGenTextures(1, &diagramTexture);
	glBindTexture(GL_TEXTURE_2D, diagramTexture);

	//glTextureParameteri(diagramTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);


	float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, textureResolution, textureResolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	glGenFramebuffers(1, &diagramFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, diagramFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, diagramTexture, 0);


	glGenTextures(1, &diagramMultisampledTexture);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, diagramMultisampledTexture);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 12, GL_RGBA16F, textureResolution, textureResolution, false);

	glGenFramebuffers(1, &diagramMultisampledFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, diagramMultisampledFramebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, diagramMultisampledTexture, 0);





}

void STLPDiagram::generateTemperatureNotches() {
	vector<glm::vec2> vertices;

	float T;

	temperaturePointsCount = 0;
	temperaturePoints.clear();
	for (int i = MIN_TEMP; i <= MAX_TEMP; i += 10) {
		T = getNormalizedTemp(i, ymax);
		temperaturePoints.push_back(glm::vec2(T, ymax));
		vertices.push_back(glm::vec2(T, ymax + temperatureNotchSize / 2.0f));
		vertices.push_back(glm::vec2(T, ymax - temperatureNotchSize / 2.0f));
		temperaturePointsCount++;
	}

	glBindBuffer(GL_ARRAY_BUFFER, temperaturePointsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

}

void STLPDiagram::uploadMainParameterPointsToBuffer() {

	if (!mainParameterPoints.empty()) {
		mainParameterPoints.clear();
	}


	if (CCLFound) {
		mainParameterPoints.push_back(glm::vec3(CCLNormalized, 0.0f));
	}
	if (TcFound) {
		mainParameterPoints.push_back(glm::vec3(TcNormalized, 0.0f));
	}
	if (ELFound) {
		mainParameterPoints.push_back(glm::vec3(ELNormalized, 0.0f));
	}
	if (LCLFound) {
		mainParameterPoints.push_back(glm::vec3(LCLNormalized, 0.0f));
	}
	if (LFCFound) {
		mainParameterPoints.push_back(glm::vec3(LFCNormalized, 0.0f));
	}
	if (orographicELFound) {
		mainParameterPoints.push_back(glm::vec3(orographicELNormalized, 0.0f));
	}

	if (!mainParameterPoints.empty()) {
		glBindBuffer(GL_ARRAY_BUFFER, mainParameterPointsVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * mainParameterPoints.size(), &mainParameterPoints[0], GL_STATIC_DRAW);
	}
}















//void STLPDiagram::initBuffersOld() {
//
//
//	// Initialize main variables
//
//	float xmin = 0.0f;
//	float xmax = 1.0f;
//
//	float ymin = getNormalizedPres(MIN_P);
//	float ymax = getNormalizedPres(MAX_P);
//
//	float P0 = soundingData[0].data[PRES];
//	float P;
//	float T;
//
//	xaxis.vertices.push_back(glm::vec2(xmin, ymax));
//	xaxis.vertices.push_back(glm::vec2(xmax, ymax));
//	yaxis.vertices.push_back(glm::vec2(xmin, ymin));
//	yaxis.vertices.push_back(glm::vec2(xmin, ymax));
//
//	xaxis.initBuffers();
//	yaxis.initBuffers();
//
//	TcProfiles.reserve(numProfiles);
//	CCLProfiles.reserve(numProfiles);
//	ELProfiles.reserve(numProfiles);
//	dryAdiabatProfiles.reserve(numProfiles);
//	moistAdiabatProfiles.reserve(numProfiles);
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// ISOBARS
//	///////////////////////////////////////////////////////////////////////////////////////
//
//	vector<glm::vec2> vertices;
//
//	numIsobars = 0;
//	for (P = MAX_P; P >= MIN_P; P -= CURVE_DELTA_P) {
//		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
//		//P = soundingData[profileIndex].data[PRES];
//		float y = getNormalizedPres(P);
//		vertices.push_back(glm::vec2(xmin, y));
//		vertices.push_back(glm::vec2(xmax, y));
//		numIsobars++;
//	}
//	float y;
//	P = soundingData[0].data[PRES];
//	y = getNormalizedPres(P);
//	vertices.push_back(glm::vec2(xmin, y));
//	vertices.push_back(glm::vec2(xmax, y));
//	numIsobars++;
//
//
//	glGenVertexArrays(1, &isobarsVAO);
//	glBindVertexArray(isobarsVAO);
//	glGenBuffers(1, &isobarsVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, isobarsVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// TEMPERATURE POINTS
//	///////////////////////////////////////////////////////////////////////////////////////
//
//	//vertices.clear();
//	temperaturePointsCount = 0;
//	for (int i = MIN_TEMP; i <= MAX_TEMP; i += 10) {
//		T = getNormalizedTemp(i, ymax);
//		temperaturePoints.push_back(glm::vec2(T, ymax));
//		temperaturePointsCount++;
//	}
//
//	glGenVertexArrays(1, &temperaturePointsVAO);
//	glBindVertexArray(temperaturePointsVAO);
//	glGenBuffers(1, &temperaturePointsVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, temperaturePointsVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * temperaturePoints.size(), &temperaturePoints[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// ISOTHERMS
//	///////////////////////////////////////////////////////////////////////////////////////
//
//	vertices.clear();
//
//	float x;
//	y;
//
//	isothermsCount = 0;
//	for (int i = MIN_TEMP - 80.0f; i <= MAX_TEMP; i += 10) {
//
//		y = ymax;
//		x = getNormalizedTemp(i, y);
//		vertices.push_back(glm::vec2(x, y));
//
//		y = ymin;
//		x = getNormalizedTemp(i, y);
//		vertices.push_back(glm::vec2(x, y));
//
//		isothermsCount++;
//	}
//
//
//	glGenVertexArrays(1, &isothermsVAO);
//	glBindVertexArray(isothermsVAO);
//	glGenBuffers(1, &isothermsVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, isothermsVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// AMBIENT TEMPERATURE PIECE-WISE LINEAR CURVE
//	///////////////////////////////////////////////////////////////////////////////////////
//	vertices.clear();
//	for (int i = 0; i < soundingData.size(); i++) {
//
//		float P = soundingData[i].data[PRES];
//		float T = soundingData[i].data[TEMP];
//
//		y = getNormalizedPres(P);
//		x = getNormalizedTemp(T, y);
//
//		vertices.push_back(glm::vec2(x, y));
//	}
//
//	ambientCurve.vertices = vertices;
//
//	glGenVertexArrays(1, &ambientTemperatureVAO);
//	glBindVertexArray(ambientTemperatureVAO);
//	glGenBuffers(1, &ambientTemperatureVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, ambientTemperatureVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// DEWPOINT TEMPERATURE PIECE-WISE LINEAR CURVE
//	///////////////////////////////////////////////////////////////////////////////////////
//	vertices.clear();
//	for (int i = 0; i < soundingData.size(); i++) {
//
//		float P = soundingData[i].data[PRES];
//		float T = soundingData[i].data[DWPT];
//
//		y = getNormalizedPres(P);
//		x = getNormalizedTemp(T, y);
//
//		vertices.push_back(glm::vec2(x, y));
//	}
//	dewpointCurve.vertices = vertices;
//
//
//	glGenVertexArrays(1, &dewTemperatureVAO);
//	glBindVertexArray(dewTemperatureVAO);
//	glGenBuffers(1, &dewTemperatureVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, dewTemperatureVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// ISOHUMES (MIXING RATIO LINES)
//	///////////////////////////////////////////////////////////////////////////////////////
//	cout << "////////////////////////////////////////////////////" << endl;
//	cout << "// ISOHUMES (MIXING RATIO LINES)" << endl;
//	cout << "////////////////////////////////////////////////////" << endl;
//
//	vertices.clear();
//
//	float Rd = 287.05307f;	// gas constant for dry air [J kg^-1 K^-1]
//	float Rm = 461.5f;		// gas constant for moist air [J kg^-1 K^-1]
//
//							// w(T,P) = (eps * e(T)) / (P - e(T))
//							// where
//							//		eps = Rd / Rm
//							//		e(T) ... saturation vapor pressure (can be approx'd by August-Roche-Magnus formula
//							//		e(T) =(approx)= C exp( (A*T) / (T + B))
//							//		where
//							//				A = 17.625
//							//				B = 243.04
//							//				C = 610.94
//
//	float A = 17.625f;
//	float B = 243.04f;
//	float C = 610.94f;
//
//	// given that w(T,P) is const., let W = w(T,P), we can express T in terms of P (see Equation 3.13)
//
//	// to determine the mixing ratio line that passes through (T,P), we calculate the value of the temperature
//	// T(P + delta) where delta is a small integer
//	// -> the points (T,P) nad (T(P + delta), P + delta) define a mixing ratio line, whose points all have the same mixing ratio
//
//
//	// Compute CCL using a mixing ratio line
//	float w0 = soundingData[0].data[MIXR];
//	T = soundingData[0].data[DWPT];
//	P = soundingData[0].data[PRES];
//
//
//	float eps = Rd / Rm;
//	//float satVP = C * exp((A * T) / (T + B));	// saturation vapor pressure: e(T)
//	float satVP = e_s_degC(T);
//	//float W = (eps * satVP) / (P - satVP);
//	float W = getMixingRatioOfWaterVapor(T, P);
//
//	cout << " -> Computed W = " << W << endl;
//
//	float deltaP = 20.0f;
//
//	while (P >= MIN_P) {
//
//
//		float fracPart = log((W * P) / (C * (W + eps)));
//		float computedT = (B * fracPart) / (A - fracPart);
//
//		cout << " -> Computed T = " << computedT << endl;
//
//		y = getNormalizedPres(P);
//		x = getNormalizedTemp(T, y);
//
//		if (x < xmin || x > xmax || y < 0.0f || y > 1.0f) {
//			break;
//		}
//
//		vertices.push_back(glm::vec2(x, y));
//
//
//		float delta = 10.0f; // should be a small integer - produces dashed line (logarithmic)
//							 //float offsetP = P - delta;
//		float offsetP = P - deltaP; // produces continuous line
//		fracPart = log((W * offsetP) / (C * (W + eps)));
//		computedT = (B * fracPart) / (A - fracPart);
//		cout << " -> Second computed T = " << computedT << endl;
//
//
//		y = getNormalizedPres(offsetP);
//		x = getNormalizedTemp(T, y);
//		vertices.push_back(glm::vec2(x, y));
//
//		P -= deltaP;
//
//	}
//
//	mixingCCL.vertices = vertices;
//
//
//	CCLNormalized = findIntersection(mixingCCL, ambientCurve);
//	cout << "CCL (normalized) = " << CCLNormalized.x << ", " << CCLNormalized.y << endl;
//
//	CCL = getDenormalizedCoords(CCLNormalized);
//
//	cout << "CCL = " << CCL.x << ", " << CCL.y << endl;
//
//
//
//	glGenVertexArrays(1, &CCLVAO);
//	glBindVertexArray(CCLVAO);
//	glGenBuffers(1, &CCLVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, CCLVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2), &CCLNormalized, GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//	glGenVertexArrays(1, &isohumesVAO);
//	glBindVertexArray(isohumesVAO);
//	glGenBuffers(1, &isohumesVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, isohumesVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// DRY ADIABATS
//	///////////////////////////////////////////////////////////////////////////////////////
//	cout << "////////////////////////////////////////////////////" << endl;
//	cout << "// DRY ADIABATS" << endl;
//	cout << "////////////////////////////////////////////////////" << endl;
//
//	vertices.clear();
//
//	/*
//	Dry adiabats feature the thermodynamic behaviour of unsaturated air parcels moving upwards (or downwards).
//	They represent the dry adiabatic lapse rate (DALR).
//	This thermodynamic behaviour is valid for all air parcels moving between the ground and the convective
//	condensation level (CCLNormalized).
//
//	T(P) = theta / ((P0 / P)^(Rd / cp))
//	where
//	P0 is the initial value of pressure (profileIndex.e. ground pressure)
//	cp is the heat capacity of dry air at constant pressure
//	(cv is the heat capacity of dry air at constant volume)
//	Rd is the gas constant for dry air [J kg^-1 K^-1]
//	k = Rd / cp = (cp - cv) / cp =(approx)= 0.286
//	*/
//
//	float k = 0.286f; // Rd / cp
//
//	numDryAdiabats = 0;
//	int counter;
//
//	for (float theta = MIN_TEMP; theta <= MAX_TEMP * 5; theta += 10.0f) {
//		counter = 0;
//
//		for (P = MAX_P; P >= MIN_P; P -= CURVE_DELTA_P) {
//			//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
//			//float P = soundingData[profileIndex].data[PRES];
//
//			float T = (theta + 273.16f) / pow((P0 / P), k);
//			T -= 273.16f;
//
//			y = getNormalizedPres(P);
//			x = getNormalizedTemp(T, y);
//
//			vertices.push_back(glm::vec2(x, y));
//			counter++;
//		}
//		numDryAdiabats++;
//		dryAdiabatEdgeCount.push_back(counter);
//	}
//
//
//
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	// TESTING Tc computation - special dry adiabat (and its x axis intersection)
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	{
//		P0 = soundingData[0].data[PRES];
//
//		float theta = (CCL.x + 273.15f) * powf(P0 / CCL.y, k);
//		theta -= 273.15f;
//		cout << "CCL theta = " << theta << endl;
//		cout << "Tc Dry adiabat: " << endl;
//
//		counter = 0;
//
//
//		for (int i = 0; i < soundingData.size(); i++) {
//
//			float P = soundingData[i].data[PRES];
//
//			float T = (theta + 273.15f) * pow((P / P0), k); // do not forget to use Kelvin
//			T -= 273.15f; // convert back to Celsius
//
//			y = getNormalizedPres(P);
//			x = getNormalizedTemp(T, y);
//
//			TcDryAdiabat.vertices.push_back(glm::vec2(x, y));
//			vertices.push_back(glm::vec2(x, y));
//			counter++;
//
//			cout << " | " << x << ", " << y << endl;
//		}
//		cout << endl;
//		numDryAdiabats++;
//		dryAdiabatEdgeCount.push_back(counter);
//
//	}
//
//
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	// TESTING LCL computation - special dry adiabat (starts in ground ambient temp.)
//	///////////////////////////////////////////////////////////////////////////////////////////////
//
//	{
//		P0 = soundingData[0].data[PRES];
//
//		float theta = (soundingData[0].data[TEMP] + 273.15f)/* * powf(P0 / P0, k)*/;
//		theta -= 273.15f;
//		cout << "LCL Dry adiabat theta = " << theta << endl;
//
//		for (int i = 0; i < soundingData.size(); i++) {
//
//			float P = soundingData[i].data[PRES];
//
//			float T = (theta + 273.15f) * pow((P / P0), k); // do not forget to use Kelvin
//			T -= 273.15f; // convert back to Celsius
//
//			y = getNormalizedPres(P);
//			x = getNormalizedTemp(T, y);
//
//			LCLDryAdiabatCurve.vertices.push_back(glm::vec2(x, y));
//			//vertices.push_back(glm::vec2(x, y));
//			cout << " | " << x << ", " << y << endl;
//		}
//		cout << endl;
//		//numDryAdiabats++;
//	}
//
//
//	//TcNormalized = findIntersection(xaxis, TcDryAdiabat);
//	TcNormalized = TcDryAdiabat.vertices[0]; // no need for intersection search here
//	cout << "TcNormalized: " << TcNormalized.x << ", " << TcNormalized.y << endl;
//	Tc = getDenormalizedCoords(TcNormalized);
//	cout << "Tc: " << Tc.x << ", " << Tc.y << endl;
//
//	// Check correctness by computing thetaCCL == Tc
//	float thetaCCL = (CCL.x + 273.15f) * pow((P0 / CCL.y), k);
//	thetaCCL -= 273.15f;
//	cout << "THETA CCL = " << thetaCCL << endl;
//
//	LCLNormalized = findIntersection(LCLDryAdiabatCurve, mixingCCL);
//	LCL = getDenormalizedCoords(LCLNormalized);
//
//
//	// Testing out profiles
//	for (int i = 0; i < numProfiles; i++) {
//		TcProfiles.push_back(Tc + glm::vec2((i + 1) * profileDelta, 0.0f));
//		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(TcProfiles.back()), -2.0f)); // point
//		float tint = (i + 1) * profileDelta;
//		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
//		visualizationPoints.push_back(glm::vec3(tint, 0.0f, 0.0f)); // color	
//	}
//
//	cout << "NUMBER OF PROFILES = " << numProfiles << ", profileDelta = " << profileDelta << endl;
//	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
//
//		dryAdiabatProfiles.push_back(Curve());
//
//		counter = 0;
//
//		P0 = soundingData[0].data[PRES];
//
//		float theta = (TcProfiles[profileIndex].x + 273.15f)/* * powf(P0 / P0, k)*/;
//		theta -= 273.15f;
//
//		for (int i = 0; i < soundingData.size(); i++) {
//
//			float P = soundingData[i].data[PRES];
//
//			float T = (theta + 273.15f) * pow((P / P0), k); // do not forget to use Kelvin
//			T -= 273.15f; // convert back to Celsius
//
//			y = getNormalizedPres(P);
//			x = getNormalizedTemp(T, y);
//
//			dryAdiabatProfiles[profileIndex].vertices.push_back(glm::vec2(x, y));
//			vertices.push_back(glm::vec2(x, y));
//			counter++;
//		}
//		numDryAdiabats++;
//		dryAdiabatEdgeCount.push_back(counter);
//
//		CCLProfiles.push_back(getDenormalizedCoords(findIntersection(dryAdiabatProfiles[profileIndex], ambientCurve)));
//
//		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(CCLProfiles.back()), -2.0f)); // point
//		float tint = (profileIndex + 1) * profileDelta;
//		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
//		visualizationPoints.push_back(glm::vec3(tint, 0.0f, 1.0f)); // color	
//
//	}
//
//
//
//	glGenVertexArrays(1, &TcVAO);
//	glBindVertexArray(TcVAO);
//	glGenBuffers(1, &TcVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, TcVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2), &TcNormalized, GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//	glGenVertexArrays(1, &dryAdiabatsVAO);
//	glBindVertexArray(dryAdiabatsVAO);
//	glGenBuffers(1, &dryAdiabatsVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, dryAdiabatsVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//	glBindVertexArray(0);
//
//
//
//
//	///////////////////////////////////////////////////////////////////////////////////////
//	// MOIST ADIABATS
//	///////////////////////////////////////////////////////////////////////////////////////
//	cout << "////////////////////////////////////////////////////" << endl;
//	cout << "// MOIST ADIABATS" << endl;
//	cout << "////////////////////////////////////////////////////" << endl;
//
//	vertices.clear();
//
//
//	float a = -6.14342f * 0.00001f;
//	float b = 1.58927 * 0.001f;
//	float c = -2.36418f;
//	float d = 2500.79f;
//
//	float g = -9.81f;
//	numMoistAdiabats = 0;
//
//	// Lv(T) = (aT^3 + bT^2 + cT + d) * 1000
//	// Lv(T)	... latent heat of vaporisation/condensation
//
//	//for (float T = MIN_TEMP; T <= MAX_TEMP; T++) {
//	T = 30.0f; // for example - create first testing curvePtr
//			   //float Lv = (a*T*T*T + b*T*T + c*T + d) * 1000.0f;
//
//	float T_P0;
//	float T_P1;
//	float P1;
//	float currP;
//
//#define MOIST_ADIABAT_OPTION 5
//
//#if MOIST_ADIABAT_OPTION == 0
//	T_P0 = T;
//	// LooooooooooooooP
//	deltaP = 1.0f;
//	for (float p = P0; p >= MIN_P; p -= deltaP) {
//		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
//		//P = soundingData[profileIndex].data[PRES];
//
//		//T_P1 = ???
//		//P1 = P;
//		P1 = p;
//
//		toKelvin(T_P0);
//		toKelvin(T);
//
//
//		// integral pres P0 az P1
//		float integratedVal = 0.0f;
//		//for (int profileIndex = P0 + 0.1f; profileIndex < P1; profileIndex += 0.1f) {
//		//integratedVal += computePseudoadiabaticLapseRate(T, profileIndex);
//		//integratedVal /= computeRho(T, profileIndex) * (-9.81f);
//		//integratedVal += getMoistAdiabatIntegralVal(T_P0, profileIndex);
//		//}
//		//integratedVal += (getMoistAdiabatIntegralVal(T_P0, P0) + getMoistAdiabatIntegralVal(T_P0, P1)) / 2.0f;
//		//integratedVal *= (P1 - P0) / CURVE_DELTA_P;
//
//		//T_P1 = T_P0 + integratedVal;
//
//		P0 *= 100.0f;
//		P1 *= 100.0f;
//
//		integratedVal = (P1 - P0) * getMoistAdiabatIntegralVal(T_P0, P1);
//		T_P1 = T_P0 + integratedVal * 100.0f;
//
//		//cout << "Integrated val = " << integratedVal << endl;
//
//		P0 /= 100.0f;
//		P1 /= 100.0f;
//
//		toCelsius(T);
//		toCelsius(T_P0);
//		toCelsius(T_P1);
//
//		y = getNormalizedPres(P0);
//		x = getNormalizedTemp(T_P0, y);
//		//x = (T_P0 - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
//		vertices.push_back(glm::vec2(x, y));
//
//		y = getNormalizedPres(P1);
//		x = getNormalizedTemp(T_P1, y);
//		//x = (T_P1 - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
//
//		vertices.push_back(glm::vec2(x, y));
//
//
//		// jump to next
//		P0 = P1;
//		T_P0 = T_P1;
//	}
//#elif MOIST_ADIABAT_OPTION == 1 // Taken from existing code
//	/////////////////////////////
//	T_P0 = T;
//	float ept = computeEquivalentTheta(getKelvin(T_P0), getKelvin(T_P0), 1000.0f);
//	cout << "EPT = " << ept << endl;
//	//P0 = 1000.0f;
//
//	T_P0 = getSaturatedAirTemperature(ept, P0);
//	////////////////////////////////
//	for (int i = 0; i < soundingData.size(); i++) {
//		P = soundingData[i].data[PRES];
//		P1 = P;
//
//		T_P1 = getSaturatedAirTemperature(ept, P1);
//
//		toCelsius(T_P0);
//		toCelsius(T_P1);
//
//		y = getNormalizedPres(P0);
//		x = getNormalizedTemp(T_P0, y);
//
//		vertices.push_back(glm::vec2(x, y));
//
//		y = getNormalizedPres(P1);
//		x = getNormalizedTemp(T_P1, y);
//
//		vertices.push_back(glm::vec2(x, y));
//
//		toKelvin(T_P0);
//		toKelvin(T_P1);
//
//		// jump to next
//		P0 = P1;
//		T_P0 = T_P1;
//	}
//#elif MOIST_ADIABAT_OPTION == 2 // Bakhshaii iterative description
//	//T = 24.0f;
//	T_P0 = T;
//
//	float e_0 = 6.112f;
//	float e_s = e_0 * exp((17.67f * (T_P0 - 273.15f)) / T_P0 - 29.65f);
//	float r_s = 0.622f * e_s / (P0 - e_s);
//	float bApprox = 1.0;
//
//	float Lv = (a*T*T*T + b*T*T + c*T + d) * 1000.0f;
//	cout << "Lv = " << Lv << endl;
//
//	float dTdP = (bApprox / P0) * ((R_d * T_P0 + Lv * r_s) / (1004.0f + (Lv * Lv * r_s * EPS * bApprox) / (R_d * T_P0 * T_P0)));
//
//	for (int i = 0; i < soundingData.size(); i++) {
//		currP = soundingData[i].data[PRES];
//		P1 = currP;
//
//		float deltaP = P1 - P0;
//
//
//		toKelvin(T_P0);
//		toKelvin(T_P1);
//		toKelvin(T);
//		///////
//		float e_s = e_0 * exp((17.67f * (T_P0 - 273.15f)) / T_P0 - 29.65f);
//		float r_s = 0.622f * e_s / (P0 - e_s);
//		float Lv = (a*T_P0*T_P0*T_P0 + b*T_P0*T_P0 + c*T_P0 + d) * P0;
//
//		float dTdP = (bApprox / P0) * ((R_d * T_P0 + Lv * r_s) / (1004.0f + (Lv * Lv * r_s * EPS * bApprox) / (R_d * T_P0 * T_P0)));
//
//		/*float e_s = e_0 * exp((17.67f * (T - 273.15f)) / T - 29.65f);
//		float r_s = 0.622f * e_s / (P0 - e_s);
//		float dTdP = (bApprox / P0) * ((R_d * T + Lv * r_s) / (1004.0f + (Lv * Lv * r_s * EPS * bApprox) / (R_d * T * T)));*/
//		//////////////////////
//
//		T_P1 = T_P0 + deltaP * dTdP;
//
//
//		toCelsius(T_P0);
//		toCelsius(T_P1);
//		toCelsius(T);
//		//cout << "T_P0 = " << T_P0 << ", T_P1 = " << T_P1 << endl;
//		//cout << "P0 = " << P0 << ", P1 = " << P1 << endl;
//
//		y = getNormalizedPres(P0);
//		x = getNormalizedTemp(T_P0, y);
//
//		vertices.push_back(glm::vec2(x, y));
//
//
//		P0 = P1;
//		T_P0 = T_P1;
//
//	}
//#elif MOIST_ADIABAT_OPTION == 3 // Bakhshaii non-iterative approach
//	T = 9.0f; // Celsius
//	P = 800.0f; // 800hPa = 80kPa
//	float theta_w = getWetBulbPotentialTemperature_degC_hPa(T, P);
//	cout << "THETA W = " << theta_w << endl;
//	/*
//	desired results:
//	g1 = -1.26
//	g2 = 53.24
//	g3 = 0.58
//	g4 = -8.84
//	g5 = -25.99
//	g6 = 0.15
//	theta_w = 17.9 degC
//	*/
//
//	theta_w = 28.0f;
//	P = 250.0f;
//	T = getPseudoadiabatTemperature_degC_hPa(theta_w, P);
//	cout << "T = " << T << endl;
//
//	for (float theta_w = MIN_TEMP; theta_w <= MAX_TEMP; theta_w += 10.0f) {
//		for (int i = 0; i < soundingData.size(); i++) {
//			P = soundingData[i].data[PRES];
//			T = getPseudoadiabatTemperature_degC_hPa(theta_w, P);
//			y = getNormalizedPres(P);
//			x = getNormalizedTemp(T, y);
//			vertices.push_back(glm::vec2(x, y));
//			numMoistAdiabats++;
//		}
//	}
//#elif MOIST_ADIABAT_OPTION == 4 // pyMeteo implementation
//
//	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += 5.0f) {
//		//T = -10.0f;
//		T = currT;
//		T += 273.15f;
//		float origT = T;
//
//		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
//		//	float p = soundingData[profileIndex].data[PRES];
//		deltaP = 1.0f;
//		for (float p = 1000.0f; p >= MIN_P; p -= deltaP) {
//			p *= 100.0f;
//			T -= dTdP_moist_degK(T, p) * deltaP * 100.0f;
//			p /= 100.0f;
//
//			y = getNormalizedPres(p);
//			x = getNormalizedTemp(getCelsius(T), y);
//			vertices.push_back(glm::vec2(x, y));
//		}
//		T = origT;
//		for (float p = 1000.0f; p <= MAX_P; p += deltaP) {
//			p *= 100.0f;
//			T += dTdP_moist_degK(T, p) * deltaP * 100.0f;
//			p /= 100.0f;
//
//			y = getNormalizedPres(p);
//			x = getNormalizedTemp(getCelsius(T), y);
//			vertices.push_back(glm::vec2(x, y));
//		}
//		numMoistAdiabats++;
//
//	}
//
//#elif MOIST_ADIABAT_OPTION == 5 // pyMeteo implementation - with spacing
//
//	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += 5.0f) {
//		//T = -10.0f;
//		T = currT;
//		T += 273.15f;
//		float origT = T;
//
//		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
//		//	float p = soundingData[profileIndex].data[PRES];
//		deltaP = 1.0f;
//		int counter = 0;
//
//		for (float p = 1000.0f; p <= MAX_P; p += deltaP) {
//			p *= 100.0f;
//			T += dTdP_moist_degK(T, p) * deltaP * 100.0f;
//			p /= 100.0f;
//
//			if ((int)p % 25 == 0 && p != 1000.0f) {
//				y = getNormalizedPres(p);
//				x = getNormalizedTemp(getCelsius(T), y);
//				vertices.push_back(glm::vec2(x, y));
//				counter++;
//			}
//		}
//		reverse(vertices.end() - counter, vertices.end()); // to draw continuous line
//		T = origT;
//
//		for (float p = 1000.0f; p >= MIN_P; p -= deltaP) {
//			p *= 100.0f;
//			T -= dTdP_moist_degK(T, p) * deltaP * 100.0f;
//			p /= 100.0f;
//
//			if ((int)p % 25 == 0) {
//				y = getNormalizedPres(p);
//				x = getNormalizedTemp(getCelsius(T), y);
//				vertices.push_back(glm::vec2(x, y));
//				counter++;
//			}
//		}
//		//cout << "Counter = " << counter << ", num isobars = " << numIsobars << endl;
//		//numMoistAdiabatEdges = counter;
//		numMoistAdiabats++;
//		moistAdiabatEdgeCount.push_back(counter);
//
//	}
//
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	// TESTING EL (regular) computation - special moist adiabat (goes through CCL)
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	{
//		int counter = 0;
//		T = CCL.x + 273.15f;
//		deltaP = 1.0f;
//		for (float p = CCL.y; p >= MIN_P; p -= deltaP) {
//			p *= 100.0f;
//			T -= dTdP_moist_degK(T, p) * deltaP * 100.0f;
//			p /= 100.0f;
//
//			if ((int)p % 25 == 0 || p == CCL.y) {
//				y = getNormalizedPres(p);
//				x = getNormalizedTemp(getCelsius(T), y);
//				vertices.push_back(glm::vec2(x, y));
//				moistAdiabat_CCL_EL.vertices.push_back(glm::vec2(x, y));
//				counter++;
//			}
//		}
//		numMoistAdiabats++;
//		moistAdiabatEdgeCount.push_back(counter);
//
//
//
//		///////////////////////////////////////////////////////////////////////////////////////////////
//		// Find EL 
//		///////////////////////////////////////////////////////////////////////////////////////////////
//
//		reverse(moistAdiabat_CCL_EL.vertices.begin(), moistAdiabat_CCL_EL.vertices.end()); // temporary reverse for finding EL
//		ELNormalized = findIntersection(moistAdiabat_CCL_EL, ambientCurve);
//		cout << "EL (normalized): x = " << ELNormalized.x << ", y = " << ELNormalized.y << endl;
//		EL = getDenormalizedCoords(ELNormalized);
//		cout << "EL: T = " << EL.x << ", P = " << EL.y << endl;
//		reverse(moistAdiabat_CCL_EL.vertices.begin(), moistAdiabat_CCL_EL.vertices.end()); // reverse back for the simulation
//
//		visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
//		visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	
//
//		visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
//		visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	
//	}
//
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	// TESTING EL (orographic) computation - special moist adiabat (goes through LCL)
//	///////////////////////////////////////////////////////////////////////////////////////////////
//	{
//		T = LCL.x + 273.15f;
//		deltaP = 1.0f;
//		for (float p = LCL.y; p >= MIN_P; p -= deltaP) {
//			p *= 100.0f;
//			T -= dTdP_moist_degK(T, p) * deltaP * 100.0f;
//			p /= 100.0f;
//
//			if ((int)p % 25 == 0 || p == LCL.y) {
//				y = getNormalizedPres(p);
//				x = getNormalizedTemp(getCelsius(T), y);
//				//vertices.push_back(glm::vec2(x, y));
//				moistAdiabat_LCL_EL.vertices.push_back(glm::vec2(x, y));
//			}
//		}
//		//numMoistAdiabats++;
//
//
//		LFCNormalized = findIntersection(moistAdiabat_LCL_EL, ambientCurve);
//		LFC = getDenormalizedCoords(LFCNormalized);
//
//		reverse(moistAdiabat_LCL_EL.vertices.begin(), moistAdiabat_LCL_EL.vertices.end());
//
//		orographicELNormalized = findIntersection(moistAdiabat_LCL_EL, ambientCurve);
//		orographicEL = getDenormalizedCoords(orographicELNormalized);
//
//		reverse(moistAdiabat_LCL_EL.vertices.begin(), moistAdiabat_LCL_EL.vertices.end());
//
//
//	}
//
//	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
//		int counter = 0;
//		moistAdiabatProfiles.push_back(Curve());
//		T = CCLProfiles[profileIndex].x + 273.15f;
//		deltaP = 1.0f;
//		for (float p = CCLProfiles[profileIndex].y; p >= MIN_P; p -= deltaP) {
//			p *= 100.0f;
//			T -= dTdP_moist_degK(T, p) * deltaP * 100.0f;
//			p /= 100.0f;
//
//			if ((int)p % 25 == 0 || p == CCLProfiles[profileIndex].y) {
//				y = getNormalizedPres(p);
//				x = getNormalizedTemp(getCelsius(T), y);
//				vertices.push_back(glm::vec2(x, y));
//				//moistAdiabat_LCL_EL.vertices.push_back(glm::vec2(x, y));
//				moistAdiabatProfiles[profileIndex].vertices.push_back(glm::vec2(x, y));
//				counter++;
//			}
//		}
//		numMoistAdiabats++;
//		moistAdiabatEdgeCount.push_back(counter);
//
//
//
//		reverse(moistAdiabatProfiles[profileIndex].vertices.begin(), moistAdiabatProfiles[profileIndex].vertices.end());
//
//		glm::vec2 tmp = findIntersection(moistAdiabatProfiles[profileIndex], ambientCurve);
//		ELProfiles.push_back(getDenormalizedCoords(tmp));
//
//		reverse(moistAdiabatProfiles[profileIndex].vertices.begin(), moistAdiabatProfiles[profileIndex].vertices.end());
//
//		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(ELProfiles.back()), -2.0f)); // point
//		float tint = (profileIndex + 1) * profileDelta;
//		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
//		visualizationPoints.push_back(glm::vec3(tint, 1.0f, 1.0f)); // color	
//	}
//
//#endif
//
//	//numMoistAdiabats++;
//
//	if (!vertices.empty()) {
//
//		glGenVertexArrays(1, &moistAdiabatsVAO);
//		glBindVertexArray(moistAdiabatsVAO);
//		glGenBuffers(1, &moistAdiabatsVBO);
//		glBindBuffer(GL_ARRAY_BUFFER, moistAdiabatsVBO);
//
//		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
//
//		glEnableVertexAttribArray(0);
//		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
//
//		glBindVertexArray(0);
//	}
//
//
//
//
//	// trying out stuff
//	P = 432.2f;
//	float normP = getNormalizedPres(P);
//	cout << "Pressure = " << P << ", normalized pressure = " << normP << endl;
//	visualizationPoints.push_back(glm::vec3(ambientCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
//	visualizationPoints.push_back(glm::vec3(1.0f, 0.0f, 0.0f)); // color
//
//	visualizationPoints.push_back(glm::vec3(dewpointCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
//	visualizationPoints.push_back(glm::vec3(0.0f, 0.0f, 1.0f)); // color
//
//
//	glGenVertexArrays(1, &visPointsVAO);
//	glBindVertexArray(visPointsVAO);
//	glGenBuffers(1, &visPointsVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, visPointsVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * visualizationPoints.size(), &visualizationPoints[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);
//
//
//	glEnableVertexAttribArray(1);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)(sizeof(glm::vec3)));
//
//	glBindVertexArray(0);
//
//
//
//
//	// Main parameters visualization
//
//	mainParameterPoints.push_back(glm::vec3(CCLNormalized, 0.0f));
//	mainParameterPoints.push_back(glm::vec3(0.0f));
//	mainParameterPoints.push_back(glm::vec3(TcNormalized, 0.0f));
//	mainParameterPoints.push_back(glm::vec3(0.0f));
//
//	mainParameterPoints.push_back(glm::vec3(ELNormalized, 0.0f));
//	mainParameterPoints.push_back(glm::vec3(0.0f));
//
//	mainParameterPoints.push_back(glm::vec3(LCLNormalized, 0.0f));
//	mainParameterPoints.push_back(glm::vec3(0.0f));
//
//	mainParameterPoints.push_back(glm::vec3(LFCNormalized, 0.0f));
//	mainParameterPoints.push_back(glm::vec3(0.0f));
//
//	mainParameterPoints.push_back(glm::vec3(orographicELNormalized, 0.0f));
//	mainParameterPoints.push_back(glm::vec3(0.0f));
//
//
//	glGenVertexArrays(1, &mainParameterPointsVAO);
//	glBindVertexArray(mainParameterPointsVAO);
//	glGenBuffers(1, &mainParameterPointsVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, mainParameterPointsVBO);
//
//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * mainParameterPoints.size(), &mainParameterPoints[0], GL_STATIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);
//
//	glEnableVertexAttribArray(1);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)(sizeof(glm::vec3)));
//
//	glBindVertexArray(0);
//
//
//	// QUAD
//	glGenVertexArrays(1, &overlayDiagramVAO);
//	glGenBuffers(1, &overlayDiagramVBO);
//	glBindVertexArray(overlayDiagramVAO);
//	glBindBuffer(GL_ARRAY_BUFFER, overlayDiagramVBO);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
//	glEnableVertexAttribArray(1);
//	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
//	glBindVertexArray(0);
//
//	// TEXTURE AND FRAMEBUFFER
//
//	glGenTextures(1, &diagramTexture);
//	glBindTexture(GL_TEXTURE_2D, diagramTexture);
//
//	//glTextureParameteri(diagramTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
//
//
//	float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
//	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
//
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, textureResolution, textureResolution, 0, GL_RGBA, GL_FLOAT, nullptr);
//
//	glGenFramebuffers(1, &diagramFramebuffer);
//	glBindFramebuffer(GL_FRAMEBUFFER, diagramFramebuffer);
//	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, diagramTexture, 0);
//
//
//	glGenTextures(1, &diagramMultisampledTexture);
//	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, diagramMultisampledTexture);
//	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 12, GL_RGBA16F, textureResolution, textureResolution, false);
//
//	glGenFramebuffers(1, &diagramMultisampledFramebuffer);
//	glBindFramebuffer(GL_FRAMEBUFFER, diagramMultisampledFramebuffer);
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, diagramMultisampledTexture, 0);
//
//
//	GLfloat lineWidthRange[2] = { 0.0f, 0.0f };
//	glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, lineWidthRange);
//	// Maximum supported line width is in lineWidthRange[1].
//	cout << lineWidthRange[0] << " , " << lineWidthRange[1] << endl;
//}
//
//


