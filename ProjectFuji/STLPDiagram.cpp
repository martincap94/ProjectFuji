#include "STLPDiagram.h"

#include <iostream>
#include <fstream>
#include "Config.h"

#include "STLPUtils.h"
#include "Utils.h"
#include "MainFramebuffer.h"

STLPDiagram::STLPDiagram() {
	recalculateProfileDelta();
}

STLPDiagram::STLPDiagram(string filename) : STLPDiagram() {

	loadSoundingData(filename);
}


STLPDiagram::~STLPDiagram() {
	delete textRend;
}

void STLPDiagram::init(string filename) {
	loadSoundingData(filename);

	initFreetype();

	initBuffers();
	initCurves();
}

void STLPDiagram::loadSoundingData(string filename) {

	//cout << "Sounding filename = " << filename << endl;
	soundingFile = filename;
	filename = string(SOUNDING_DATA_DIR) + filename;
	ifstream infile(filename);

	if (!infile.is_open()) {
		cerr << filename << " could not be opened!" << endl;
		exit(EXIT_FAILURE);
		//return;
	}
	// assume the file is in correct format!
	string line;

	// read first line (the header)
	getline(infile, line);

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
	float y;
	P = soundingData[0].data[PRES];
	y = getNormalizedPres(P);
	vertices.push_back(glm::vec2(xmin, y));
	vertices.push_back(glm::vec2(xmax, y));
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

	glBindBuffer(GL_ARRAY_BUFFER, ambientTemperatureVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
}

void STLPDiagram::generateMixingRatioLineOld() {
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
	float T = getDenormalizedTemp(findIntersectionNaive(groundIsobar, dewpointCurve).x, getNormalizedPres(P));	// this is more general (when user changes dewpoint curve for example)



	float eps = Rd / Rm;
	//float satVP = C * exp((A * T) / (T + B));	// saturation vapor pressure: e(T)
	float satVP = getSaturationVaporPressure(T);
	//float W = (eps * satVP) / (P - satVP);
	float W = getMixingRatioOfWaterVapor(T, P);

	cout << " -> Computed W = " << W << endl;

	float deltaP = 20.0f;


	float x, y;
	while (P >= MIN_P) {


		float fracPart = log((W * P) / (C * (W + eps)));
		float computedT = (B * fracPart) / (A - fracPart);

		y = getNormalizedPres(P);
		x = getNormalizedTemp(T, y);

		if (x < xmin || x > xmax || y < 0.0f || y > 1.0f) {
			break;
		}

		vertices.push_back(glm::vec2(x, y));


		float delta = 10.0f; // should be a small integer - produces dashed line (logarithmic)
							 //float offsetP = P - delta;
		float offsetP = P - deltaP; // produces continuous line
		fracPart = log((W * offsetP) / (C * (W + eps)));
		computedT = (B * fracPart) / (A - fracPart);

		y = getNormalizedPres(offsetP);
		x = getNormalizedTemp(T, y);
		vertices.push_back(glm::vec2(x, y));

		P -= deltaP;

	}
	mixingCCL.vertices = vertices;

	glBindBuffer(GL_ARRAY_BUFFER, isohumesVBO);
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
	float T = getDenormalizedTemp(findIntersectionNaive(groundIsobar, dewpointCurve).x, getNormalizedPres(P));	// this is more general (when user changes dewpoint curve for example)



	float eps = Rd / Rm;
	//float satVP = C * exp((A * T) / (T + B));	// saturation vapor pressure: e(T)
	//float satVP = getSaturationVaporPressure(T) / 100.0f;
	//float W = (eps * satVP) / (P - satVP);
	float W = getMixingRatioOfWaterVapor(T, P * 100.0f);

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

void STLPDiagram::generateDryAdiabat(float theta, vector<glm::vec2>& vertices, float P0, vector<int>* edgeCounter, bool incrementCounter, float deltaP, Curve * curve) {

	int vertexCounter = 0;

	if (curve != nullptr) {
		curve->vertices.clear();
	}

	float x, y, T, P;

	for (P = MAX_P; P >= MIN_P; P -= deltaP) {

		T = computeAbsoluteFromThetaC(theta, P, P0);

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
		numDryAdiabats++;
		edgeCounter->push_back(vertexCounter);
		//dryAdiabatEdgeCount.push_back(vertexCounter);
	}


}

void STLPDiagram::generateMoistAdiabat(float theta, float startP, vector<glm::vec2>& vertices, float P0, vector<int>* edgeCounter, bool incrementCounter, float deltaP, Curve * curve, float smallDeltaP) {

	float T = getKelvin(theta);
	int vertexCounter = 0;
	float x, y;

	if (curve != nullptr) {
		curve->vertices.clear();
	}

	for (float p = startP; p >= MIN_P; p -= smallDeltaP) {
		p *= 100.0f;
		T -= dTdp_moist(T, p) * smallDeltaP * 100.0f;
		p /= 100.0f;

		if ((int)p % (int)deltaP == 0 || (int)p % (int)startP == 0) {
			y = getNormalizedPres(p);
			x = getNormalizedTemp(getCelsius(T), y);
			vertices.push_back(glm::vec2(x, y));
			if (curve != nullptr) {
				curve->vertices.push_back(glm::vec2(x, y));
			}
			vertexCounter++;
		}
	}
	//cout << "Counter = " << counter << ", num isobars = " << numIsobars << endl;
	if (incrementCounter && edgeCounter != nullptr) {
		numMoistAdiabats++;
		edgeCounter->push_back(vertexCounter);
	}





}



void STLPDiagram::resetToDefault() {
	cout << "Resetting diagram to default sounding data values" << endl;

	initAmbientTemperatureCurve();
	initDewpointCurve();

	recalculateParameters();

}

void STLPDiagram::initBuffers() {
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
	glGenVertexArrays(1, &dryAdiabatsVAO);
	glBindVertexArray(dryAdiabatsVAO);
	glGenBuffers(1, &dryAdiabatsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, dryAdiabatsVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// MOIST ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	glGenVertexArrays(1, &moistAdiabatsVAO);
	glBindVertexArray(moistAdiabatsVAO);
	glGenBuffers(1, &moistAdiabatsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, moistAdiabatsVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);



	///////////////////////////////////////////////////////////////////////////////////////
	// PARTICLES
	///////////////////////////////////////////////////////////////////////////////////////
	glGenVertexArrays(1, &particlesVAO);
	glBindVertexArray(particlesVAO);
	glGenBuffers(1, &particlesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);

}

void STLPDiagram::initCurves() {

	float x, y;
	// Initialize main variables

	xmin = 0.0f;
	xmax = 1.0f;

	ymin = getNormalizedPres(MIN_P);
	ymax = getNormalizedPres(MAX_P);

	sP0 = soundingData[0].data[PRES];

	float P0 = soundingData[0].data[PRES];
	float P;
	float T;
	float y0 = getNormalizedPres(P0);

	xaxis.vertices.push_back(glm::vec2(xmin, ymax));
	xaxis.vertices.push_back(glm::vec2(xmax, ymax));
	yaxis.vertices.push_back(glm::vec2(xmin, ymin));
	yaxis.vertices.push_back(glm::vec2(xmin, ymax));

	xaxis.initBuffers();
	yaxis.initBuffers();

	groundIsobar.vertices.push_back(glm::vec2(xmin, y0));
	groundIsobar.vertices.push_back(glm::vec2(xmax * 10.0f, y0)); // we want it to be long for Tc computation (intersection beyond xmax)

	groundIsobar.initBuffers();

	TcProfiles.reserve(numProfiles);
	CCLProfiles.reserve(numProfiles);
	ELProfiles.reserve(numProfiles);
	dryAdiabatProfiles.reserve(numProfiles);
	moistAdiabatProfiles.reserve(numProfiles);

	///////////////////////////////////////////////////////////////////////////////////////
	// ISOBARS
	///////////////////////////////////////////////////////////////////////////////////////

	generateIsobars();

	vector<glm::vec2> vertices;


	///////////////////////////////////////////////////////////////////////////////////////
	// TEMPERATURE POINTS
	///////////////////////////////////////////////////////////////////////////////////////

	//vertices.clear();
	temperaturePointsCount = 0;
	for (int i = MIN_TEMP; i <= MAX_TEMP; i += 10) {
		T = getNormalizedTemp(i, ymax);
		temperaturePoints.push_back(glm::vec2(T, ymax));
		temperaturePointsCount++;
	}

	glBindBuffer(GL_ARRAY_BUFFER, temperaturePointsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * temperaturePoints.size(), &temperaturePoints[0], GL_STATIC_DRAW);


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

	CCLNormalized = findIntersectionNaive(mixingCCL, ambientCurve);
	CCL = getDenormalizedCoords(CCLNormalized);


	///////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	vertices.clear();
	numDryAdiabats = 0;
	int counter;

	for (float theta = MIN_TEMP; theta <= MAX_TEMP * 5; theta += dryAdiabatDeltaT) {
		generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: T_c <-> CCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	float theta = computeThetaFromAbsoluteC(CCL.x, CCL.y, P0);
	generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount, true, CURVE_DELTA_P, &TcDryAdiabat);

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: Ground ambient temperature <-> LCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	theta = computeThetaFromAbsoluteC(soundingData[0].data[TEMP], P0, P0);
	generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount, true, CURVE_DELTA_P, &LCLDryAdiabatCurve);


	///////////////////////////////////////////////////////////////////////////////////////////////
	// Tc and LCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	TcNormalized = findIntersectionNaive(groundIsobar, TcDryAdiabat);
	Tc = getDenormalizedCoords(TcNormalized);

	// Check correctness by computing thetaCCL == Tc
	float thetaCCL = (CCL.x + 273.15f) * pow((P0 / CCL.y), k_ratio);
	thetaCCL -= 273.15f;

	LCLNormalized = findIntersectionNaive(LCLDryAdiabatCurve, mixingCCL);
	LCL = getDenormalizedCoords(LCLNormalized);
	///////////////////////////////////////////////////////////////////////////////////////////////



	// Testing out profiles
	for (int i = 0; i < numProfiles; i++) {
		TcProfiles.push_back(Tc + glm::vec2((i + 1) * profileDelta, 0.0f));
		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(TcProfiles.back()), -2.0f)); // point
		float tint = (i + 1) * profileDelta;
		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		visualizationPoints.push_back(glm::vec3(tint, 0.0f, 0.0f)); // color	
	}

	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {

		dryAdiabatProfiles.push_back(Curve());

		float theta = computeAbsoluteFromThetaC(TcProfiles[profileIndex].x, P0, P0);
		generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount, true, CURVE_DELTA_P, &dryAdiabatProfiles[profileIndex]);


		CCLProfiles.push_back(getDenormalizedCoords(findIntersectionNaive(dryAdiabatProfiles[profileIndex], ambientCurve)));

		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(CCLProfiles.back()), -2.0f)); // point
		float tint = (profileIndex + 1) * profileDelta;
		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		visualizationPoints.push_back(glm::vec3(tint, 0.0f, 1.0f)); // color	

	}


	glBindBuffer(GL_ARRAY_BUFFER, dryAdiabatsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);



	///////////////////////////////////////////////////////////////////////////////////////
	// MOIST ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	/*cout << "////////////////////////////////////////////////////" << endl;
	cout << "// MOIST ADIABATS" << endl;
	cout << "////////////////////////////////////////////////////" << endl;*/

	float deltaP = 20.0f;

	vertices.clear();

	numMoistAdiabats = 0;


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

	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += moistAdiabatDeltaT) {
		generateMoistAdiabat(currT, MAX_P, vertices, P0, &moistAdiabatEdgeCount);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING EL (regular) computation - special moist adiabat (goes through CCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(CCL.x, CCL.y, vertices, P0, &moistAdiabatEdgeCount, true, CURVE_DELTA_P, &moistAdiabat_CCL_EL);

	{
		///////////////////////////////////////////////////////////////////////////////////////////////
		// Find EL 
		///////////////////////////////////////////////////////////////////////////////////////////////

		ELNormalized = findIntersectionNaive(moistAdiabat_CCL_EL, ambientCurve, true);
		//cout << "EL (normalized): x = " << ELNormalized.x << ", y = " << ELNormalized.y << endl;
		EL = getDenormalizedCoords(ELNormalized);
		//cout << "EL: T = " << EL.x << ", P = " << EL.y << endl;

		visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
		visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	

		visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
		visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	
	}


	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING EL (orographic) computation - special moist adiabat (goes through LCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(LCL.x, LCL.y, vertices, P0, &moistAdiabatEdgeCount, true, CURVE_DELTA_P, &moistAdiabat_LCL_EL);
	{

		LFCNormalized = findIntersectionNaive(moistAdiabat_LCL_EL, ambientCurve);
		LFC = getDenormalizedCoords(LFCNormalized);

		OrographicELNormalized = findIntersectionNaive(moistAdiabat_LCL_EL, ambientCurve, true);
		OrographicEL = getDenormalizedCoords(OrographicELNormalized);


	}

	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
		//int counter = 0;
		moistAdiabatProfiles.push_back(Curve());
		generateMoistAdiabat(CCLProfiles[profileIndex].x, CCLProfiles[profileIndex].y, vertices, P0, &moistAdiabatEdgeCount, true, CURVE_DELTA_P, &moistAdiabatProfiles[profileIndex]);

		glm::vec2 tmp = findIntersectionNaive(moistAdiabatProfiles[profileIndex], ambientCurve, true);
		ELProfiles.push_back(getDenormalizedCoords(tmp));

		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(ELProfiles.back()), -2.0f)); // point
		float tint = (profileIndex + 1) * profileDelta;
		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		visualizationPoints.push_back(glm::vec3(tint, 1.0f, 1.0f)); // color	
	}



	//numMoistAdiabats++;
	glBindBuffer(GL_ARRAY_BUFFER, moistAdiabatsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);





	// trying out stuff
	P = 432.2f;
	float normP = getNormalizedPres(P);
	//cout << "Pressure = " << P << ", normalized pressure = " << normP << endl;
	visualizationPoints.push_back(glm::vec3(ambientCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
	visualizationPoints.push_back(glm::vec3(1.0f, 0.0f, 0.0f)); // color

	visualizationPoints.push_back(glm::vec3(dewpointCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
	visualizationPoints.push_back(glm::vec3(0.0f, 0.0f, 1.0f)); // color


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

	mainParameterPoints.push_back(glm::vec3(CCLNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));
	mainParameterPoints.push_back(glm::vec3(TcNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(ELNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(LCLNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(LFCNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(OrographicELNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));


	glGenVertexArrays(1, &mainParameterPointsVAO);
	glBindVertexArray(mainParameterPointsVAO);
	glGenBuffers(1, &mainParameterPointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, mainParameterPointsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * mainParameterPoints.size(), &mainParameterPoints[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)(sizeof(glm::vec3)));

	glBindVertexArray(0);


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


	GLfloat lineWidthRange[2] = { 0.0f, 0.0f };
	glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, lineWidthRange);
	// Maximum supported line width is in lineWidthRange[1].
	//cout << lineWidthRange[0] << " , " << lineWidthRange[1] << endl;
}

void STLPDiagram::recalculateParameters() {

	
	recalculateProfileDelta();


	generateMixingRatioLine();

	CCLNormalized = findIntersectionNaive(mixingCCL, ambientCurve);
	CCL = getDenormalizedCoords(CCLNormalized);


	float P0 = soundingData[0].data[PRES];
	vector<glm::vec2> vertices;

	numDryAdiabats = 0;
	dryAdiabatEdgeCount.clear();

	for (float theta = MIN_TEMP; theta <= MAX_TEMP * 5; theta += dryAdiabatDeltaT) {
		generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: T_c <-> CCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	float theta = computeThetaFromAbsoluteC(CCL.x, CCL.y, P0);
	TcDryAdiabat.vertices.clear(); // hack
	generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount, true, CURVE_DELTA_P, &TcDryAdiabat);

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABAT: Ground ambient temperature <-> LCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	theta = computeThetaFromAbsoluteC(soundingData[0].data[TEMP], P0, P0); // will need line-to-line intersection for the ground temperature
	LCLDryAdiabatCurve.vertices.clear(); // hack
	generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount, true, CURVE_DELTA_P, &LCLDryAdiabatCurve);



	///////////////////////////////////////////////////////////////////////////////////////////////
	// Tc and LCL
	///////////////////////////////////////////////////////////////////////////////////////////////
	TcNormalized = findIntersectionNaive(groundIsobar, TcDryAdiabat);
	Tc = getDenormalizedCoords(TcNormalized);

	LCLNormalized = findIntersectionNaive(LCLDryAdiabatCurve, mixingCCL);
	LCL = getDenormalizedCoords(LCLNormalized);
	///////////////////////////////////////////////////////////////////////////////////////////////


	//// Profiles
	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
		TcProfiles[profileIndex] = Tc + glm::vec2((profileIndex + 1) * profileDelta, 0.0f);

		float theta = computeAbsoluteFromThetaC(TcProfiles[profileIndex].x, P0, P0);
		dryAdiabatProfiles[profileIndex].vertices.clear();
		generateDryAdiabat(theta, vertices, P0, &dryAdiabatEdgeCount, true, CURVE_DELTA_P, &dryAdiabatProfiles[profileIndex]);

		CCLProfiles[profileIndex] = getDenormalizedCoords(findIntersectionNaive(dryAdiabatProfiles[profileIndex], ambientCurve));
	}

	int sum = 0;
	for (int i = 0; i < numDryAdiabats - numProfiles - 2; i++) {
		sum += dryAdiabatEdgeCount[i];
	}

	glNamedBufferData(dryAdiabatsVBO, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);




	vertices.clear();
	numMoistAdiabats = 0;
	moistAdiabatEdgeCount.clear();

	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += moistAdiabatDeltaT) {
		generateMoistAdiabat(currT, MAX_P, vertices, P0, &moistAdiabatEdgeCount);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING EL (regular) computation - special moist adiabat (goes through CCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(CCL.x, CCL.y, vertices, P0, &moistAdiabatEdgeCount, true, CURVE_DELTA_P, &moistAdiabat_CCL_EL);


	ELNormalized = findIntersectionNaive(moistAdiabat_CCL_EL, ambientCurve, true);
	EL = getDenormalizedCoords(ELNormalized);


	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING EL (orographic) computation - special moist adiabat (goes through LCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	generateMoistAdiabat(LCL.x, LCL.y, vertices, P0, &moistAdiabatEdgeCount, true, CURVE_DELTA_P, &moistAdiabat_LCL_EL);

	LFCNormalized = findIntersectionNaive(moistAdiabat_LCL_EL, ambientCurve);
	LFC = getDenormalizedCoords(LFCNormalized);

	OrographicELNormalized = findIntersectionNaive(moistAdiabat_LCL_EL, ambientCurve, true);
	OrographicEL = getDenormalizedCoords(OrographicELNormalized);



	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
		generateMoistAdiabat(CCLProfiles[profileIndex].x, CCLProfiles[profileIndex].y, vertices, P0, &moistAdiabatEdgeCount, true, CURVE_DELTA_P, &moistAdiabatProfiles[profileIndex]);

		glm::vec2 tmp = findIntersectionNaive(moistAdiabatProfiles[profileIndex], ambientCurve, true);
		ELProfiles[profileIndex] = getDenormalizedCoords(tmp);
	}

	glNamedBufferData(moistAdiabatsVBO, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);


	// very hacky - rewrite
	mainParameterPoints.clear();
	mainParameterPoints.push_back(glm::vec3(CCLNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));
	mainParameterPoints.push_back(glm::vec3(TcNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(ELNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(LCLNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(LFCNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(OrographicELNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));


	glBindBuffer(GL_ARRAY_BUFFER, mainParameterPointsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * mainParameterPoints.size(), &mainParameterPoints[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
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

void STLPDiagram::draw(ShaderProgram &shader, ShaderProgram &altShader) {

	int counter;
	glLineWidth(1.0f);

	reportGLErrors("STLP 1");

	glUseProgram(shader.id);

	shader.setBool("u_CropBounds", (bool)cropBounds);

	if (showIsobars) {
		shader.setVec3("color", glm::vec3(0.8f, 0.8f, 0.8f));
		glBindVertexArray(isobarsVAO);
		glDrawArrays(GL_LINES, 0, numIsobars * 2);
	}
	reportGLErrors("STLP 2");


	glPointSize(8.0f);
	shader.setVec3("color", glm::vec3(0.5f, 0.7f, 0.0f));
	glBindVertexArray(temperaturePointsVAO);
	glDrawArrays(GL_POINTS, 0, temperaturePointsCount);

	reportGLErrors("STLP 3");

	if (showIsotherms) {
		glPointSize(8.0f);
		shader.setVec3("color", glm::vec3(0.8f, 0.8f, 0.8f));
		glBindVertexArray(isothermsVAO);
		glDrawArrays(GL_LINES, 0, isothermsCount * 2);
	}
	reportGLErrors("STLP 4");


	if (showAmbientTemperatureCurve) {
		shader.setVec3("color", glm::vec3(0.7f, 0.1f, 0.15f));
		glBindVertexArray(ambientTemperatureVAO);
		glDrawArrays(GL_LINE_STRIP, 0, soundingData.size() * 2);
	}
	reportGLErrors("STLP 5");


	if (showDewpointCurve) {
		shader.setVec3("color", glm::vec3(0.1f, 0.7f, 0.15f));
		glBindVertexArray(dewTemperatureVAO);
		glDrawArrays(GL_LINE_STRIP, 0, soundingData.size() * 2);
	}
	reportGLErrors("STLP 6");


	if (showIsohumes) {
		shader.setVec3("color", glm::vec3(0.1f, 0.15f, 0.7f));
		glBindVertexArray(isohumesVAO);
		glDrawArrays(GL_LINES, 0, soundingData.size() * 2);
	}
	reportGLErrors("STLP 7");


	if (showDryAdiabats) {
		shader.setVec3("color", glm::vec3(0.6f, 0.6f, 0.6f));
		glBindVertexArray(dryAdiabatsVAO);

		glLineWidth(0.01f);

		counter = 0;
		for (int i = 0; i < numDryAdiabats; i++) {
			//glDrawArrays(GL_LINE_STRIP, (numIsobars-1) * i, numIsobars - 1);

			glDrawArrays(GL_LINE_STRIP, counter, dryAdiabatEdgeCount[i]);
			counter += dryAdiabatEdgeCount[i];
		}
	}
	reportGLErrors("STLP 8");


	if (showMoistAdiabats) {
		glPointSize(2.0f);
		shader.setVec3("color", glm::vec3(0.2f, 0.6f, 0.8f));
		glBindVertexArray(moistAdiabatsVAO);
		//glDrawArrays(GL_LINE_STRIP, 0, 1000000);
		//glDrawArrays(GL_POINTS, 0, 100000);

		counter = 0;
		for (int i = 0; i < numMoistAdiabats; i++) {
			//glDrawArrays(GL_LINE_STRIP, (numIsobars - 1) * profileIndex, numIsobars - 1);
			//glDrawArrays(GL_POINTS, 2 * (numIsobars - 1) * profileIndex, 2 * (numIsobars - 1));
			//glDrawArrays(GL_LINE_STRIP, numMoistAdiabatEdges * i, numMoistAdiabatEdges);
			//glDrawArrays(GL_POINTS, 2 * numMoistAdiabatEdges * profileIndex, 2 * numMoistAdiabatEdges);

			glDrawArrays(GL_LINE_STRIP, counter, moistAdiabatEdgeCount[i]);
			counter += moistAdiabatEdgeCount[i];
		}
	}
	reportGLErrors("STLP 9");


	//glPointSize(9.0f);
	//shader.setVec3("color", glm::vec3(1.0f, 0.0f, 0.0f));
	//glBindVertexArray(CCLVAO);
	//glDrawArrays(GL_POINTS, 0, 1);

	//glPointSize(9.0f);
	//shader.setVec3("color", glm::vec3(0.6f, 0.3f, 0.6f));
	//glBindVertexArray(TcVAO);
	//glDrawArrays(GL_POINTS, 0, 1);


	xaxis.draw(shader);
	yaxis.draw(shader);
	groundIsobar.draw(shader);
	reportGLErrors("STLP 10");


	glPointSize(3.0f);
	glUseProgram(altShader.id);
	glBindVertexArray(visPointsVAO);
	glDrawArrays(GL_POINTS, 0, visualizationPoints.size() / 2);
	reportGLErrors("STLP 11");


	glPointSize(6.0f);
	glBindVertexArray(mainParameterPointsVAO);
	glDrawArrays(GL_POINTS, 0, mainParameterPoints.size() / 2);
	reportGLErrors("STLP 12");




	// draw particles on top of everything
	shader.use();
	GLboolean depthTestEnabled;
	glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);
	glDisable(GL_DEPTH_TEST);
	if (particlePoints.size() > 0) {
		//cout << "?" << endl;
		glPointSize(2.0f);
		shader.setVec3("color", glm::vec3(1.0f, 0.0f, 0.0f));

		glBindVertexArray(particlesVAO);
		//glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
		//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * particlePoints.size(), &particlePoints[0], GL_DYNAMIC_DRAW);
		glNamedBufferData(particlesVBO, sizeof(glm::vec2) * particlePoints.size(), &particlePoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_POINTS, 0, particlePoints.size());
	}
	if (depthTestEnabled) {
		glEnable(GL_DEPTH_TEST);
	}


}

void STLPDiagram::drawText(ShaderProgram &shader) {

	int i = 0;
	for (int temp = MIN_TEMP; temp <= MAX_TEMP; temp += 10) {
		textRend->RenderText(shader, to_string(temp), temperaturePoints[i].x, temperaturePoints[i].y + 0.02f);
		i++;
	}

	textRend->RenderText(shader, "CCL", CCLNormalized.x, CCLNormalized.y);
	textRend->RenderText(shader, "Tc", TcNormalized.x, TcNormalized.y);
	textRend->RenderText(shader, "EL", ELNormalized.x, ELNormalized.y);
	textRend->RenderText(shader, "LCL", LCLNormalized.x, LCLNormalized.y);
	textRend->RenderText(shader, "LFC", LFCNormalized.x, LFCNormalized.y);
	textRend->RenderText(shader, "EL2", OrographicELNormalized.x, OrographicELNormalized.y);

	textRend->RenderText(shader, to_string((int)soundingData[0].data[PRES]), 0.0f - 0.04f, 1.0f);
	for (i = 1000.0f; i >= MIN_P; i -= 100) {
		textRend->RenderText(shader, to_string(i), 0.0f - 0.04f, getNormalizedPres(i));
		textRend->RenderText(shader, to_string((int)getAltitudeFromPressure(i)) + "[m]", 0.0f + 0.01f, getNormalizedPres(i));
	}

	textRend->RenderText(shader, "Temperature (C)", 0.45f, 1.10f);
	textRend->RenderText(shader, "P (hPa)", -0.15f, 0.5f);

	textRend->RenderText(shader, "SkewT/LogP (" + soundingFile + ")", 0.4f, -0.05f, 0.0006f);


}

void STLPDiagram::drawOverlayDiagram(ShaderProgram *shader, GLuint textureId) {
	//GLint current_program_id = 0;
	//glGetIntegerv(GL_CURRENT_PROGRAM, &current_program_id);
	GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);

	glDisable(GL_DEPTH_TEST);
	shader->use();
	glActiveTexture(GL_TEXTURE0);

	if (textureId == -1) {
		glBindTextureUnit(0, diagramTexture);
	} else {
		glBindTextureUnit(0, textureId);
	}
	glUniform1i(glGetUniformLocation(shader->id, "u_Texture"), 0);

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
	float width = overlayDiagramWidth;
	float height = overlayDiagramHeight;
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

	} else {
		cout << "Incorrect index (out of bounds)." << endl;
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



















void STLPDiagram::initBuffersOld() {


	// Initialize main variables

	float xmin = 0.0f;
	float xmax = 1.0f;

	float ymin = getNormalizedPres(MIN_P);
	float ymax = getNormalizedPres(MAX_P);

	float P0 = soundingData[0].data[PRES];
	float P;
	float T;

	xaxis.vertices.push_back(glm::vec2(xmin, ymax));
	xaxis.vertices.push_back(glm::vec2(xmax, ymax));
	yaxis.vertices.push_back(glm::vec2(xmin, ymin));
	yaxis.vertices.push_back(glm::vec2(xmin, ymax));

	xaxis.initBuffers();
	yaxis.initBuffers();

	TcProfiles.reserve(numProfiles);
	CCLProfiles.reserve(numProfiles);
	ELProfiles.reserve(numProfiles);
	dryAdiabatProfiles.reserve(numProfiles);
	moistAdiabatProfiles.reserve(numProfiles);

	///////////////////////////////////////////////////////////////////////////////////////
	// ISOBARS
	///////////////////////////////////////////////////////////////////////////////////////

	vector<glm::vec2> vertices;

	numIsobars = 0;
	for (P = MAX_P; P >= MIN_P; P -= CURVE_DELTA_P) {
		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
		//P = soundingData[profileIndex].data[PRES];
		float y = getNormalizedPres(P);
		vertices.push_back(glm::vec2(xmin, y));
		vertices.push_back(glm::vec2(xmax, y));
		numIsobars++;
	}
	float y;
	P = soundingData[0].data[PRES];
	y = getNormalizedPres(P);
	vertices.push_back(glm::vec2(xmin, y));
	vertices.push_back(glm::vec2(xmax, y));
	numIsobars++;


	glGenVertexArrays(1, &isobarsVAO);
	glBindVertexArray(isobarsVAO);
	glGenBuffers(1, &isobarsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, isobarsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// TEMPERATURE POINTS
	///////////////////////////////////////////////////////////////////////////////////////

	//vertices.clear();
	temperaturePointsCount = 0;
	for (int i = MIN_TEMP; i <= MAX_TEMP; i += 10) {
		T = getNormalizedTemp(i, ymax);
		temperaturePoints.push_back(glm::vec2(T, ymax));
		temperaturePointsCount++;
	}

	glGenVertexArrays(1, &temperaturePointsVAO);
	glBindVertexArray(temperaturePointsVAO);
	glGenBuffers(1, &temperaturePointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, temperaturePointsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * temperaturePoints.size(), &temperaturePoints[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	///////////////////////////////////////////////////////////////////////////////////////
	// ISOTHERMS
	///////////////////////////////////////////////////////////////////////////////////////

	vertices.clear();

	float x;
	y;

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


	glGenVertexArrays(1, &isothermsVAO);
	glBindVertexArray(isothermsVAO);
	glGenBuffers(1, &isothermsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, isothermsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);



	///////////////////////////////////////////////////////////////////////////////////////
	// AMBIENT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////
	vertices.clear();
	for (int i = 0; i < soundingData.size(); i++) {

		float P = soundingData[i].data[PRES];
		float T = soundingData[i].data[TEMP];

		y = getNormalizedPres(P);
		x = getNormalizedTemp(T, y);

		vertices.push_back(glm::vec2(x, y));
	}

	ambientCurve.vertices = vertices;

	glGenVertexArrays(1, &ambientTemperatureVAO);
	glBindVertexArray(ambientTemperatureVAO);
	glGenBuffers(1, &ambientTemperatureVBO);
	glBindBuffer(GL_ARRAY_BUFFER, ambientTemperatureVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);




	///////////////////////////////////////////////////////////////////////////////////////
	// DEWPOINT TEMPERATURE PIECE-WISE LINEAR CURVE
	///////////////////////////////////////////////////////////////////////////////////////
	vertices.clear();
	for (int i = 0; i < soundingData.size(); i++) {

		float P = soundingData[i].data[PRES];
		float T = soundingData[i].data[DWPT];

		y = getNormalizedPres(P);
		x = getNormalizedTemp(T, y);

		vertices.push_back(glm::vec2(x, y));
	}
	dewpointCurve.vertices = vertices;


	glGenVertexArrays(1, &dewTemperatureVAO);
	glBindVertexArray(dewTemperatureVAO);
	glGenBuffers(1, &dewTemperatureVBO);
	glBindBuffer(GL_ARRAY_BUFFER, dewTemperatureVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);



	///////////////////////////////////////////////////////////////////////////////////////
	// ISOHUMES (MIXING RATIO LINES)
	///////////////////////////////////////////////////////////////////////////////////////
	cout << "////////////////////////////////////////////////////" << endl;
	cout << "// ISOHUMES (MIXING RATIO LINES)" << endl;
	cout << "////////////////////////////////////////////////////" << endl;

	vertices.clear();

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
	T = soundingData[0].data[DWPT];
	P = soundingData[0].data[PRES];


	float eps = Rd / Rm;
	//float satVP = C * exp((A * T) / (T + B));	// saturation vapor pressure: e(T)
	float satVP = getSaturationVaporPressure(T);
	//float W = (eps * satVP) / (P - satVP);
	float W = getMixingRatioOfWaterVapor(T, P);

	cout << " -> Computed W = " << W << endl;

	float deltaP = 20.0f;

	while (P >= MIN_P) {


		float fracPart = log((W * P) / (C * (W + eps)));
		float computedT = (B * fracPart) / (A - fracPart);

		cout << " -> Computed T = " << computedT << endl;

		y = getNormalizedPres(P);
		x = getNormalizedTemp(T, y);

		if (x < xmin || x > xmax || y < 0.0f || y > 1.0f) {
			break;
		}

		vertices.push_back(glm::vec2(x, y));


		float delta = 10.0f; // should be a small integer - produces dashed line (logarithmic)
							 //float offsetP = P - delta;
		float offsetP = P - deltaP; // produces continuous line
		fracPart = log((W * offsetP) / (C * (W + eps)));
		computedT = (B * fracPart) / (A - fracPart);
		cout << " -> Second computed T = " << computedT << endl;


		y = getNormalizedPres(offsetP);
		x = getNormalizedTemp(T, y);
		vertices.push_back(glm::vec2(x, y));

		P -= deltaP;

	}

	mixingCCL.vertices = vertices;


	CCLNormalized = findIntersectionNaive(mixingCCL, ambientCurve);
	cout << "CCL (normalized) = " << CCLNormalized.x << ", " << CCLNormalized.y << endl;

	CCL = getDenormalizedCoords(CCLNormalized);

	cout << "CCL = " << CCL.x << ", " << CCL.y << endl;



	glGenVertexArrays(1, &CCLVAO);
	glBindVertexArray(CCLVAO);
	glGenBuffers(1, &CCLVBO);
	glBindBuffer(GL_ARRAY_BUFFER, CCLVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2), &CCLNormalized, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	glGenVertexArrays(1, &isohumesVAO);
	glBindVertexArray(isohumesVAO);
	glGenBuffers(1, &isohumesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, isohumesVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);



	///////////////////////////////////////////////////////////////////////////////////////
	// DRY ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	cout << "////////////////////////////////////////////////////" << endl;
	cout << "// DRY ADIABATS" << endl;
	cout << "////////////////////////////////////////////////////" << endl;

	vertices.clear();

	/*
	Dry adiabats feature the thermodynamic behaviour of unsaturated air parcels moving upwards (or downwards).
	They represent the dry adiabatic lapse rate (DALR).
	This thermodynamic behaviour is valid for all air parcels moving between the ground and the convective
	condensation level (CCLNormalized).

	T(P) = theta / ((P0 / P)^(Rd / cp))
	where
	P0 is the initial value of pressure (profileIndex.e. ground pressure)
	cp is the heat capacity of dry air at constant pressure
	(cv is the heat capacity of dry air at constant volume)
	Rd is the gas constant for dry air [J kg^-1 K^-1]
	k = Rd / cp = (cp - cv) / cp =(approx)= 0.286
	*/

	float k = 0.286f; // Rd / cp

	numDryAdiabats = 0;
	int counter;

	for (float theta = MIN_TEMP; theta <= MAX_TEMP * 5; theta += 10.0f) {
		counter = 0;

		for (P = MAX_P; P >= MIN_P; P -= CURVE_DELTA_P) {
			//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
			//float P = soundingData[profileIndex].data[PRES];

			float T = (theta + 273.16f) / pow((P0 / P), k);
			T -= 273.16f;

			y = getNormalizedPres(P);
			x = getNormalizedTemp(T, y);

			vertices.push_back(glm::vec2(x, y));
			counter++;
		}
		numDryAdiabats++;
		dryAdiabatEdgeCount.push_back(counter);
	}



	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING Tc computation - special dry adiabat (and its x axis intersection)
	///////////////////////////////////////////////////////////////////////////////////////////////
	{
		P0 = soundingData[0].data[PRES];

		float theta = (CCL.x + 273.15f) * powf(P0 / CCL.y, k);
		theta -= 273.15f;
		cout << "CCL theta = " << theta << endl;
		cout << "Tc Dry adiabat: " << endl;

		counter = 0;


		for (int i = 0; i < soundingData.size(); i++) {

			float P = soundingData[i].data[PRES];

			float T = (theta + 273.15f) * pow((P / P0), k); // do not forget to use Kelvin
			T -= 273.15f; // convert back to Celsius

			y = getNormalizedPres(P);
			x = getNormalizedTemp(T, y);

			TcDryAdiabat.vertices.push_back(glm::vec2(x, y));
			vertices.push_back(glm::vec2(x, y));
			counter++;

			cout << " | " << x << ", " << y << endl;
		}
		cout << endl;
		numDryAdiabats++;
		dryAdiabatEdgeCount.push_back(counter);

	}


	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING LCL computation - special dry adiabat (starts in ground ambient temp.)
	///////////////////////////////////////////////////////////////////////////////////////////////

	{
		P0 = soundingData[0].data[PRES];

		float theta = (soundingData[0].data[TEMP] + 273.15f)/* * powf(P0 / P0, k)*/;
		theta -= 273.15f;
		cout << "LCL Dry adiabat theta = " << theta << endl;

		for (int i = 0; i < soundingData.size(); i++) {

			float P = soundingData[i].data[PRES];

			float T = (theta + 273.15f) * pow((P / P0), k); // do not forget to use Kelvin
			T -= 273.15f; // convert back to Celsius

			y = getNormalizedPres(P);
			x = getNormalizedTemp(T, y);

			LCLDryAdiabatCurve.vertices.push_back(glm::vec2(x, y));
			//vertices.push_back(glm::vec2(x, y));
			cout << " | " << x << ", " << y << endl;
		}
		cout << endl;
		//numDryAdiabats++;
	}


	//TcNormalized = findIntersectionNaive(xaxis, TcDryAdiabat);
	TcNormalized = TcDryAdiabat.vertices[0]; // no need for intersection search here
	cout << "TcNormalized: " << TcNormalized.x << ", " << TcNormalized.y << endl;
	Tc = getDenormalizedCoords(TcNormalized);
	cout << "Tc: " << Tc.x << ", " << Tc.y << endl;

	// Check correctness by computing thetaCCL == Tc
	float thetaCCL = (CCL.x + 273.15f) * pow((P0 / CCL.y), k);
	thetaCCL -= 273.15f;
	cout << "THETA CCL = " << thetaCCL << endl;

	LCLNormalized = findIntersectionNaive(LCLDryAdiabatCurve, mixingCCL);
	LCL = getDenormalizedCoords(LCLNormalized);


	// Testing out profiles
	for (int i = 0; i < numProfiles; i++) {
		TcProfiles.push_back(Tc + glm::vec2((i + 1) * profileDelta, 0.0f));
		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(TcProfiles.back()), -2.0f)); // point
		float tint = (i + 1) * profileDelta;
		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		visualizationPoints.push_back(glm::vec3(tint, 0.0f, 0.0f)); // color	
	}

	cout << "NUMBER OF PROFILES = " << numProfiles << ", profileDelta = " << profileDelta << endl;
	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {

		dryAdiabatProfiles.push_back(Curve());

		counter = 0;

		P0 = soundingData[0].data[PRES];

		float theta = (TcProfiles[profileIndex].x + 273.15f)/* * powf(P0 / P0, k)*/;
		theta -= 273.15f;

		for (int i = 0; i < soundingData.size(); i++) {

			float P = soundingData[i].data[PRES];

			float T = (theta + 273.15f) * pow((P / P0), k); // do not forget to use Kelvin
			T -= 273.15f; // convert back to Celsius

			y = getNormalizedPres(P);
			x = getNormalizedTemp(T, y);

			dryAdiabatProfiles[profileIndex].vertices.push_back(glm::vec2(x, y));
			vertices.push_back(glm::vec2(x, y));
			counter++;
		}
		numDryAdiabats++;
		dryAdiabatEdgeCount.push_back(counter);

		CCLProfiles.push_back(getDenormalizedCoords(findIntersectionNaive(dryAdiabatProfiles[profileIndex], ambientCurve)));

		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(CCLProfiles.back()), -2.0f)); // point
		float tint = (profileIndex + 1) * profileDelta;
		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		visualizationPoints.push_back(glm::vec3(tint, 0.0f, 1.0f)); // color	

	}



	glGenVertexArrays(1, &TcVAO);
	glBindVertexArray(TcVAO);
	glGenBuffers(1, &TcVBO);
	glBindBuffer(GL_ARRAY_BUFFER, TcVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2), &TcNormalized, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);


	glGenVertexArrays(1, &dryAdiabatsVAO);
	glBindVertexArray(dryAdiabatsVAO);
	glGenBuffers(1, &dryAdiabatsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, dryAdiabatsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);




	///////////////////////////////////////////////////////////////////////////////////////
	// MOIST ADIABATS
	///////////////////////////////////////////////////////////////////////////////////////
	cout << "////////////////////////////////////////////////////" << endl;
	cout << "// MOIST ADIABATS" << endl;
	cout << "////////////////////////////////////////////////////" << endl;

	vertices.clear();


	float a = -6.14342f * 0.00001f;
	float b = 1.58927 * 0.001f;
	float c = -2.36418f;
	float d = 2500.79f;

	float g = -9.81f;
	numMoistAdiabats = 0;

	// Lv(T) = (aT^3 + bT^2 + cT + d) * 1000
	// Lv(T)	... latent heat of vaporisation/condensation

	//for (float T = MIN_TEMP; T <= MAX_TEMP; T++) {
	T = 30.0f; // for example - create first testing curvePtr
			   //float Lv = (a*T*T*T + b*T*T + c*T + d) * 1000.0f;

	float T_P0;
	float T_P1;
	float P1;
	float currP;

#define MOIST_ADIABAT_OPTION 5

#if MOIST_ADIABAT_OPTION == 0
	T_P0 = T;
	// LooooooooooooooP
	deltaP = 1.0f;
	for (float p = P0; p >= MIN_P; p -= deltaP) {
		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
		//P = soundingData[profileIndex].data[PRES];

		//T_P1 = ???
		//P1 = P;
		P1 = p;

		toKelvin(T_P0);
		toKelvin(T);


		// integral pres P0 az P1
		float integratedVal = 0.0f;
		//for (int profileIndex = P0 + 0.1f; profileIndex < P1; profileIndex += 0.1f) {
		//integratedVal += computePseudoadiabaticLapseRate(T, profileIndex);
		//integratedVal /= computeRho(T, profileIndex) * (-9.81f);
		//integratedVal += getMoistAdiabatIntegralVal(T_P0, profileIndex);
		//}
		//integratedVal += (getMoistAdiabatIntegralVal(T_P0, P0) + getMoistAdiabatIntegralVal(T_P0, P1)) / 2.0f;
		//integratedVal *= (P1 - P0) / CURVE_DELTA_P;

		//T_P1 = T_P0 + integratedVal;

		P0 *= 100.0f;
		P1 *= 100.0f;

		integratedVal = (P1 - P0) * getMoistAdiabatIntegralVal(T_P0, P1);
		T_P1 = T_P0 + integratedVal * 100.0f;

		//cout << "Integrated val = " << integratedVal << endl;

		P0 /= 100.0f;
		P1 /= 100.0f;

		toCelsius(T);
		toCelsius(T_P0);
		toCelsius(T_P1);

		y = getNormalizedPres(P0);
		x = getNormalizedTemp(T_P0, y);
		//x = (T_P0 - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
		vertices.push_back(glm::vec2(x, y));

		y = getNormalizedPres(P1);
		x = getNormalizedTemp(T_P1, y);
		//x = (T_P1 - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);

		vertices.push_back(glm::vec2(x, y));


		// jump to next
		P0 = P1;
		T_P0 = T_P1;
	}
#elif MOIST_ADIABAT_OPTION == 1 // Taken from existing code
	/////////////////////////////
	T_P0 = T;
	float ept = computeEquivalentTheta(getKelvin(T_P0), getKelvin(T_P0), 1000.0f);
	cout << "EPT = " << ept << endl;
	//P0 = 1000.0f;

	T_P0 = getSaturatedAirTemperature(ept, P0);
	////////////////////////////////
	for (int i = 0; i < soundingData.size(); i++) {
		P = soundingData[i].data[PRES];
		P1 = P;

		T_P1 = getSaturatedAirTemperature(ept, P1);

		toCelsius(T_P0);
		toCelsius(T_P1);

		y = getNormalizedPres(P0);
		x = getNormalizedTemp(T_P0, y);

		vertices.push_back(glm::vec2(x, y));

		y = getNormalizedPres(P1);
		x = getNormalizedTemp(T_P1, y);

		vertices.push_back(glm::vec2(x, y));

		toKelvin(T_P0);
		toKelvin(T_P1);

		// jump to next
		P0 = P1;
		T_P0 = T_P1;
	}
#elif MOIST_ADIABAT_OPTION == 2 // Bakhshaii iterative description
	//T = 24.0f;
	T_P0 = T;

	float e_0 = 6.112f;
	float e_s = e_0 * exp((17.67f * (T_P0 - 273.15f)) / T_P0 - 29.65f);
	float r_s = 0.622f * e_s / (P0 - e_s);
	float bApprox = 1.0;

	float Lv = (a*T*T*T + b*T*T + c*T + d) * 1000.0f;
	cout << "Lv = " << Lv << endl;

	float dTdP = (bApprox / P0) * ((R_d * T_P0 + Lv * r_s) / (1004.0f + (Lv * Lv * r_s * EPS * bApprox) / (R_d * T_P0 * T_P0)));

	for (int i = 0; i < soundingData.size(); i++) {
		currP = soundingData[i].data[PRES];
		P1 = currP;

		float deltaP = P1 - P0;


		toKelvin(T_P0);
		toKelvin(T_P1);
		toKelvin(T);
		///////
		float e_s = e_0 * exp((17.67f * (T_P0 - 273.15f)) / T_P0 - 29.65f);
		float r_s = 0.622f * e_s / (P0 - e_s);
		float Lv = (a*T_P0*T_P0*T_P0 + b*T_P0*T_P0 + c*T_P0 + d) * P0;

		float dTdP = (bApprox / P0) * ((R_d * T_P0 + Lv * r_s) / (1004.0f + (Lv * Lv * r_s * EPS * bApprox) / (R_d * T_P0 * T_P0)));

		/*float e_s = e_0 * exp((17.67f * (T - 273.15f)) / T - 29.65f);
		float r_s = 0.622f * e_s / (P0 - e_s);
		float dTdP = (bApprox / P0) * ((R_d * T + Lv * r_s) / (1004.0f + (Lv * Lv * r_s * EPS * bApprox) / (R_d * T * T)));*/
		//////////////////////

		T_P1 = T_P0 + deltaP * dTdP;


		toCelsius(T_P0);
		toCelsius(T_P1);
		toCelsius(T);
		//cout << "T_P0 = " << T_P0 << ", T_P1 = " << T_P1 << endl;
		//cout << "P0 = " << P0 << ", P1 = " << P1 << endl;

		y = getNormalizedPres(P0);
		x = getNormalizedTemp(T_P0, y);

		vertices.push_back(glm::vec2(x, y));


		P0 = P1;
		T_P0 = T_P1;

	}
#elif MOIST_ADIABAT_OPTION == 3 // Bakhshaii non-iterative approach
	T = 9.0f; // Celsius
	P = 800.0f; // 800hPa = 80kPa
	float theta_w = getWetBulbPotentialTemperature(T, P);
	cout << "THETA W = " << theta_w << endl;
	/*
	desired results:
	g1 = -1.26
	g2 = 53.24
	g3 = 0.58
	g4 = -8.84
	g5 = -25.99
	g6 = 0.15
	theta_w = 17.9 degC
	*/

	theta_w = 28.0f;
	P = 250.0f;
	T = getPseudoadiabatTemperature(theta_w, P);
	cout << "T = " << T << endl;

	for (float theta_w = MIN_TEMP; theta_w <= MAX_TEMP; theta_w += 10.0f) {
		for (int i = 0; i < soundingData.size(); i++) {
			P = soundingData[i].data[PRES];
			T = getPseudoadiabatTemperature(theta_w, P);
			y = getNormalizedPres(P);
			x = getNormalizedTemp(T, y);
			vertices.push_back(glm::vec2(x, y));
			numMoistAdiabats++;
		}
	}
#elif MOIST_ADIABAT_OPTION == 4 // pyMeteo implementation

	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += 5.0f) {
		//T = -10.0f;
		T = currT;
		T += 273.15f;
		float origT = T;

		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
		//	float p = soundingData[profileIndex].data[PRES];
		deltaP = 1.0f;
		for (float p = 1000.0f; p >= MIN_P; p -= deltaP) {
			p *= 100.0f;
			T -= dTdp_moist(T, p) * deltaP * 100.0f;
			p /= 100.0f;

			y = getNormalizedPres(p);
			x = getNormalizedTemp(getCelsius(T), y);
			vertices.push_back(glm::vec2(x, y));
		}
		T = origT;
		for (float p = 1000.0f; p <= MAX_P; p += deltaP) {
			p *= 100.0f;
			T += dTdp_moist(T, p) * deltaP * 100.0f;
			p /= 100.0f;

			y = getNormalizedPres(p);
			x = getNormalizedTemp(getCelsius(T), y);
			vertices.push_back(glm::vec2(x, y));
		}
		numMoistAdiabats++;

	}

#elif MOIST_ADIABAT_OPTION == 5 // pyMeteo implementation - with spacing

	for (float currT = MIN_TEMP; currT <= MAX_TEMP; currT += 5.0f) {
		//T = -10.0f;
		T = currT;
		T += 273.15f;
		float origT = T;

		//for (int profileIndex = 0; profileIndex < soundingData.size(); profileIndex++) {
		//	float p = soundingData[profileIndex].data[PRES];
		deltaP = 1.0f;
		int counter = 0;

		for (float p = 1000.0f; p <= MAX_P; p += deltaP) {
			p *= 100.0f;
			T += dTdp_moist(T, p) * deltaP * 100.0f;
			p /= 100.0f;

			if ((int)p % 25 == 0 && p != 1000.0f) {
				y = getNormalizedPres(p);
				x = getNormalizedTemp(getCelsius(T), y);
				vertices.push_back(glm::vec2(x, y));
				counter++;
			}
		}
		reverse(vertices.end() - counter, vertices.end()); // to draw continuous line
		T = origT;

		for (float p = 1000.0f; p >= MIN_P; p -= deltaP) {
			p *= 100.0f;
			T -= dTdp_moist(T, p) * deltaP * 100.0f;
			p /= 100.0f;

			if ((int)p % 25 == 0) {
				y = getNormalizedPres(p);
				x = getNormalizedTemp(getCelsius(T), y);
				vertices.push_back(glm::vec2(x, y));
				counter++;
			}
		}
		//cout << "Counter = " << counter << ", num isobars = " << numIsobars << endl;
		numMoistAdiabatEdges = counter;
		numMoistAdiabats++;
		moistAdiabatEdgeCount.push_back(counter);

	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING EL (regular) computation - special moist adiabat (goes through CCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	{
		int counter = 0;
		T = CCL.x + 273.15f;
		deltaP = 1.0f;
		for (float p = CCL.y; p >= MIN_P; p -= deltaP) {
			p *= 100.0f;
			T -= dTdp_moist(T, p) * deltaP * 100.0f;
			p /= 100.0f;

			if ((int)p % 25 == 0 || p == CCL.y) {
				y = getNormalizedPres(p);
				x = getNormalizedTemp(getCelsius(T), y);
				vertices.push_back(glm::vec2(x, y));
				moistAdiabat_CCL_EL.vertices.push_back(glm::vec2(x, y));
				counter++;
			}
		}
		numMoistAdiabats++;
		moistAdiabatEdgeCount.push_back(counter);



		///////////////////////////////////////////////////////////////////////////////////////////////
		// Find EL 
		///////////////////////////////////////////////////////////////////////////////////////////////

		reverse(moistAdiabat_CCL_EL.vertices.begin(), moistAdiabat_CCL_EL.vertices.end()); // temporary reverse for finding EL
		ELNormalized = findIntersectionNaive(moistAdiabat_CCL_EL, ambientCurve);
		cout << "EL (normalized): x = " << ELNormalized.x << ", y = " << ELNormalized.y << endl;
		EL = getDenormalizedCoords(ELNormalized);
		cout << "EL: T = " << EL.x << ", P = " << EL.y << endl;
		reverse(moistAdiabat_CCL_EL.vertices.begin(), moistAdiabat_CCL_EL.vertices.end()); // reverse back for the simulation

		visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
		visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	

		visualizationPoints.push_back(glm::vec3(ELNormalized, -2.0f)); // point
		visualizationPoints.push_back(glm::vec3(0.0f, 1.0f, 1.0f)); // color	
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// TESTING EL (orographic) computation - special moist adiabat (goes through LCL)
	///////////////////////////////////////////////////////////////////////////////////////////////
	{
		T = LCL.x + 273.15f;
		deltaP = 1.0f;
		for (float p = LCL.y; p >= MIN_P; p -= deltaP) {
			p *= 100.0f;
			T -= dTdp_moist(T, p) * deltaP * 100.0f;
			p /= 100.0f;

			if ((int)p % 25 == 0 || p == LCL.y) {
				y = getNormalizedPres(p);
				x = getNormalizedTemp(getCelsius(T), y);
				//vertices.push_back(glm::vec2(x, y));
				moistAdiabat_LCL_EL.vertices.push_back(glm::vec2(x, y));
			}
		}
		//numMoistAdiabats++;


		LFCNormalized = findIntersectionNaive(moistAdiabat_LCL_EL, ambientCurve);
		LFC = getDenormalizedCoords(LFCNormalized);

		reverse(moistAdiabat_LCL_EL.vertices.begin(), moistAdiabat_LCL_EL.vertices.end());

		OrographicELNormalized = findIntersectionNaive(moistAdiabat_LCL_EL, ambientCurve);
		OrographicEL = getDenormalizedCoords(OrographicELNormalized);

		reverse(moistAdiabat_LCL_EL.vertices.begin(), moistAdiabat_LCL_EL.vertices.end());


	}

	for (int profileIndex = 0; profileIndex < numProfiles; profileIndex++) {
		int counter = 0;
		moistAdiabatProfiles.push_back(Curve());
		T = CCLProfiles[profileIndex].x + 273.15f;
		deltaP = 1.0f;
		for (float p = CCLProfiles[profileIndex].y; p >= MIN_P; p -= deltaP) {
			p *= 100.0f;
			T -= dTdp_moist(T, p) * deltaP * 100.0f;
			p /= 100.0f;

			if ((int)p % 25 == 0 || p == CCLProfiles[profileIndex].y) {
				y = getNormalizedPres(p);
				x = getNormalizedTemp(getCelsius(T), y);
				vertices.push_back(glm::vec2(x, y));
				//moistAdiabat_LCL_EL.vertices.push_back(glm::vec2(x, y));
				moistAdiabatProfiles[profileIndex].vertices.push_back(glm::vec2(x, y));
				counter++;
			}
		}
		numMoistAdiabats++;
		moistAdiabatEdgeCount.push_back(counter);



		reverse(moistAdiabatProfiles[profileIndex].vertices.begin(), moistAdiabatProfiles[profileIndex].vertices.end());

		glm::vec2 tmp = findIntersectionNaive(moistAdiabatProfiles[profileIndex], ambientCurve);
		ELProfiles.push_back(getDenormalizedCoords(tmp));

		reverse(moistAdiabatProfiles[profileIndex].vertices.begin(), moistAdiabatProfiles[profileIndex].vertices.end());

		visualizationPoints.push_back(glm::vec3(getNormalizedCoords(ELProfiles.back()), -2.0f)); // point
		float tint = (profileIndex + 1) * profileDelta;
		rangeToRange(tint, 0.0f, convectiveTempRange, 0.0f, 1.0f);
		visualizationPoints.push_back(glm::vec3(tint, 1.0f, 1.0f)); // color	
	}

#endif

	//numMoistAdiabats++;

	if (!vertices.empty()) {

		glGenVertexArrays(1, &moistAdiabatsVAO);
		glBindVertexArray(moistAdiabatsVAO);
		glGenBuffers(1, &moistAdiabatsVBO);
		glBindBuffer(GL_ARRAY_BUFFER, moistAdiabatsVBO);

		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

		glBindVertexArray(0);
	}




	// trying out stuff
	P = 432.2f;
	float normP = getNormalizedPres(P);
	cout << "Pressure = " << P << ", normalized pressure = " << normP << endl;
	visualizationPoints.push_back(glm::vec3(ambientCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
	visualizationPoints.push_back(glm::vec3(1.0f, 0.0f, 0.0f)); // color

	visualizationPoints.push_back(glm::vec3(dewpointCurve.getIntersectionWithIsobar(normP), 0.0f)); // point
	visualizationPoints.push_back(glm::vec3(0.0f, 0.0f, 1.0f)); // color


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

	mainParameterPoints.push_back(glm::vec3(CCLNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));
	mainParameterPoints.push_back(glm::vec3(TcNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(ELNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(LCLNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(LFCNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));

	mainParameterPoints.push_back(glm::vec3(OrographicELNormalized, 0.0f));
	mainParameterPoints.push_back(glm::vec3(0.0f));


	glGenVertexArrays(1, &mainParameterPointsVAO);
	glBindVertexArray(mainParameterPointsVAO);
	glGenBuffers(1, &mainParameterPointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, mainParameterPointsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * mainParameterPoints.size(), &mainParameterPoints[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)(sizeof(glm::vec3)));

	glBindVertexArray(0);


	// QUAD
	glGenVertexArrays(1, &overlayDiagramVAO);
	glGenBuffers(1, &overlayDiagramVBO);
	glBindVertexArray(overlayDiagramVAO);
	glBindBuffer(GL_ARRAY_BUFFER, overlayDiagramVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glBindVertexArray(0);

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


	GLfloat lineWidthRange[2] = { 0.0f, 0.0f };
	glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, lineWidthRange);
	// Maximum supported line width is in lineWidthRange[1].
	cout << lineWidthRange[0] << " , " << lineWidthRange[1] << endl;
}