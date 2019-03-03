#include "STLPSimulator.h"

#include <iostream>

#include "STLPUtils.h"
#include "Utils.h"
#include "ShaderManager.h"

using namespace std;

STLPSimulator::STLPSimulator(VariableManager *vars, STLPDiagram *stlpDiagram) : vars(vars), stlpDiagram(stlpDiagram) {
	groundHeight = stlpDiagram->P0;
	boxTopHeight = groundHeight + simulationBoxHeight;

	layerVisShader = ShaderManager::getShaderPtr("singleColorAlpha");
	
	initBuffers();

	stlpDiagram->particlePoints.reserve(maxNumParticles);
	stlpDiagram->particlePoints.push_back(glm::vec2(0.0f));

}


STLPSimulator::~STLPSimulator() {
}

void STLPSimulator::initBuffers() {

	glGenVertexArrays(1, &particlesVAO);
	glBindVertexArray(particlesVAO);
	glGenBuffers(1, &particlesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);

	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);



	vector<glm::vec3> vertices;

	glGenVertexArrays(1, &CCLLevelVAO);
	glBindVertexArray(CCLLevelVAO);
	glGenBuffers(1, &CCLLevelVBO);
	glBindBuffer(GL_ARRAY_BUFFER, CCLLevelVBO);

	float altitude;
	altitude = getAltitudeFromPressure(stlpDiagram->CCL.y);
	mapToSimulationBox(altitude);
	vertices.push_back(glm::vec3(0.0f, altitude, 0.0f));
	vertices.push_back(glm::vec3(0.0f, altitude, vars->latticeDepth));
	vertices.push_back(glm::vec3(vars->latticeWidth, altitude, vars->latticeDepth));
	vertices.push_back(glm::vec3(vars->latticeWidth, altitude, 0.0f));


	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 4, &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);

	
	vertices.clear();

	glGenVertexArrays(1, &ELLevelVAO);
	glBindVertexArray(ELLevelVAO);
	glGenBuffers(1, &ELLevelVBO);
	glBindBuffer(GL_ARRAY_BUFFER, ELLevelVBO);

	altitude = getAltitudeFromPressure(stlpDiagram->EL.y);
	mapToSimulationBox(altitude);
	vertices.push_back(glm::vec3(0.0f, altitude, 0.0f));
	vertices.push_back(glm::vec3(0.0f, altitude, vars->latticeDepth));
	vertices.push_back(glm::vec3(vars->latticeWidth, altitude, vars->latticeDepth));
	vertices.push_back(glm::vec3(vars->latticeWidth, altitude, 0.0f));

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 4, &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);
	
}

void STLPSimulator::doStep() {


	// quick testing
		
	while (numParticles < maxNumParticles) {
		stlpDiagram->particlePoints.push_back(glm::vec2(0.0f));

		generateParticle();
	}
	while (numParticles > maxNumParticles) {
		particles.pop_back();
		particlePositions.pop_back();
		stlpDiagram->particlePoints.pop_back();
		numParticles--;
	}

	//if (numParticles < MAX_PARTICLE_COUNT) {
	//	generateParticle();
	//}
	


	for (int i = 0; i < numParticles; i++) {
		if (particles[i].pressure > stlpDiagram->CCLProfiles[particles[i].profileIndex].y) { // if P_i > P_{CCL_i}

			//cout << "===== DRY LIFT STEP =======================================================================================" << endl;

			//printf("pressure = %0.2f\n", particles[i].pressure);

			// l <- isobar line

			// equation 3.14 - theta = T_{c_i}
			float T = (stlpDiagram->TcProfiles[particles[i].profileIndex].x + 273.15f) * pow((particles[i].pressure / stlpDiagram->soundingData[0].data[PRES]), 0.286f); // do not forget to use Kelvin
			T -= 273.15f; // convert back to Celsius

			// find intersection of isobar at P_i with C_a and C_d (ambient and dewpoint sounding curves)
			float normP = stlpDiagram->getNormalizedPres(particles[i].pressure);
			glm::vec2 ambientIntersection = stlpDiagram->ambientCurve.getIntersectionWithIsobar(normP);
			glm::vec2 dryAdiabatIntersection = stlpDiagram->dryAdiabatProfiles[particles[i].profileIndex].getIntersectionWithIsobar(normP);



			float ambientTemp = stlpDiagram->getDenormalizedTemp(ambientIntersection.x, normP);
			float particleTemp = stlpDiagram->getDenormalizedTemp(dryAdiabatIntersection.x, normP);

			stlpDiagram->particlePoints[i] = stlpDiagram->getNormalizedCoords(particleTemp, particles[i].pressure);

			/*printf("ambientTemp [deg C] = %0.2f\n", ambientTemp);
			printf("particleTemp [deg C] = %0.2f\n", particleTemp);*/

			toKelvin(ambientTemp);
			toKelvin(particleTemp);

			float ambientTheta = computeThetaFromAbsoluteK(ambientTemp, particles[i].pressure);
			//float particleTheta = computeThetaFromAbsoluteK(getKelvin(T), particles[i].pressure);
			float particleTheta = computeThetaFromAbsoluteK(particleTemp, particles[i].pressure);

			float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;

			//printf("a = %0.2f\n", a);


			if (!usePrevVelocity) {
				particles[i].velocity.y = 0.0f;
			}
			particles[i].velocity.y = particles[i].velocity.y + a * delta_t;
			float deltaY = particles[i].velocity.y * delta_t + 0.5f * a * delta_t * delta_t;

			//printf("delta y = %0.2f\n", deltaY);



			particles[i].position.y += deltaY;
			particles[i].updatePressureVal();


			if (simulateWind) {

				glm::vec2 windDeltas = stlpDiagram->getWindDeltasFromAltitude(particles[i].position.y); // this is in meters per second
																										// we need to map it to our system
				windDeltas /= GRID_WIDTH; // just testing
				//rangeToRange(windDeltas.x, )

				particles[i].position.x += windDeltas.x;
				particles[i].position.z += windDeltas.y;
			}



		} else {

			// l <- isobar line

			// find intersection of isobar at P_i with C_a and C_d (ambient and dewpoint sounding curves)
			float normP = stlpDiagram->getNormalizedPres(particles[i].pressure);
			glm::vec2 ambientIntersection = stlpDiagram->ambientCurve.getIntersectionWithIsobar(normP);
			glm::vec2 moistAdiabatIntersection = stlpDiagram->moistAdiabatProfiles[particles[i].profileIndex].getIntersectionWithIsobar(normP);

			float ambientTemp = stlpDiagram->getDenormalizedTemp(ambientIntersection.x, normP);
			float particleTemp = stlpDiagram->getDenormalizedTemp(moistAdiabatIntersection.x, normP);

			stlpDiagram->particlePoints[i] = stlpDiagram->getNormalizedCoords(particleTemp, particles[i].pressure);


			toKelvin(ambientTemp);
			toKelvin(particleTemp);

			float ambientTheta = computeThetaFromAbsoluteK(ambientTemp, particles[i].pressure);
			float particleTheta = computeThetaFromAbsoluteK(particleTemp, particles[i].pressure);


			float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;

			if (!usePrevVelocity) {
				particles[i].velocity.y = 0.0f;
			}
			particles[i].velocity.y = particles[i].velocity.y + a * delta_t;
			float deltaY = particles[i].velocity.y * delta_t + 0.5f * a * delta_t * delta_t;

			particles[i].position.y += deltaY;
			particles[i].updatePressureVal();



			if (simulateWind) {
				glm::vec2 windDeltas = stlpDiagram->getWindDeltasFromAltitude(particles[i].position.y); // this is in meters per second
				// we need to map it to our system
				windDeltas /= GRID_WIDTH; // just testing

				particles[i].position.x += windDeltas.x;
				particles[i].position.z += windDeltas.y;
			}

		}

		// hack
		glm::vec3 tmpPos = particles[i].position;
		//rangeToRange(tmpPos.y, 0.0f, 15000.0f, 0.0f, GRID_HEIGHT);

		mapToSimulationBox(tmpPos.y);
		//rangeToRange()
		//printf("height = %0.2f\n", tmpPos.y);


		particlePositions[i] = tmpPos;

	}


}


// naive solution
void STLPSimulator::resetSimulation() {

	cout << "Resetting simulation" << endl;
	particles.clear();
	particlePositions.clear();
	numParticles = 0;

}

void STLPSimulator::generateParticle() {

	float randx = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->width - 2.0f)));
	float randz = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->height - 2.0f)));

	// let's use small square 
	//float randx = (float)(rand() / (float)(RAND_MAX / ((float)GRID_WIDTH / 10.0f - 2.0f)));
	//float randz = (float)(rand() / (float)(RAND_MAX / ((float)GRID_DEPTH / 10.0f - 2.0f)));

	// interpolate
	int leftx = (int)randx;
	int rightx = leftx + 1;
	int leftz = (int)randz;
	int rightz = leftz + 1;

	// leftx and leftz cannot be < 0 and rightx and rightz cannot be >= GRID_WIDTH or GRID_DEPTH
	float xRatio = randx - leftx;
	float zRatio = randz - leftz;

	float y1 = heightMap->data[leftx][leftz];
	float y2 = heightMap->data[leftx][rightz];
	float y3 = heightMap->data[rightx][leftz];
	float y4 = heightMap->data[rightx][rightz];

	float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
	float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;

	float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;

	particlePositions.push_back(glm::vec3(randx, y, randz));

	//rangeToRange(y, 0.0f, GRID_HEIGHT, 0.0f, 15000.0f);
	cout << "height (sim box) = " << y << endl;
	mapFromSimulationBox(y);
	cout << "height (real)    = " << y << endl;
	//cout << y << endl;

	//y = 1500.0f;

	Particle p;
	p.position = glm::vec3(randx, y, randz);
	p.velocity = glm::vec3(0.0f);
	//p.updatePressureVal();
	//p.convectiveTemperature = stlpDiagram->Tc.x;
	p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
	//p.convectiveTemperature = stlpDiagram->TcProfiles[p.profileIndex].x;

	//cout << "Pressure at " << y << " is " << p.pressure << endl;

	//float tmpP = 943.0f;
	//float tmpz = getAltitudeFromPressure(tmpP);
	//cout << "Altitude at pressure " << tmpP << " is " << tmpz << endl;
	//tmpP = getPressureFromAltitude(tmpz);
	//cout << "Pressure at altitude " << tmpz << " is " << tmpP << endl;

	//tmpP = 100.0f;
	//tmpz = getAltitudeFromPressure(tmpP);
	//cout << "Altitude at pressure " << tmpP << " is " << tmpz << endl;

	//p.position.y = getAltitudeFromPressure(stlpDiagram->soundingData[0].data[PRES]);

	p.updatePressureVal();

	particles.push_back(p);
	numParticles++;


	/*int randx = rand() % (GRID_WIDTH - 1);
	int randz = rand() % (GRID_DEPTH - 1);

	float y = heightMap->data[randx][randz];

	particlePositions.push_back(glm::vec3(randx, y, randz));
	numParticles++;*/
}

void STLPSimulator::draw(ShaderProgram &particlesShader) {
	//heightMap->draw();

	glUseProgram(particlesShader.id);

	glPointSize(1.0f);
	particlesShader.setVec4("color", glm::vec4(1.0f, 0.4f, 1.0f, 1.0f));

	glBindVertexArray(particlesVAO);

	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &testParticle.position[0], GL_DYNAMIC_DRAW);

	//glDrawArrays(GL_POINTS, 0, numParticles);
	glDrawArrays(GL_POINTS, 0, numParticles);

	if (showCCLLevelLayer || showELLevelLayer) {
		GLboolean cullFaceEnabled;
		glGetBooleanv(GL_CULL_FACE, &cullFaceEnabled);
		glDisable(GL_CULL_FACE);

		layerVisShader->use();

		if (showCCLLevelLayer) {
			layerVisShader->setVec4("u_Color", glm::vec4(1.0f, 0.0f, 0.0f, 0.2f));

			glBindVertexArray(CCLLevelVAO);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		}

		if (showELLevelLayer) {
			layerVisShader->setVec4("u_Color", glm::vec4(0.0f, 1.0f, 0.0f, 0.2f));


			glBindVertexArray(ELLevelVAO);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		}

		if (cullFaceEnabled) {
			glEnable(GL_CULL_FACE);
		}
	}


}

void STLPSimulator::initParticles() {
}

void STLPSimulator::mapToSimulationBox(float & val) {
	rangeToRange(val, groundHeight, boxTopHeight, 0.0f, vars->latticeHeight);
}

void STLPSimulator::mapFromSimulationBox(float & val) {
	rangeToRange(val, 0.0f, vars->latticeHeight, groundHeight, boxTopHeight);
}
