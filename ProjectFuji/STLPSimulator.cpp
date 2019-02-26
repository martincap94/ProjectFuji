#include "STLPSimulator.h"

#include <iostream>

#include "STLPUtils.h"
#include "Utils.h"

using namespace std;

STLPSimulator::STLPSimulator() {
	initBuffers();
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

}

void STLPSimulator::doStep() {

	if (!testing) {

		// quick testing
		while (numParticles < maxNumParticles) {
			generateParticle();
		}
		while (numParticles >= maxNumParticles) {
			particles.pop_back();
			particlePositions.pop_back();
			numParticles--;
		}

		//if (numParticles < MAX_PARTICLE_COUNT) {
		//	generateParticle();
		//}
	}
	
	if (testing) {
		// TESTING PARTICLE MOTION
		if (testParticle.pressure > stlpDiagram->CCL.y) { // if P_i > P_{CCL_i}

			cout << "===== DRY LIFT STEP =======================================================================================" << endl;


			// l <- isobar line

			// equation 3.14 - theta = T_{c_i}
			float T = (testParticle.convectiveTemperature + 273.15f) * pow((testParticle.pressure / stlpDiagram->soundingData[0].data[PRES]), 0.286f); // do not forget to use Kelvin
			T -= 273.15f; // convert back to Celsius

			// find intersection of isobar at P_i with C_a and C_d (ambient and dewpoint sounding curves)
			float normP = stlpDiagram->getNormalizedPres(testParticle.pressure);
			glm::vec2 ambientIntersection = stlpDiagram->ambientCurve.getIntersectionWithIsobar(normP);
			glm::vec2 dewpointIntersection = stlpDiagram->dewpointCurve.getIntersectionWithIsobar(normP);

			float ambientTemp = stlpDiagram->getDenormalizedTemp(ambientIntersection.x, normP);
			float dewpointTemp = stlpDiagram->getDenormalizedTemp(dewpointIntersection.x, normP);

			stlpDiagram->setVisualizationPoint(glm::vec3(ambientTemp, testParticle.pressure, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 1, false);
			stlpDiagram->setVisualizationPoint(glm::vec3(dewpointTemp, testParticle.pressure, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2, false);

			toKelvin(ambientTemp);
			toKelvin(dewpointTemp);

			cout << "ambient temp [K] = " << ambientTemp << endl;
			cout << "dewpoint temp [K] = " << dewpointTemp << endl;


			float ambientTheta = computeThetaFromAbsoluteK(ambientTemp, testParticle.pressure);
			float dewpointTheta = computeThetaFromAbsoluteK(dewpointTemp, testParticle.pressure);
			float particleTheta = computeThetaFromAbsoluteK(getKelvin(T), testParticle.pressure);

			cout << "convective temp [K] = " << getKelvin(testParticle.convectiveTemperature) << endl;
			cout << "ambient theta [K] = " << ambientTheta << ", dewpoint theta [K] = " << dewpointTheta << endl;

			//float a = -9.81f * (dewpointTheta - ambientTheta) / ambientTheta; // is this correct? -> is this a mistake in Duarte's thesis? BEWARE: C_d is dry adiabat, not dewpoint!!! -> misleading notation in Duarte's thesis
			//float a = 9.81f * (getKelvin(testParticle.convectiveTemperature) - ambientTheta) / ambientTheta; -> this is incorrect (?)

			cout << "Particle theta [K] = " << particleTheta << endl;

			//toCelsius(ambientTheta);
			//toCelsius(particleTheta);

			float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;

			cout << "ACCELERATION a = " << a << endl;

			testParticle.velocity.y = testParticle.velocity.y + a * delta_t;
			float deltaY = testParticle.velocity.y + 0.5f * a * delta_t * delta_t;

			testParticle.position.y += deltaY;

			printf("Particle velocity = (%f, %f, %f)\n", testParticle.velocity.x, testParticle.velocity.y, testParticle.velocity.z);
			cout << "Particle height = " << testParticle.position.y << endl;

			testParticle.updatePressureVal();

			stlpDiagram->setVisualizationPoint(glm::vec3(T, testParticle.pressure, 0.0f), glm::vec3(0.0f, 1.0f, 0.3f), 0, false);



		} else {

			cout << "===== MOIST LIFT STEP ============================================================================" << endl;


			float normP = stlpDiagram->getNormalizedPres(testParticle.pressure);
			glm::vec2 ambientIntersection = stlpDiagram->ambientCurve.getIntersectionWithIsobar(normP);
			glm::vec2 moistAdiabatIntersection = stlpDiagram->moistAdiabat_CCL_EL.getIntersectionWithIsobar(normP);

			float ambientTemp = stlpDiagram->getDenormalizedTemp(ambientIntersection.x, normP);
			float particleTemp = stlpDiagram->getDenormalizedTemp(moistAdiabatIntersection.x, normP);

			stlpDiagram->setVisualizationPoint(glm::vec3(ambientTemp, testParticle.pressure, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 1, false);
			stlpDiagram->setVisualizationPoint(glm::vec3(particleTemp, testParticle.pressure, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2, false);


			cout << "ambient temp = " << ambientTemp << endl;
			cout << "particle temp = " << particleTemp << endl;

			toKelvin(ambientTemp);
			toKelvin(particleTemp);

			float ambientTheta = computeThetaFromAbsoluteK(ambientTemp, testParticle.pressure);
			float particleTheta = computeThetaFromAbsoluteK(particleTemp, testParticle.pressure);

			cout << "convective temp [K] = " << getKelvin(testParticle.convectiveTemperature) << endl;
			cout << "ambient theta [K] = " << ambientTheta << ", particle theta [K] = " << particleTheta << endl;

			//float a = -9.81f * (dewpointTheta - ambientTheta) / ambientTheta; // is this correct? -> is this a mistake in Duarte's thesis? BEWARE: C_d is dry adiabat, not dewpoint!!! -> misleading notation in Duarte's thesis
			//float a = 9.81f * (getKelvin(testParticle.convectiveTemperature) - ambientTheta) / ambientTheta; -> this is incorrect (?)

			cout << "Particle theta [K] = " << particleTheta << endl;


			//toCelsius(ambientTheta);
			//toCelsius(particleTheta);

			float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;

			cout << "ACCELERATION a = " << a << endl;

			testParticle.velocity.y = testParticle.velocity.y + a * delta_t;
			float deltaY = testParticle.velocity.y + 0.5f * a * delta_t * delta_t;

			testParticle.position.y += deltaY;

			printf("Particle velocity = (%f, %f, %f)\n", testParticle.velocity.x, testParticle.velocity.y, testParticle.velocity.z);
			cout << "Particle height = " << testParticle.position.y << endl;


			testParticle.updatePressureVal();

			stlpDiagram->setVisualizationPoint(glm::vec3(getCelsius(particleTemp), testParticle.pressure, 0.0f), glm::vec3(0.0f, 1.0f, 0.3f), 0, false);


		}

		// hack

		glm::vec3 tmpPos = testParticle.position;
		rangeToRange(tmpPos.y, 0.0f, 15000.0f, 0.0f, GRID_HEIGHT); // 10 km

		particlePositions[0] = tmpPos;
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// NEW --- multiple particles -----
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if (!testing) {
		for (int i = 0; i < numParticles; i++) {
			if (particles[i].pressure > stlpDiagram->CCL.y) { // if P_i > P_{CCL_i}

				//cout << "===== DRY LIFT STEP =======================================================================================" << endl;


				// l <- isobar line

				// equation 3.14 - theta = T_{c_i}
				float T = (particles[i].convectiveTemperature + 273.15f) * pow((particles[i].pressure / stlpDiagram->soundingData[0].data[PRES]), 0.286f); // do not forget to use Kelvin
				T -= 273.15f; // convert back to Celsius

							  // find intersection of isobar at P_i with C_a and C_d (ambient and dewpoint sounding curves)
				float normP = stlpDiagram->getNormalizedPres(particles[i].pressure);
				glm::vec2 ambientIntersection = stlpDiagram->ambientCurve.getIntersectionWithIsobar(normP);
				glm::vec2 dewpointIntersection = stlpDiagram->dewpointCurve.getIntersectionWithIsobar(normP);

				float ambientTemp = stlpDiagram->getDenormalizedTemp(ambientIntersection.x, normP);
				float dewpointTemp = stlpDiagram->getDenormalizedTemp(dewpointIntersection.x, normP);


				toKelvin(ambientTemp);
				toKelvin(dewpointTemp);

				float ambientTheta = computeThetaFromAbsoluteK(ambientTemp, particles[i].pressure);
				float dewpointTheta = computeThetaFromAbsoluteK(dewpointTemp, particles[i].pressure);
				float particleTheta = computeThetaFromAbsoluteK(getKelvin(T), particles[i].pressure);

				//float a = -9.81f * (dewpointTheta - ambientTheta) / ambientTheta; // is this correct? -> is this a mistake in Duarte's thesis? BEWARE: C_d is dry adiabat, not dewpoint!!! -> misleading notation in Duarte's thesis
				//float a = 9.81f * (getKelvin(particles[i].convectiveTemperature) - ambientTheta) / ambientTheta; -> this is incorrect (?)




				float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;

				particles[i].velocity.y = particles[i].velocity.y + a * delta_t;
				float deltaY = particles[i].velocity.y + 0.5f * a * delta_t * delta_t;

				particles[i].position.y += deltaY;
				particles[i].updatePressureVal();



			} else {

				// l <- isobar line

				// find intersection of isobar at P_i with C_a and C_d (ambient and dewpoint sounding curves)
				float normP = stlpDiagram->getNormalizedPres(particles[i].pressure);
				glm::vec2 ambientIntersection = stlpDiagram->ambientCurve.getIntersectionWithIsobar(normP);
				glm::vec2 moistAdiabatIntersection = stlpDiagram->moistAdiabatProfiles[particles[i].profileIndex].getIntersectionWithIsobar(normP);

				float ambientTemp = stlpDiagram->getDenormalizedTemp(ambientIntersection.x, normP);
				float particleTemp = stlpDiagram->getDenormalizedTemp(moistAdiabatIntersection.x, normP);

				/*stlpDiagram->setVisualizationPoint(glm::vec3(ambientTemp, particles[i].pressure, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), 1, false);
				stlpDiagram->setVisualizationPoint(glm::vec3(particleTemp, particles[i].pressure, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2, false);*/


				toKelvin(ambientTemp);
				toKelvin(particleTemp);

				float ambientTheta = computeThetaFromAbsoluteK(ambientTemp, particles[i].pressure);
				float particleTheta = computeThetaFromAbsoluteK(particleTemp, particles[i].pressure);

				//float a = -9.81f * (dewpointTheta - ambientTheta) / ambientTheta; // is this correct? -> is this a mistake in Duarte's thesis? BEWARE: C_d is dry adiabat, not dewpoint!!! -> misleading notation in Duarte's thesis
				//float a = 9.81f * (getKelvin(particles[i].convectiveTemperature) - ambientTheta) / ambientTheta; -> this is incorrect (?)

				float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;


				particles[i].velocity.y = particles[i].velocity.y + a * delta_t;
				float deltaY = particles[i].velocity.y + 0.5f * a * delta_t * delta_t;

				particles[i].position.y += deltaY;
				particles[i].updatePressureVal();


			}

			// hack
			glm::vec3 tmpPos = particles[i].position;
			rangeToRange(tmpPos.y, 0.0f, 15000.0f, 0.0f, GRID_HEIGHT); // 10 km

			particlePositions[i] = tmpPos;

		}
	}

}


// naive solution
void STLPSimulator::resetSimulation() {

	cout << "Resetting simulation" << endl;
	particles.clear();
	particlePositions.clear();
	numParticles = 0;

}

void STLPSimulator::generateParticle(bool setTestParticle) {

	float randx = (float)(rand() / (float)(RAND_MAX / ((float)GRID_WIDTH - 2.0f)));
	float randz = (float)(rand() / (float)(RAND_MAX / ((float)GRID_DEPTH - 2.0f)));

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


	Particle p;
	p.position = glm::vec3(randx, y, randz);
	p.velocity = glm::vec3(0.0f);
	//p.updatePressureVal();
	//p.convectiveTemperature = stlpDiagram->Tc.x;
	p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
	p.convectiveTemperature = stlpDiagram->TcProfiles[p.profileIndex].x;

	//cout << "Pressure at " << y << " is " << p.pressure << endl;

	//float tmpP = 943.0f;
	//float tmpz = getAltitudeFromPressure(tmpP);
	//cout << "Altitude at pressure " << tmpP << " is " << tmpz << endl;
	//tmpP = getPressureFromAltitude(tmpz);
	//cout << "Pressure at altitude " << tmpz << " is " << tmpP << endl;

	//tmpP = 100.0f;
	//tmpz = getAltitudeFromPressure(tmpP);
	//cout << "Altitude at pressure " << tmpP << " is " << tmpz << endl;

	p.position.y = getAltitudeFromPressure(stlpDiagram->soundingData[0].data[PRES]);
	p.updatePressureVal();

	particles.push_back(p);
	particlePositions.push_back(glm::vec3(randx, y, randz));
	numParticles++;

	if (setTestParticle) {
		p.convectiveTemperature = stlpDiagram->Tc.x;
		p.updatePressureVal();
		testParticle = p;
		stlpDiagram->setVisualizationPoint(glm::vec3(testParticle.convectiveTemperature, testParticle.pressure, 0.0f), glm::vec3(0.0f, 1.0f, 0.3f), 0, false);
	}

	/*int randx = rand() % (GRID_WIDTH - 1);
	int randz = rand() % (GRID_DEPTH - 1);

	float y = heightMap->data[randx][randz];

	particlePositions.push_back(glm::vec3(randx, y, randz));
	numParticles++;*/
}

void STLPSimulator::draw(ShaderProgram &particlesShader) {
	heightMap->draw();

	glUseProgram(particlesShader.id);

	glPointSize(1.0f);
	particlesShader.setVec4("color", glm::vec4(1.0f, 0.4f, 1.0f, 1.0f));

	glBindVertexArray(particlesVAO);

	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &testParticle.position[0], GL_DYNAMIC_DRAW);

	//glDrawArrays(GL_POINTS, 0, numParticles);
	glDrawArrays(GL_POINTS, 0, numParticles);

}

void STLPSimulator::initParticles() {
	generateParticle(true); // testing particle for dry and moist lift
}
