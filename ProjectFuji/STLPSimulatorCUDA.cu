#include "STLPSimulatorCUDA.h"

#include "ShaderManager.h"
#include "STLPUtils.h"
#include "Utils.h"
#include "HeightMap.h"
#include "CUDAUtils.cuh"
#include "ParticleSystem.h"

#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"




__constant__ int d_const_numProfiles;
__constant__ float d_const_maxP;
__constant__ float d_const_delta_t;

__constant__ float d_const_groundHeight;
__constant__ float d_const_boxTopHeight;
__constant__ float d_const_latticeHeight;

//__constant__ glm::vec2 *d_const_ambientTempCurve;
//
//__constant__ glm::vec2 *d_const_dryAdiabatProfiles;
//__constant__ int *d_const_dryAdiabatOffsets; // since each dry adiabat can have different amount of vertices
//
//__constant__ glm::vec2 *d_const_moistAdiabatProfiles;
//__constant__ int *d_const_moistAdiabatOffsets; // since each moist adiabat can have different amount of vertices
//
//__constant__ glm::vec2 *d_const_CCLProfiles;
//__constant__ glm::vec2 *d_const_TcProfiles;




__device__ float getNormalizedTemp(float T, float y) {
	return (T - MIN_TEMP) / (MAX_TEMP - MIN_TEMP) + (1.0f - y);
}

__device__ float getNormalizedPres(float P) {
	return ((log10f(P) - log10f(MIN_P)) / (log10f(d_const_maxP) - log10f(MIN_P)));
}

__device__ float getDenormalizedTemp(float x, float y) {
	return (x + y - 1.0f) * (MAX_TEMP - MIN_TEMP) + MIN_TEMP;
}

__device__ float getDenormalizedPres(float y) {
	return powf(10.0f, y * (log10f(d_const_maxP) - log10f(MIN_P)) + log10f(MIN_P));
}

__device__ glm::vec2 getNormalizedCoords(glm::vec2 coords) {
	glm::vec2 res;
	res.y = getNormalizedPres(coords.y);
	res.x = getNormalizedTemp(coords.x, res.y);
	return res;
}

__device__ glm::vec2 getDenormalizedCoords(glm::vec2 coords) {
	glm::vec2 res;
	res.x = getDenormalizedTemp(coords.x, coords.y);
	res.y = getDenormalizedPres(coords.y);
	return res;
}

__device__ glm::vec2 getNormalizedCoords(float T, float P) {
	return getNormalizedCoords(glm::vec2(T, P));
}

__device__ glm::vec2 getDenormalizedCoords(float x, float y) {
	return getDenormalizedCoords(glm::vec2(x, y));
}

__device__ float computeThetaFromAbsoluteK_dev(float T, float P, float P0 = 1000.0f) {
	float tmp = (P == P0) ? 1.0f : pow(P0 / P, k_ratio);
	return T * tmp;
}

__device__ float getKelvin_dev(float T) {
	return T + 273.15f;
}

__device__ float getCelsius_dev(float T) {
	return T - 273.15f;
}

__device__ void toKelvin_dev(float &T) {
	T += 273.15f;
}

__device__ void toCelsius_dev(float &T) {
	T -= 273.15f;
}

__device__ float getPressureVal_dev(float height) {
	// based on CRC Handbook of Chemistry and Physics
	return pow(((44331.514f - height) / 11880.516f), 1 / 0.1902632f);
}

__device__ void normalizeFromRange_dev(float &val, float min, float max) {
	val = (val - min) / (max - min);
}

__device__ void rangeToRange_dev(float &val, float origMin, float origMax, float newMin, float newMax) {
	normalizeFromRange_dev(val, origMin, origMax);
	val *= (newMax - newMin);
	val += newMin;
}

__device__ void mapToSimulationBox_dev(float & val) {
	rangeToRange_dev(val, d_const_groundHeight, d_const_boxTopHeight, 0.0f, d_const_latticeHeight);
}

__device__ void mapFromSimulationBox_dev(float & val) {
	rangeToRange_dev(val, 0.0f, d_const_latticeHeight, d_const_groundHeight, d_const_boxTopHeight);
}



__device__ glm::vec2 getIntersectionWithIsobar(glm::vec2 *curveVertices, int numCurveVertices, float normP) {
	// naively search for correct interval - better solutions are: binary search and direct indexation using (non-normalized) pressure - needs better design
	for (int i = 0; i < numCurveVertices - 1; i += 1) {
		if (curveVertices[i + 1].y > normP) {
			continue;
		}
		if (curveVertices[i + 1].y <= normP) {
			float t = (normP - curveVertices[i + 1].y) / (curveVertices[i].y - curveVertices[i + 1].y);
			float normalizedTemperature = t * curveVertices[i].x + (1.0f - t) * curveVertices[i + 1].x;
			return glm::vec2(normalizedTemperature, normP);
		}
	}
	return glm::vec2();
}



__global__ void simulationStepKernel(glm::vec3 *particleVertices, int numParticles, float *verticalVelocities, int *profileIndices, float *particlePressures, glm::vec2 *ambientTempCurve, int numAmbientTempCurveVertices, glm::vec2 *dryAdiabatProfiles, glm::ivec2 *dryAdiabatOffsetsAndLengths, glm::vec2 *moistAdiabatProfiles, glm::ivec2 *moistAdiabatOffsetsAndLengths, glm::vec2 *CCLProfiles, glm::vec2 *TcProfiles) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {

		if (particlePressures[idx] > CCLProfiles[profileIndices[idx]].y) {

			//printf("| pressure = %0.2f\n", particlePressures[idx]);
			//particleVertices[idx].y += 0.1f;
			float normP = getNormalizedPres(particlePressures[idx]);
			glm::vec2 ambientIntersection = getIntersectionWithIsobar(ambientTempCurve, numAmbientTempCurveVertices, normP);
			glm::vec2 dryAdiabatIntersection = getIntersectionWithIsobar(&dryAdiabatProfiles[dryAdiabatOffsetsAndLengths[profileIndices[idx]].x], dryAdiabatOffsetsAndLengths[profileIndices[idx]].y, normP);

			float ambientTemp = getDenormalizedTemp(ambientIntersection.x, normP);
			float particleTemp = getDenormalizedTemp(dryAdiabatIntersection.x, normP);

			//printf("| ambientTemp [deg C] = %0.2f\n", ambientTemp);
			//printf("| particleTemp [deg C] = %0.2f\n", particleTemp);


			toKelvin_dev(ambientTemp);
			toKelvin_dev(particleTemp);

			float ambientTheta = computeThetaFromAbsoluteK_dev(ambientTemp, particlePressures[idx]);
			float particleTheta = computeThetaFromAbsoluteK_dev(particleTemp, particlePressures[idx]);

			float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;

			//printf("| a = %0.2f\n", a);

			verticalVelocities[idx] = verticalVelocities[idx] + a * d_const_delta_t;
			float deltaY = verticalVelocities[idx] * d_const_delta_t + 0.5f * a * d_const_delta_t * d_const_delta_t;

			//printf("| delta y = %0.2f\n", deltaY);

			//printf("| height (before unmap) = %0.2f\n", particleVertices[idx].y);


			mapFromSimulationBox_dev(particleVertices[idx].y);
			//float tmpY = getRealWorldCoords_dev(particleVertices[idx].y);

			//printf("| height (after unmap) = %0.2f\n", particleVertices[idx].y);
			particleVertices[idx].y += deltaY;

			//printf("| height (after unmap) = %0.2f\n", tmpY);

			//tmpY += deltaY;

			//printf("| height (after unmap + delta y) = %0.2f\n", particleVertices[idx].y);
			particlePressures[idx] = getPressureVal_dev(particleVertices[idx].y);

			//printf("| height (after unmap + delta y) = %0.2f\n", tmpY);

			//particlePressures[idx] = getPressureVal_dev(tmpY);

			//particleVertices[idx].y = getSimulationBoxCoords_dev(tmpY);

			mapToSimulationBox_dev(particleVertices[idx].y);

			//printf("| height (final) = %0.2f\n", particleVertices[idx].y);


		} else {
			float normP = getNormalizedPres(particlePressures[idx]);
			glm::vec2 ambientIntersection = getIntersectionWithIsobar(ambientTempCurve, numAmbientTempCurveVertices, normP);
			glm::vec2 moistAdiabatIntersection = getIntersectionWithIsobar(&moistAdiabatProfiles[moistAdiabatOffsetsAndLengths[profileIndices[idx]].x], moistAdiabatOffsetsAndLengths[profileIndices[idx]].y, normP);

			float ambientTemp = getDenormalizedTemp(ambientIntersection.x, normP);
			float particleTemp = getDenormalizedTemp(moistAdiabatIntersection.x, normP);

			//printf("| ambientTemp [deg C] = %0.2f\n", ambientTemp);
			//printf("| particleTemp [deg C] = %0.2f\n", particleTemp);


			toKelvin_dev(ambientTemp);
			toKelvin_dev(particleTemp);

			float ambientTheta = computeThetaFromAbsoluteK_dev(ambientTemp, particlePressures[idx]);
			float particleTheta = computeThetaFromAbsoluteK_dev(particleTemp, particlePressures[idx]);

			float a = 9.81f * (particleTheta - ambientTheta) / ambientTheta;

			//printf("| a = %0.2f\n", a);

			verticalVelocities[idx] = verticalVelocities[idx] + a * d_const_delta_t;
			float deltaY = verticalVelocities[idx] * d_const_delta_t + 0.5f * a * d_const_delta_t * d_const_delta_t;

			//printf("| delta y = %0.2f\n", deltaY);

			//printf("| height (before unmap) = %0.2f\n", particleVertices[idx].y);


			mapFromSimulationBox_dev(particleVertices[idx].y);
			//float tmpY = getRealWorldCoords_dev(particleVertices[idx].y);

			//printf("| height (after unmap) = %0.2f\n", particleVertices[idx].y);
			particleVertices[idx].y += deltaY;

			//printf("| height (after unmap) = %0.2f\n", tmpY);

			//tmpY += deltaY;

			//printf("| height (after unmap + delta y) = %0.2f\n", particleVertices[idx].y);
			particlePressures[idx] = getPressureVal_dev(particleVertices[idx].y);

			//printf("| height (after unmap + delta y) = %0.2f\n", tmpY);

			//particlePressures[idx] = getPressureVal_dev(tmpY);

			//particleVertices[idx].y = getSimulationBoxCoords_dev(tmpY);

			mapToSimulationBox_dev(particleVertices[idx].y);

			//printf("| height (final) = %0.2f\n", particleVertices[idx].y);



		}





	}
}

__global__ void simulationStepKernelAlt(glm::vec3 *particleVertices, int numParticles, float *verticalVelocities, int *profileIndices, float *particlePressures) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {



		particleVertices[idx].y += 0.01f;

	}
}



STLPSimulatorCUDA::STLPSimulatorCUDA(VariableManager * vars, STLPDiagram * stlpDiagram) : vars(vars), stlpDiagram(stlpDiagram) {
	heightMap = vars->heightMap;

	groundHeight = getAltitudeFromPressure(stlpDiagram->P0);
	boxTopHeight = groundHeight + simulationBoxHeight;

	layerVisShader = ShaderManager::getShaderPtr("singleColorAlpha");

	initBuffers();
	//initCUDA();

	spriteTexture.loadTexture(((string)TEXTURES_DIR + "testTexture.png").c_str());
	secondarySpriteTexture.loadTexture(((string)TEXTURES_DIR + "testTexture2.png").c_str());

	profileMap = new ppmImage("profileMaps/120x80_pm_03.ppm");

	//spriteTexture.loadTexture(((string)TEXTURES_DIR + "pointTex.png").c_str());

	
}

STLPSimulatorCUDA::~STLPSimulatorCUDA() {
	CHECK_ERROR(cudaGraphicsUnregisterResource(cudaParticleVerticesVBO));

	if (profileMap) {
		delete(profileMap);
	}

}

void STLPSimulatorCUDA::initBuffers() {

	glGenVertexArrays(1, &particlesVAO);
	glBindVertexArray(particlesVAO);

	glGenBuffers(1, &particlesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glGenBuffers(1, &particleProfilesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleProfilesVBO);

	glEnableVertexAttribArray(5);
	glVertexAttribIPointer(5, 1, GL_INT, sizeof(int), (void *)0);

	glBindVertexArray(0);


	//glGenBuffers(1, &profileDataSSBO);
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, profileDataSSBO);
	//glNamedBufferStorage(profileDataSSBO, n * sizeof(PointLightData), NULL, GL_DYNAMIC_STORAGE_BIT);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, profileDataSSBO);







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


	vertices.clear();

	glGenVertexArrays(1, &groundLevelVAO);
	glBindVertexArray(groundLevelVAO);
	glGenBuffers(1, &groundLevelVBO);
	glBindBuffer(GL_ARRAY_BUFFER, groundLevelVBO);

	altitude = getAltitudeFromPressure(stlpDiagram->P0);
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

void STLPSimulatorCUDA::initCUDA() {

	blockDim = dim3(256, 1, 1);
	gridDim = dim3((int)ceil((float)maxNumParticles / (float)blockDim.x), 1, 1);


	cudaMalloc((void**)&d_verticalVelocities, sizeof(float) * maxNumParticles);
	cudaMalloc((void**)&d_profileIndices, sizeof(int) * maxNumParticles);
	cudaMalloc((void**)&d_particlePressures, sizeof(float) * maxNumParticles);
	
	cudaMemset(d_verticalVelocities, 0, sizeof(float) * maxNumParticles);
	//cudaMemset(d_profileIndices, 0, sizeof(int) * maxNumParticles);
	//cudaMemset(d_particlePressures, 0, sizeof(float) * maxNumParticles);

	vector<int> itmp;
	vector<float> ftmp;

	for (int i = 0; i < numParticles; i++) {
		itmp.push_back(particles[i].profileIndex);
		ftmp.push_back(particles[i].pressure);
	}
	cudaMemcpy(d_profileIndices, &itmp[0], sizeof(int) * itmp.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_particlePressures, &ftmp[0], sizeof(float) * ftmp.size(), cudaMemcpyHostToDevice);

	


	cudaMalloc((void**)&d_ambientTempCurve, sizeof(glm::vec2) * stlpDiagram->ambientCurve.vertices.size());

	cudaMemcpy(d_ambientTempCurve, &stlpDiagram->ambientCurve.vertices[0], sizeof(glm::vec2) * stlpDiagram->ambientCurve.vertices.size(), cudaMemcpyHostToDevice);

	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticleVerticesVBO, particlesVBO, cudaGraphicsRegisterFlagsWriteDiscard));

	CHECK_ERROR(cudaMemcpyToSymbol(d_const_numProfiles, &stlpDiagram->numProfiles, sizeof(int)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_const_maxP, &stlpDiagram->maxP, sizeof(float)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_const_delta_t, &delta_t, sizeof(float)));

	CHECK_ERROR(cudaMemcpyToSymbol(d_const_boxTopHeight, &boxTopHeight, sizeof(float)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_const_groundHeight, &groundHeight, sizeof(float)));

	float latticeH = (float)vars->latticeHeight;
	CHECK_ERROR(cudaMemcpyToSymbol(d_const_latticeHeight, &latticeH, sizeof(float)));


	itmp.clear();
	vector<glm::vec2> tmp;
	vector<glm::ivec2> ivectmp;
	tmp.reserve(stlpDiagram->numProfiles * stlpDiagram->dryAdiabatProfiles[0].vertices.size()); // probably the largest possible collection

	// DRY ADIABAT OFFSETS
	tmp.clear();
	ivectmp.clear();
	int sum = 0;
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		itmp.push_back(sum);
		float prevSum = sum;
		sum += stlpDiagram->dryAdiabatProfiles[i].vertices.size();
		ivectmp.push_back(glm::ivec2(prevSum, sum - prevSum)); // x = offset, y = length
		//cout << stlpDiagram->dryAdiabatProfiles[i].vertices.size() << endl;
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_dryAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size()));
	//cudaMalloc((void**)&d_dryAdiabatOffsets, sizeof(int) * itmp.size());
	//CHECK_ERROR(cudaMemcpy(d_dryAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size(), cudaMemcpyHostToDevice));
	cudaMalloc((void**)&d_dryAdiabatOffsetsAndLengths, sizeof(glm::ivec2) * ivectmp.size());
	CHECK_ERROR(cudaMemcpy(d_dryAdiabatOffsetsAndLengths, &ivectmp[0], sizeof(glm::ivec2) * ivectmp.size(), cudaMemcpyHostToDevice));



	// MOIST ADIABAT OFFSETS
	itmp.clear();
	tmp.clear();
	ivectmp.clear();
	sum = 0;
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		itmp.push_back(sum);
		float prevSum = sum;
		sum += stlpDiagram->moistAdiabatProfiles[i].vertices.size();
		ivectmp.push_back(glm::ivec2(prevSum, sum - prevSum)); // x = offset, y = length
		//cout << stlpDiagram->moistAdiabatProfiles[i].vertices.size() << endl;
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_moistAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size()));
	//cudaMalloc((void**)&d_moistAdiabatOffsets, sizeof(int) * itmp.size());
	//CHECK_ERROR(cudaMemcpy(d_moistAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size(), cudaMemcpyHostToDevice));
	cudaMalloc((void**)&d_moistAdiabatOffsetsAndLengths, sizeof(glm::ivec2) * ivectmp.size());
	CHECK_ERROR(cudaMemcpy(d_moistAdiabatOffsetsAndLengths, &ivectmp[0], sizeof(glm::ivec2) * ivectmp.size(), cudaMemcpyHostToDevice));


	// DRY ADIABATS
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		for (int j = 0; j < stlpDiagram->dryAdiabatProfiles[i].vertices.size(); j++) {
			tmp.push_back(stlpDiagram->dryAdiabatProfiles[i].vertices[j]);
		}
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_dryAdiabatProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size()));
	cudaMalloc((void**)&d_dryAdiabatProfiles, sizeof(glm::vec2) * tmp.size());
	CHECK_ERROR(cudaMemcpy(d_dryAdiabatProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size(), cudaMemcpyHostToDevice));


	// MOIST ADIABATS
	tmp.clear();
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		for (int j = 0; j < stlpDiagram->moistAdiabatProfiles[i].vertices.size(); j++) {
			tmp.push_back(stlpDiagram->moistAdiabatProfiles[i].vertices[j]);
		}
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_moistAdiabatProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size()));
	cudaMalloc((void**)&d_moistAdiabatProfiles, sizeof(glm::vec2) * tmp.size());
	CHECK_ERROR(cudaMemcpy(d_moistAdiabatProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size(), cudaMemcpyHostToDevice));

	// CCL Profiles
	tmp.clear();
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		tmp.push_back(stlpDiagram->CCLProfiles[i]);
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_CCLProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size()));
	cudaMalloc((void**)&d_CCLProfiles, sizeof(glm::vec2) * tmp.size());
	CHECK_ERROR(cudaMemcpy(d_CCLProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size(), cudaMemcpyHostToDevice));


	// Tc Profiles
	tmp.clear();
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		tmp.push_back(stlpDiagram->TcProfiles[i]);
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_TcProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size()));
	cudaMalloc((void**)&d_TcProfiles, sizeof(glm::vec2) * tmp.size());
	CHECK_ERROR(cudaMemcpy(d_TcProfiles, &tmp[0], sizeof(glm::vec2) * tmp.size(), cudaMemcpyHostToDevice));




}

void STLPSimulatorCUDA::doStep() {
	/*
	glm::vec3 *dptr;
	CHECK_ERROR(cudaGraphicsMapResources(1, &particleSystem->cudaParticleVerticesVBO, 0));
	size_t num_bytes;
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, particleSystem->cudaParticleVerticesVBO));
	//printf("CUDA-STLP mapped VBO: May access %ld bytes\n", num_bytes);

	//CHECK_ERROR(cudaPeekAtLastError());

	// FIX d_ VALUES HERE!!! (use the ones from ParticleSystem)
	simulationStepKernel << <gridDim.x, blockDim.x >> > (dptr, numParticles, d_verticalVelocities, d_profileIndices, d_particlePressures, d_ambientTempCurve, stlpDiagram->ambientCurve.vertices.size(), d_dryAdiabatProfiles, d_dryAdiabatOffsetsAndLengths, d_moistAdiabatProfiles, d_moistAdiabatOffsetsAndLengths, d_CCLProfiles, d_TcProfiles);

	CHECK_ERROR(cudaPeekAtLastError());

	cudaGraphicsUnmapResources(1, &particleSystem->cudaParticleVerticesVBO, 0);
	*/

	
	glm::vec3 *dptr;
	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0));
	size_t num_bytes;
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaParticleVerticesVBO));
	//printf("CUDA-STLP mapped VBO: May access %ld bytes\n", num_bytes);

	//CHECK_ERROR(cudaPeekAtLastError());
	simulationStepKernel << <gridDim.x, blockDim.x >> > (dptr, numParticles, d_verticalVelocities, d_profileIndices, d_particlePressures, d_ambientTempCurve, stlpDiagram->ambientCurve.vertices.size(), d_dryAdiabatProfiles, d_dryAdiabatOffsetsAndLengths, d_moistAdiabatProfiles, d_moistAdiabatOffsetsAndLengths, d_CCLProfiles, d_TcProfiles);

	CHECK_ERROR(cudaPeekAtLastError());

	cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0);
	



}

void STLPSimulatorCUDA::updateGPU_delta_t() {
	CHECK_ERROR(cudaMemcpyToSymbol(d_const_delta_t, &delta_t, sizeof(float)));
}

void STLPSimulatorCUDA::resetSimulation() {
}

void STLPSimulatorCUDA::generateParticle() {


	// testing generation in circle
	float randx;
	float randz;

	bool incircle = false;
	if (incircle) {

		float R = 10.0f;
		static std::random_device rd;
		static std::mt19937 mt(rd());
		static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		float a = dist(mt) * 2.0f * (float)PI;
		float r = R * sqrtf(dist(mt));

		randx = r * cos(a);
		randz = r * sin(a);

		randx += heightMap->width / 2;
		randz += heightMap->height / 2;

	} else {
		randx = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->width - 2.0f)));
		randz = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->height - 2.0f)));
	}

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

	//y = 5.0f; //////////////////////////////////////////////////////// FORCE Y to dry adiabat

	particlePositions.push_back(glm::vec3(randx, y, randz));


	mapFromSimulationBox(y);

	Particle p;
	p.position = glm::vec3(randx, y, randz);
	p.velocity = glm::vec3(0.0f);


	if (profileMap && profileMap->height >= heightMap->height && profileMap->width >= heightMap->width) {

		glm::vec2 p1 = profileMap->data[leftx][leftz];
		glm::vec2 p2 = profileMap->data[leftx][rightz];
		glm::vec2 p3 = profileMap->data[rightx][leftz];
		glm::vec2 p4 = profileMap->data[rightx][rightz];

		glm::vec2 pi1 = zRatio * p2 + (1.0f - zRatio) * p1;
		glm::vec2 pi2 = zRatio * p4 + (1.0f - zRatio) * p3;

		glm::vec2 pif = xRatio * pi2 + (1.0f - xRatio) * pi1;
		glm::ivec2 pii = (glm::ivec2)pif;

		if (pii.y != pii.x) {
			p.profileIndex = (rand() % (pii.y - pii.x) + pii.x) % (stlpDiagram->numProfiles - 1);
		} else {
			p.profileIndex = pii.x % (stlpDiagram->numProfiles - 1);
		}

	} else {
		p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
	}


	p.updatePressureVal();

	particles.push_back(p);
	numParticles++;

}

void STLPSimulatorCUDA::draw(ShaderProgram & particlesShader, glm::vec3 cameraPos) {
	
	glUseProgram(particlesShader.id);

	particlesShader.setInt("u_Tex", 0);
	particlesShader.setInt("u_SecondTex", 1);
	particlesShader.setVec3("u_TintColor", vars->tintColor);

	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, spriteTexture.id);

	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, secondarySpriteTexture.id);

	glPointSize(pointSize);
	//particlesShader.setVec4("color", glm::vec4(1.0f, 0.4f, 1.0f, 1.0f));
	particlesShader.setVec3("u_CameraPos", cameraPos);
	particlesShader.setFloat("u_PointSizeModifier", pointSize);
	particlesShader.setFloat("u_OpacityMultiplier", vars->opacityMultiplier);

	glBindVertexArray(particlesVAO);

	//glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);

	glDrawArrays(GL_POINTS, 0, numParticles);



	if (vars->showCCLLevelLayer || vars->showELLevelLayer) {
		GLboolean cullFaceEnabled;
		glGetBooleanv(GL_CULL_FACE, &cullFaceEnabled);
		glDisable(GL_CULL_FACE);

		layerVisShader->use();

		if (vars->showCCLLevelLayer) {
			layerVisShader->setVec4("u_Color", glm::vec4(1.0f, 0.0f, 0.0f, 0.2f));

			glBindVertexArray(CCLLevelVAO);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		}

		if (vars->showELLevelLayer) {
			layerVisShader->setVec4("u_Color", glm::vec4(0.0f, 1.0f, 0.0f, 0.2f));


			glBindVertexArray(ELLevelVAO);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		}

		if (cullFaceEnabled) {
			glEnable(GL_CULL_FACE);
		}
	}
}

void STLPSimulatorCUDA::initParticles() {
	for (int i = 0; i < maxNumParticles; i++) {
		generateParticle();
	}
	//glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);
	glNamedBufferData(particlesVBO, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);
	cout << "Particles initialized: num particles = " << numParticles << endl;


	vector<int> particleProfiles;
	for (int i = 0; i < maxNumParticles; i++) {
		particleProfiles.push_back(particles[i].profileIndex);
	}
	glNamedBufferData(particleProfilesVBO, sizeof(int) * particleProfiles.size(), &particleProfiles[0], GL_STATIC_DRAW);


	ShaderProgram *s = ShaderManager::getShaderPtr("pointSpriteTest");
	s->use();
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		string fullName = "u_ProfileCCLs[" + to_string(i) + "]";
		float P = stlpDiagram->CCLProfiles[i].y;
		float y = getAltitudeFromPressure(P);
		mapToSimulationBox(y);
		s->setFloat(fullName, y);
	}
	s->setInt("u_NumProfiles", stlpDiagram->numProfiles);
	

}

void STLPSimulatorCUDA::mapToSimulationBox(float & val) {
	rangeToRange(val, groundHeight, boxTopHeight, 0.0f, vars->latticeHeight);
}

void STLPSimulatorCUDA::mapFromSimulationBox(float & val) {
	rangeToRange(val, 0.0f, vars->latticeHeight, groundHeight, boxTopHeight);
}
