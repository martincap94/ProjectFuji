#include "STLPSimulatorCUDA.h"

#include "ShaderManager.h"
#include "STLPUtils.h"
#include "Utils.h"
#include "HeightMap.h"
#include "CUDAUtils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"




__constant__ int d_numProfiles;

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

__global__ void simulationStepKernel(glm::vec3 *particleVertices, int numParticles, float *verticalVelocities, int *profileIndices, float *particlePressures, glm::vec2 *ambientTempCurve, glm::vec2 *dryAdiabatProfiles, int *dryAdiabatOffsets, glm::vec2 *moistAdiabatProfiles, int *moistAdiabatOffsets, glm::vec2 *CCLProfiles, glm::vec2 *TcProfiles) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {



		particleVertices[idx].y += 0.01f;

	}
}

__global__ void simulationStepKernelAlt(glm::vec3 *particleVertices, int numParticles, float *verticalVelocities, int *profileIndices, float *particlePressures) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {



		particleVertices[idx].y += 0.01f;

	}
}



STLPSimulatorCUDA::STLPSimulatorCUDA(VariableManager * vars, STLPDiagram * stlpDiagram) : vars(vars), stlpDiagram(stlpDiagram) {
	groundHeight = stlpDiagram->P0;
	boxTopHeight = groundHeight + simulationBoxHeight;

	layerVisShader = ShaderManager::getShaderPtr("singleColorAlpha");

	initBuffers();
	//initCUDA();
	
}

STLPSimulatorCUDA::~STLPSimulatorCUDA() {
	cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0);

}

void STLPSimulatorCUDA::initBuffers() {

	glGenVertexArrays(1, &particlesVAO);
	glBindVertexArray(particlesVAO);

	glGenBuffers(1, &particlesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);

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

void STLPSimulatorCUDA::initCUDA() {

	blockDim = dim3(256, 1, 1);
	gridDim = dim3((int)ceil((float)maxNumParticles / (float)blockDim.x), 1, 1);


	cudaMalloc((void**)&d_verticalVelocities, sizeof(float) * maxNumParticles);
	cudaMalloc((void**)&d_profileIndices, sizeof(int) * maxNumParticles);
	cudaMalloc((void**)&d_particlePressures, sizeof(float) * maxNumParticles);
	
	cudaMemset(d_verticalVelocities, 0, sizeof(float) * maxNumParticles);
	cudaMemset(d_profileIndices, 0, sizeof(int) * maxNumParticles);
	cudaMemset(d_particlePressures, 0, sizeof(float) * maxNumParticles);

	cudaMalloc((void**)&d_ambientTempCurve, sizeof(glm::vec2) * stlpDiagram->ambientCurve.vertices.size());

	cudaMemcpy(d_ambientTempCurve, &stlpDiagram->ambientCurve.vertices[0], sizeof(glm::vec2) * stlpDiagram->ambientCurve.vertices.size(), cudaMemcpyHostToDevice);

	cudaGraphicsGLRegisterBuffer(&cudaParticleVerticesVBO, particlesVBO, cudaGraphicsRegisterFlagsNone);

	CHECK_ERROR(cudaMemcpyToSymbol(d_numProfiles, &stlpDiagram->numProfiles, sizeof(int)));

	vector<int> itmp;
	vector<glm::vec2> tmp;
	tmp.reserve(stlpDiagram->numProfiles * stlpDiagram->dryAdiabatProfiles[0].vertices.size()); // probably the largest possible collection

	// DRY ADIABAT OFFSETS
	int sum = 0;
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		itmp.push_back(sum);
		sum += stlpDiagram->dryAdiabatProfiles[i].vertices.size();
		//cout << stlpDiagram->dryAdiabatProfiles[i].vertices.size() << endl;
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_dryAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size()));
	cudaMalloc((void**)&d_dryAdiabatOffsets, sizeof(int) * itmp.size());
	CHECK_ERROR(cudaMemcpy(d_dryAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size(), cudaMemcpyHostToDevice));



	// MOIST ADIABAT OFFSETS
	itmp.clear();
	sum = 0;
	for (int i = 0; i < stlpDiagram->numProfiles; i++) {
		itmp.push_back(sum);
		sum += stlpDiagram->moistAdiabatProfiles[i].vertices.size();
		//cout << stlpDiagram->moistAdiabatProfiles[i].vertices.size() << endl;
	}
	//CHECK_ERROR(cudaMemcpyToSymbol(d_const_moistAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size()));
	cudaMalloc((void**)&d_moistAdiabatOffsets, sizeof(int) * itmp.size());
	CHECK_ERROR(cudaMemcpy(d_moistAdiabatOffsets, &itmp[0], sizeof(int) * itmp.size(), cudaMemcpyHostToDevice));



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

	glm::vec3 *dptr;
	cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaParticleVerticesVBO);
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	simulationStepKernel << <gridDim.x, blockDim.x >> > (dptr, numParticles, d_verticalVelocities, d_profileIndices, d_particlePressures, d_ambientTempCurve, d_dryAdiabatProfiles, d_dryAdiabatOffsets, d_moistAdiabatProfiles, d_moistAdiabatOffsets, d_CCLProfiles, d_TcProfiles);



	cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0);



}

void STLPSimulatorCUDA::resetSimulation() {
}

void STLPSimulatorCUDA::generateParticle() {

	float randx = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->width - 2.0f)));
	float randz = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->height - 2.0f)));


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


	mapFromSimulationBox(y);

	Particle p;
	p.position = glm::vec3(randx, y, randz);
	p.velocity = glm::vec3(0.0f);
	p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
	p.updatePressureVal();

	particles.push_back(p);
	numParticles++;

}

void STLPSimulatorCUDA::draw(ShaderProgram & particlesShader) {
	
	glUseProgram(particlesShader.id);

	glPointSize(1.0f);
	particlesShader.setVec4("color", glm::vec4(1.0f, 0.4f, 1.0f, 1.0f));

	glBindVertexArray(particlesVAO);

	//glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);

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

void STLPSimulatorCUDA::initParticles() {
	for (int i = 0; i < maxNumParticles; i++) {
		generateParticle();
	}
	//glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);
	glNamedBufferData(particlesVBO, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);
	cout << "Particles initialized: num particles = " << numParticles << endl;

}

void STLPSimulatorCUDA::mapToSimulationBox(float & val) {
	rangeToRange(val, groundHeight, boxTopHeight, 0.0f, vars->latticeHeight);
}

void STLPSimulatorCUDA::mapFromSimulationBox(float & val) {
	rangeToRange(val, 0.0f, vars->latticeHeight, groundHeight, boxTopHeight);
}
