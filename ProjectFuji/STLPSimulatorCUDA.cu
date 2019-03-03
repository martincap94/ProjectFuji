#include "STLPSimulatorCUDA.h"

#include "ShaderManager.h"
#include "STLPUtils.h"
#include "Utils.h"





STLPSimulatorCUDA::STLPSimulatorCUDA(VariableManager * vars, STLPDiagram * stlpDiagram) : vars(vars), stlpDiagram(stlpDiagram) {
	groundHeight = stlpDiagram->P0;
	boxTopHeight = groundHeight + simulationBoxHeight;

	layerVisShader = ShaderManager::getShaderPtr("singleColorAlpha");

	initBuffers();

	
}

STLPSimulatorCUDA::~STLPSimulatorCUDA() {
}

void STLPSimulatorCUDA::initBuffers() {

	glGenVertexArrays(1, &particlesVAO);
	glBindVertexArray(particlesVAO);

	glGenBuffers(1, &particlesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);

	glEnableVertexAttribArray(0);





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

void STLPSimulatorCUDA::doStep() {
}

void STLPSimulatorCUDA::resetSimulation() {
}

void STLPSimulatorCUDA::generateParticle() {
}

void STLPSimulatorCUDA::draw(ShaderProgram & particlesShader) {
	
	//glUseProgram(particlesShader.id);

	//glPointSize(1.0f);
	//particlesShader.setVec4("color", glm::vec4(1.0f, 0.4f, 1.0f, 1.0f));

	//glBindVertexArray(particlesVAO);

	//glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particlePositions[0], GL_DYNAMIC_DRAW);
	////glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &testParticle.position[0], GL_DYNAMIC_DRAW);

	////glDrawArrays(GL_POINTS, 0, numParticles);
	//glDrawArrays(GL_POINTS, 0, numParticles);

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
}

void STLPSimulatorCUDA::mapToSimulationBox(float & val) {
	rangeToRange(val, groundHeight, boxTopHeight, 0.0f, vars->latticeHeight);
}

void STLPSimulatorCUDA::mapFromSimulationBox(float & val) {
	rangeToRange(val, 0.0f, vars->latticeHeight, groundHeight, boxTopHeight);
}
