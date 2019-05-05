#include "GeneralGrid.h"

#include "ShaderManager.h"

#include <glm\glm.hpp>
#include <vector>

GeneralGrid::GeneralGrid() {
	unlitColorShader = ShaderManager::getShaderPtr("unlitColor");
}

GeneralGrid::GeneralGrid(int range, int stepSize) : range(range), stepSize(stepSize) {
	unlitColorShader = ShaderManager::getShaderPtr("unlitColor");

	vector<glm::vec3> gridVertices;

	float axisRange = range * 10.0f;

	float epsilon = 0.1f;

	for (int x = -range; x <= range; x += stepSize) {
		gridVertices.push_back(glm::vec3(x, -epsilon, -range));
		gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));
		gridVertices.push_back(glm::vec3(x, -epsilon, range));
		gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));

	}
	//for (int y = -range; y <= range; y += stepSize) {
	//	gridVertices.push_back(glm::vec3(0.0f, y, -range));
	//	gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));
	//	gridVertices.push_back(glm::vec3(0.0f, y, range));
	//	gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));

	//}
	for (int z = -range; z <= range; z += stepSize) {
		gridVertices.push_back(glm::vec3(-range, -epsilon, z));
		gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));
		gridVertices.push_back(glm::vec3(range, -epsilon, z));
		gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));

	}





	gridVertices.push_back(glm::vec3(-axisRange, 0.0f, 0.0f));
	gridVertices.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
	gridVertices.push_back(glm::vec3(axisRange, 0.0f, 0.0f));
	gridVertices.push_back(glm::vec3(1.0f, 0.0f, 0.0f));

	gridVertices.push_back(glm::vec3(0.0f, -axisRange, 0.0f));
	gridVertices.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
	gridVertices.push_back(glm::vec3(0.0f, axisRange, 0.0f));
	gridVertices.push_back(glm::vec3(0.0f, 1.0f, 0.0f));

	gridVertices.push_back(glm::vec3(0.0f, 0.0f, -axisRange));
	gridVertices.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
	gridVertices.push_back(glm::vec3(0.0f, 0.0f, axisRange));
	gridVertices.push_back(glm::vec3(0.0f, 0.0f, 1.0f));


	numLines = (int)gridVertices.size();

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(glm::vec3), &gridVertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)(sizeof(glm::vec3)));


	glBindVertexArray(0);


}


GeneralGrid::~GeneralGrid() {
}

void GeneralGrid::draw() {

	unlitColorShader->use();

	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, numLines);

	glBindVertexArray(0);

}
