#include "GeneralGrid.h"

#include <glm\glm.hpp>
#include <vector>

GeneralGrid::GeneralGrid() {
}

GeneralGrid::GeneralGrid(int range, int stepSize, bool drawXZGrid) : range(range), stepSize(stepSize) {

	vector<glm::vec3> gridVertices;

	float axisRange = range * 100.0f;

	if (drawXZGrid) {
		for (int x = -range; x <= range; x += stepSize) {
			gridVertices.push_back(glm::vec3(x, 0.0f, -range));
			gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));
			gridVertices.push_back(glm::vec3(x, 0.0f, range));
			gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));

		}
		//for (int y = -range; y <= range; y += stepSize) {
		//	gridVertices.push_back(glm::vec3(0.0f, y, -range));
		//	gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));
		//	gridVertices.push_back(glm::vec3(0.0f, y, range));
		//	gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));

		//}
		for (int z = -range; z <= range; z += stepSize) {
			gridVertices.push_back(glm::vec3(-range, 0.0f, z));
			gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));
			gridVertices.push_back(glm::vec3(range, 0.0f, z));
			gridVertices.push_back(glm::vec3(0.1f, 0.1f, 0.1f));

		}
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


	numLines = gridVertices.size();

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

void GeneralGrid::draw(ShaderProgram & shader) {
	glUseProgram(shader.id);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, numLines);

	glBindVertexArray(0);

}
