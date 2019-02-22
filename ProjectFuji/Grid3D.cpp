#include "Grid3D.h"


Grid3D::Grid3D(int width, int height, int depth, int stepX, int stepY, int stepZ) {

	//for (int z = 0; z < GRID_DEPTH; z += stepZ) {
	//	for (int x = 0; x < GRID_WIDTH; x += stepX) {
	//		gridVertices.push_back(glm::vec3(x, 0.0f, z));
	//		gridVertices.push_back(glm::vec3(x, GRID_HEIGHT - 1, z));
	//	}
	//	for (int y = 0; y < GRID_HEIGHT; y += stepY) {
	//		gridVertices.push_back(glm::vec3(0.0f, y, z));
	//		gridVertices.push_back(glm::vec3(GRID_WIDTH - 1, y, z));
	//	}
	//}



	//glGenVertexArrays(1, &VAO);
	//glBindVertexArray(VAO);
	//glGenBuffers(1, &VBO);
	//glBindBuffer(GL_ARRAY_BUFFER, VBO);

	//glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(glm::vec3), &gridVertices[0], GL_STATIC_DRAW);

	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	//glBindVertexArray(0);


	//glGenVertexArrays(1, &pointsVAO);
	//glBindVertexArray(pointsVAO);
	//glGenBuffers(1, &pointsVBO);
	//glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);

	vector<glm::vec3> bData;

	//for (int x = 0; x < GRID_WIDTH; x += stepX) {
	//	for (int y = 0; y < GRID_HEIGHT; y += stepY) {
	//		for (int z = 0; z < GRID_DEPTH; z += stepZ) {
	//			bData.push_back(glm::vec3(x, y, z));
	//		}
	//	}
	//}

	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bData.size(), &bData[0], GL_STATIC_DRAW);

	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	//glBindVertexArray(0);


	glGenVertexArrays(1, &boxVAO);
	glBindVertexArray(boxVAO);
	glGenBuffers(1, &boxVBO);
	glBindBuffer(GL_ARRAY_BUFFER, boxVBO);

	bData.clear();

	float gw = width - 1;
	float gh = height - 1;
	float gd = depth - 1;
	bData.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
	bData.push_back(glm::vec3(gw, 0.0f, 0.0f));

	bData.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
	bData.push_back(glm::vec3(0.0f, gh, 0.0f));

	bData.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
	bData.push_back(glm::vec3(0.0f, 0.0f, gd));

	bData.push_back(glm::vec3(0.0f, 0.0f, gd));
	bData.push_back(glm::vec3(gw, 0.0f, gd));

	bData.push_back(glm::vec3(gw, 0.0f, gd));
	bData.push_back(glm::vec3(gw, 0.0f, 0.0f));

	bData.push_back(glm::vec3(gw, 0.0f, 0.0f));
	bData.push_back(glm::vec3(gw, gh, 0.0f));

	bData.push_back(glm::vec3(gw, gh, gd));
	bData.push_back(glm::vec3(gw, gh, 0.0f));

	bData.push_back(glm::vec3(gw, 0.0f, gd));
	bData.push_back(glm::vec3(gw, gh, gd));
	
	bData.push_back(glm::vec3(0.0f, gh, 0.0f));
	bData.push_back(glm::vec3(0.0f, gh, gd));

	bData.push_back(glm::vec3(0.0f, gh, 0.0f));
	bData.push_back(glm::vec3(gw, gh, 0.0f));

	bData.push_back(glm::vec3(0.0f, gh, gd));
	bData.push_back(glm::vec3(gw, gh, gd));

	bData.push_back(glm::vec3(0.0f, 0.0f, gd));
	bData.push_back(glm::vec3(0.0f, gh, gd));


	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bData.size(), &bData[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);

}


Grid3D::~Grid3D() {}

void Grid3D::draw(ShaderProgram &shader) {

	glUseProgram(shader.id);

	//glLineWidth(1.0f);
	//shader.setVec3("uColor", glm::vec3(0.1f, 0.1f, 0.1f));
	//glBindVertexArray(VAO);
	//glDrawArrays(GL_LINES, 0, gridVertices.size());

	//glPointSize(1.0f);
	//shader.setVec3("uColor", glm::vec3(0.4f, 0.4f, 0.1f));

	/*glBindVertexArray(pointsVAO);
	glDrawArrays(GL_POINTS, 0, GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH);
	*/

	glLineWidth(4.0f);
	shader.setVec3("uColor", glm::vec3(0.9f, 0.9f, 0.2f));

	glBindVertexArray(boxVAO);
	glDrawArrays(GL_LINES, 0, 24);
}
