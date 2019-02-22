#include "Grid2D.h"


Grid2D::Grid2D(int width, int height, int stepX, int stepY) {

	printf("WIDTH = %d, HEIGHT = %d, STEPX = %d, STEPY = %d\n", width, height, stepX, stepY);
	for (int x = 0; x < width; x += stepX) {
		gridVertices.push_back(glm::vec3(x, 0.0f, -2.0f));
		gridVertices.push_back(glm::vec3(x, height - 1, -2.0f));
	}
	for (int y = 0; y < height; y += stepY) {
		gridVertices.push_back(glm::vec3(0.0f, y, -2.0f));
		gridVertices.push_back(glm::vec3(width - 1, y, -2.0f));
	}


	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	
	glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(glm::vec3), &gridVertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
	
	glBindVertexArray(0);



}


Grid2D::~Grid2D() {}

void Grid2D::draw(ShaderProgram &shader) {
	glUseProgram(shader.id);
	shader.setVec3("uColor", glm::vec3(0.1f, 0.1f, 0.1f));
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, gridVertices.size());


}
