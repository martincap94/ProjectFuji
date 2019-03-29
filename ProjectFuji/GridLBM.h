#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>
#include <vector>


class LBM3D_1D_indices;
class ShaderProgram;

class GridLBM {
public:

	GridLBM(LBM3D_1D_indices *owner, glm::vec3 stepSize = glm::vec3(1.0f));
	~GridLBM();

	void draw();

private:

	glm::vec3 stepSize;
	glm::vec3 lineColor = glm::vec3(0.9f, 0.9f, 0.2f);

	LBM3D_1D_indices *lbm = nullptr;

	ShaderProgram *shader = nullptr;
	GLuint boxVAO;
	GLuint boxVBO;

	std::vector<glm::vec3> gridVertices;


};

