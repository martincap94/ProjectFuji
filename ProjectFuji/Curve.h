#pragma once

#include <glm\glm.hpp>

#include <vector>

#include <glad\glad.h>
#include "ShaderProgram.h"


using namespace std;

class Curve {
public:


	vector<glm::vec2> vertices;

	Curve();
	~Curve();

	void initBuffers();

	void draw(ShaderProgram &shader);
	void draw(ShaderProgram *shader);

	glm::vec2 getIntersectionWithIsobar(float normalizedPressure);

	void printVertices();

private:

	GLuint VAO;
	GLuint VBO;


};


bool findIntersectionNew(const Curve &first, const Curve &second, glm::vec2 &outIntersection, bool reverseFirst = false, bool reverseSecond = false);

glm::vec2 findIntersection(const Curve &first, const Curve &second, bool reverseFirst = false, bool reverseSecond = false);
glm::vec2 findIntersectionOld(const Curve &first, const Curve &second);

