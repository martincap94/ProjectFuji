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

	glm::vec2 getIntersectionWithIsobar(float normalizedPressure);


private:

	GLuint VAO;
	GLuint VBO;


};



glm::vec2 findIntersectionNaive(const Curve &first, const Curve &second);
glm::vec2 findIntersectionAlt(const Curve &first, const Curve &second);
