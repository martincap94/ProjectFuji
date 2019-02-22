#include "Curve.h"

#include <iostream>

#include "Utils.h"


Curve::Curve() {
}


Curve::~Curve() {
}

void Curve::initBuffers() {
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);

}

void Curve::draw(ShaderProgram & shader) {

	glUseProgram(shader.id);

	glLineWidth(4.0f);
	shader.setVec3("color", glm::vec3(0.1f, 0.1f, 0.1f));
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, vertices.size() * 2);
}

glm::vec2 Curve::getIntersectionWithIsobar(float normalizedPressure) {

	// naively search for correct interval - better solutions are: binary search and direct indexation using (non-normalized) pressure - needs better design
	for (int i = 0; i < vertices.size() - 1; i += 1) {
		//cout << vertices[i].y << endl;
		if (vertices[i + 1].y > normalizedPressure) {
			continue;
		}
		if (vertices[i + 1].y <= normalizedPressure) {
			//cout << "Intersection interval found! It is " << vertices[i].y << " to " << vertices[i + 1].y << endl;

			// solve lin. interpolation
			float t = (normalizedPressure - vertices[i + 1].y) / (vertices[i].y - vertices[i + 1].y);
			//cout << "t = " << t << endl;
			float normalizedTemperature = t * vertices[i].x + (1.0f - t) * vertices[i + 1].x;
			//cout << "normalized temperature = " << normalizedTemperature << endl;

			return glm::vec2(normalizedTemperature, normalizedPressure);
		}
	}

	return glm::vec2();
}





// Based on: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/
glm::vec2 findIntersectionNaive(const Curve & first, const Curve & second) {
	// Test each edge pair ( O(|E|^2) )
	for (int i = 0; i < first.vertices.size() - 1; i++) {
		for (int j = 0; j < second.vertices.size() - 1; j++) {
			bool result = doLineSegmentsIntersect(first.vertices[i], first.vertices[i + 1], second.vertices[j], second.vertices[j + 1]);
			if (result) {
				cout << "Lines intersect! Looking for intersection point..." << endl;
				return getIntersectionPoint(first.vertices[i], first.vertices[i + 1], second.vertices[j], second.vertices[j + 1]);
			}
		}



	}
	return glm::vec2(-2.0f); // random
}

glm::vec2 findIntersectionAlt(const Curve & first, const Curve & second) {
	return glm::vec2();
}
