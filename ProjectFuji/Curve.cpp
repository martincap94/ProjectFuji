#include "Curve.h"

#include <iostream>

#include "Utils.h"
#include <glm\gtc\epsilon.hpp>


Curve::Curve() {
}


Curve::~Curve() {
}

void Curve::initBuffers() {
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);

}

void Curve::uploadToBuffers() {
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
}

void Curve::init() {
	initBuffers();
	uploadToBuffers();
}

void Curve::draw(ShaderProgram & shader) {
	glUseProgram(shader.id);

	shader.setColor(color);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, vertices.size() * 2);
}

void Curve::draw(ShaderProgram *shader) {
	shader->use();
	shader->setColor(color);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, vertices.size() * 2);
}

glm::vec2 Curve::getIntersectionWithIsobar(float normalizedPressure) {

	// naively search for correct interval - better solutions are: binary search and direct indexation using (non-normalized) pressure - see GPU implementation for optimized version!
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

void Curve::printVertices() {
	for (int i = 0; i < vertices.size(); i++) {
		printVec2(vertices[i]);
	}
}



#define CURVE_START_EPSILON 0.0001f

bool findIntersectionNew(const Curve & c1, const Curve & c2, glm::vec2 & outIntersection, unsigned int intersectionNumber, bool skipStartingPoints, bool reverseFirst, bool reverseSecond) {
	int iStart, jStart, iEnd, jEnd, iDelta, jDelta;
	if (reverseFirst) {
		iStart = c1.vertices.size() - 1;
		iEnd = 0;
		iDelta = -1;
	} else {
		iStart = 0;
		iEnd = c1.vertices.size() - 1;
		iDelta = 1;
	}

	if (reverseSecond) {
		jStart = c2.vertices.size() - 1;
		jEnd = 0;
		jDelta = -1;
	} else {
		jStart = 0;
		jEnd = c2.vertices.size() - 1;
		jDelta = 1;
	}
	unsigned int foundIntersectionCounter = 0;
	for (int i = iStart; i != iEnd; i += iDelta) {
		for (int j = jStart; j != jEnd; j += jDelta) {
			glm::vec2 intersection;
			if (getIntersectionPoint(c1.vertices[i], c1.vertices[i + iDelta], c2.vertices[j], c2.vertices[j + jDelta], intersection)) {
				if (skipStartingPoints && 
					(glm::all(glm::epsilonEqual(intersection, c1.vertices[iStart], CURVE_START_EPSILON)) || 
					 glm::all(glm::epsilonEqual(intersection, c2.vertices[jStart], CURVE_START_EPSILON)))) {
					continue;
				}
				foundIntersectionCounter++;
				outIntersection = intersection;
				if (foundIntersectionCounter == intersectionNumber) {
					return true;
				}
			}
		}
	}

	return false;
}

glm::vec2 findIntersection(const Curve & c1, const Curve & c2, bool reverseFirst, bool reverseSecond) {
	// Test each edge pair ( O(|E|^2) )

	//static bool firstFlag = false;
	//if (firstFlag == false) {
	//	firstFlag = true;
	//} else {
	//	return glm::vec2(0.0f);
	//}

	int iStart, jStart, iEnd, jEnd, iDelta, jDelta;
	if (reverseFirst) {
		iStart = c1.vertices.size() - 1;
		iEnd = 0;
		iDelta = -1;
	} else {
		iStart = 0;
		iEnd = c1.vertices.size() - 1;
		iDelta = 1;
	}

	if (reverseSecond) {
		jStart = c2.vertices.size() - 1;
		jEnd = 0;
		jDelta = -1;
	} else {
		jStart = 0;
		jEnd = c2.vertices.size() - 1;
		jDelta = 1;
	}

	for (int i = iStart; i != iEnd; i += iDelta) {
		for (int j = jStart; j != jEnd; j += jDelta) {
			glm::vec2 intersection;
			if (getIntersectionPoint(c1.vertices[i], c1.vertices[i + iDelta], c2.vertices[j], c2.vertices[j + jDelta], intersection)) {
				return intersection;
			}
		}
	}
	//for (int i = 0; i < c1.vertices.size() - 1; i++) {
	//	for (int j = 0; j < c2.vertices.size() - 1; j++) {
	//		glm::vec2 intersection;
	//		cout << i << " <-> " << j << endl;
	//		if (getIntersectionPoint(c1.vertices[i], c1.vertices[i + 1], c2.vertices[j], c2.vertices[j + 1], intersection)) {
	//			cout << "success" << endl;
	//			return intersection;
	//		}
	//	}
	//}
	
	return glm::vec2(0.0f); // magic (origin of diagram)
}


// Based on: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/
glm::vec2 findIntersectionOld(const Curve & c1, const Curve & c2) {
	for (int i = 0; i < c1.vertices.size() - 1; i++) {
		for (int j = 0; j < c2.vertices.size() - 1; j++) {
			bool result = doLineSegmentsIntersect(c1.vertices[i], c1.vertices[i + 1], c2.vertices[j], c2.vertices[j + 1]);
			if (result) {
				cout << "Lines intersect! Looking for intersection point..." << endl;
				return getIntersectionPoint(c1.vertices[i], c1.vertices[i + 1], c2.vertices[j], c2.vertices[j + 1]);
			}
		}
	}
	return glm::vec2(0.0f); // random
}
