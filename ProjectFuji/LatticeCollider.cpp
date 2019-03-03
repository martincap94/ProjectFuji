#include "LatticeCollider.h"

#include <iostream>
#include <fstream>
#include <glm\glm.hpp>
#include <vector>
#include "ShaderProgram.h"


LatticeCollider::LatticeCollider(string filename) {

	if (filename.find(".ppm") == string::npos) {
		cerr << "Lattice Collider only accepts .ppm files!" << endl;
		exit(-1);
		//return;
	}
	ifstream inFile(SCENES_DIR + filename);
	string line;

	getline(inFile, line);
	if (line != "P3") {
		cerr << "We require .ppm files in P3 format (ASCII)." << endl;
		return;
	}

	getline(inFile, line);
	/*while (line.rfind("#", 0) != 0) {
		getline(inFile, line);
	}*/

	inFile >> width;
	inFile >> height;
	inFile >> maxIntensity;
	cout << "Width = " << width << ", height = " << height << ", max intesity = " << maxIntensity << endl;

	area = new bool[width * height]();
	//for (int i = 0; i < width; i++) {
	//	area[i] = new bool[height]();
	//}

	int dummy;
	vector<glm::vec3> areaPoints;
	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {

			inFile >> dummy; // take red channel only
			area[x + width * y] = (bool)dummy;
			if (area[x + width * y]) {
				areaPoints.push_back(glm::vec3(x, y, 0.0f));
			}
			inFile >> dummy;
			inFile >> dummy;
		}
	}
	numPoints = areaPoints.size();



	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * areaPoints.size(), &areaPoints[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);


}


LatticeCollider::~LatticeCollider() {
	delete[] area;
}

void LatticeCollider::draw(ShaderProgram & shader) {

	glUseProgram(shader.id);
	glBindVertexArray(VAO);

	glPointSize(2.0f);
	shader.setVec3("u_Color", glm::vec3(1.0f, 1.0f, 0.4f));

	glDrawArrays(GL_POINTS, 0, numPoints);


}
