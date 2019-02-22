#include "HeightMap.h"

#include <iostream>
#include <fstream>
#include <glm\glm.hpp>
#include <vector>

HeightMap::HeightMap() {}

HeightMap::HeightMap(string filename, int latticeHeight, ShaderProgram *shader) : shader(shader) {


	if (filename.find(".ppm") == string::npos) {
		cerr << "HeightMap only accepts .ppm files!" << endl;
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

	data = new float*[width]();
	for (int i = 0; i < width; i++) {
		data[i] = new float[height]();
	}

	int maxSum = maxIntensity * 3;

	int dummy;
	vector<glm::vec3> areaPoints;
	for (int z = height - 1; z >= 0; z--) {
		for (int x = 0; x < width; x++) {
			int sum = 0;

			inFile >> dummy; // take red channel only
			sum += dummy;
			inFile >> dummy;
			sum += dummy;
			inFile >> dummy;
			sum += dummy;


			data[x][z] = ((float)sum / (float)maxSum) * (float)(latticeHeight - 1);

			//printf("x = %d, z = %d, data[x][z] = %f\n", x, z, data[x][z]);

			//areaPoints.push_back(glm::vec3(x, data[x][z], z));

		}
	}

	for (int z = height - 1; z >= 1; z--) {
		for (int x = 0; x < width - 1; x++) {

			glm::vec3 p1(x, data[x][z], z);
			glm::vec3 p2(x + 1, data[x + 1][z], z);
			glm::vec3 p3(x + 1, data[x + 1][z - 1], z - 1);
			glm::vec3 p4(x, data[x][z - 1], z - 1);

			glm::vec3 n1 = glm::normalize(glm::cross(p1 - p2, p3 - p2));
			glm::vec3 n2 = glm::normalize(glm::cross(p3 - p4, p1 - p4)); // flat shading normals

			areaPoints.push_back(p1);
			areaPoints.push_back(n1);
			areaPoints.push_back(p2);
			areaPoints.push_back(n1);
			areaPoints.push_back(p3);
			areaPoints.push_back(n1);


			areaPoints.push_back(p3);
			areaPoints.push_back(n2);
			areaPoints.push_back(p4);
			areaPoints.push_back(n2);
			areaPoints.push_back(p1);
			areaPoints.push_back(n2);


			//areaPoints.push_back(glm::vec3(x, data[x][z], z));
			//areaPoints.push_back(glm::vec3(x + 1, data[x + 1][z], z));
			//areaPoints.push_back(glm::vec3(x + 1, data[x + 1][z - 1], z - 1));
			//
			//areaPoints.push_back(glm::vec3(x + 1, data[x + 1][z - 1], z - 1));
			//areaPoints.push_back(glm::vec3(x, data[x][z - 1], z - 1));
			//areaPoints.push_back(glm::vec3(x, data[x][z], z));


		}
	}



	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * areaPoints.size(), &areaPoints[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)sizeof(glm::vec3));

	glBindVertexArray(0);

	numPoints = areaPoints.size();



}


HeightMap::~HeightMap() {
	cout << "DELETING HEIGHTMAP" << endl;
	for (int i = 0; i < width; i++) {
		delete[] data[i];
	}
	delete[] data;
}

void HeightMap::draw() {
	glUseProgram(shader->id);
	glBindVertexArray(VAO);

	//glPointSize(5.0f);
	//shader.setVec3("uColor", glm::vec3(1.0f, 1.0f, 0.4f));

	glDrawArrays(GL_TRIANGLES, 0, numPoints);


}
