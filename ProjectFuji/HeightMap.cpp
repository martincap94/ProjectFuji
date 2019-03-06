#include "HeightMap.h"

#include <iostream>
#include <fstream>
#include <glm\glm.hpp>
#include <vector>
#include "ppmImage.h"

HeightMap::HeightMap() {}

HeightMap::HeightMap(string filename, int latticeHeight, ShaderProgram *shader) : shader(shader) {

	ppmImage helper(SCENES_DIR + filename);

	width = helper.width;
	height = helper.height;
	maxIntensity = helper.maxIntensity;
	int maxSum = maxIntensity * 3;


	data = new float*[width]();
	for (int i = 0; i < width; i++) {
		data[i] = new float[height]();
	}
	vector<glm::vec3> areaPoints;

	for (int z = height - 1; z >= 0; z--) {
		for (int x = 0; x < width; x++) {
			int sum = 0;

			sum += helper.data[x][z].x;
			sum += helper.data[x][z].y;
			sum += helper.data[x][z].z;

			data[x][z] = ((float)sum / (float)maxSum) * (float)(latticeHeight - 1);
		}
	}



	/*
	vector<glm::vec3> triangles;
	vector<glm::vec3> normals;
	*/
	bool flatShading = false;

	// it would be useful to create a DCEL for the terrain so it would be much easier to modify later on
	for (int z = height - 1; z >= 1; z--) {
		for (int x = 0; x < width - 1; x++) {

			glm::vec3 p1(x, data[x][z], z);
			glm::vec3 p2(x + 1, data[x + 1][z], z);
			glm::vec3 p3(x + 1, data[x + 1][z - 1], z - 1);
			glm::vec3 p4(x, data[x][z - 1], z - 1);

			//glm::vec3 n1 = glm::normalize(glm::cross(p1 - p2, p3 - p2));
			//glm::vec3 n2 = glm::normalize(glm::cross(p3 - p4, p1 - p4)); // flat shading normals

			glm::vec3 n1 = glm::normalize(glm::cross(p2 - p3, p2 - p1));
			glm::vec3 n2 = glm::normalize(glm::cross(p4 - p1, p4 - p3)); // flat shading normals

			if (!flatShading) {
				glm::vec3 normalP1 = computeNormal(x, z);
				glm::vec3 normalP2 = computeNormal(x + 1, z);
				glm::vec3 normalP3 = computeNormal(x + 1, z - 1);
				glm::vec3 normalP4 = computeNormal(x, z - 1);


				areaPoints.push_back(p1);
				areaPoints.push_back(normalP1);
				areaPoints.push_back(p2);
				areaPoints.push_back(normalP2);
				areaPoints.push_back(p3);
				areaPoints.push_back(normalP3);


				areaPoints.push_back(p3);
				areaPoints.push_back(normalP3);
				areaPoints.push_back(p4);
				areaPoints.push_back(normalP4);
				areaPoints.push_back(p1);
				areaPoints.push_back(normalP1);
			} else {



				// FLAT SHADING APPROACH


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

			}


			/*
			triangles.push_back(p1);
			triangles.push_back(p2);
			triangles.push_back(p3);

			triangles.push_back(p3);
			triangles.push_back(p4);
			triangles.push_back(p1);
			*/


			//areaPoints.push_back(glm::vec3(x, data[x][z], z));
			//areaPoints.push_back(glm::vec3(x + 1, data[x + 1][z], z));
			//areaPoints.push_back(glm::vec3(x + 1, data[x + 1][z - 1], z - 1));
			//
			//areaPoints.push_back(glm::vec3(x + 1, data[x + 1][z - 1], z - 1));
			//areaPoints.push_back(glm::vec3(x, data[x][z - 1], z - 1));
			//areaPoints.push_back(glm::vec3(x, data[x][z], z));


		}
	}

	/*
	for (int i = 0; i < triangles.size(); i++) {
		areaPoints.push_back(triangles[i]);
		areaPoints.push_back(normals[i]);
	}
	*/


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
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glUseProgram(shader->id);
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, numPoints);
	
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

void HeightMap::draw(ShaderProgram *shader) {
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glUseProgram(shader->id);
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, numPoints);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

// Based on: https://stackoverflow.com/questions/49640250/calculate-normals-from-heightmap
glm::vec3 HeightMap::computeNormal(int x, int z) {

	int offset = 1;

	int xLeft = x - offset;
	int xRight = x + offset;
	int zBottom = z + offset;
	int zTop = z - offset;

	if (xLeft < 0) {
		xLeft = 0;
	}
	if (xRight > width - 1) {
		xRight = width - 1;
	}
	if (zBottom > height - 1) {
		zBottom = height - 1;
	}
	if (zTop < 0) {
		zTop = 0;
	}
	float hLeft = data[xLeft][z];
	float hRight = data[xRight][z];
	float hBottom = data[x][zBottom];
	float hTop = data[x][zTop];

	glm::vec3 normal;
	normal.x = hLeft - hRight;
	//normal.y = hBottom - hTop;
	//normal.z = -2.0f;
	normal.y = 2.0f;
	normal.z = hTop - hBottom;

	return glm::normalize(normal);
}












