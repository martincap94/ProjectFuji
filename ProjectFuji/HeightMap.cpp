#include "HeightMap.h"

#include <iostream>
#include <fstream>
#include <glm\glm.hpp>
#include <vector>
#include "ppmImage.h"
#include "VariableManager.h"

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

	for (int z = height - 1; z >= 0; z--) {
		for (int x = 0; x < width; x++) {
			int sum = 0;

			sum += helper.data[x][z].x;
			sum += helper.data[x][z].y;
			sum += helper.data[x][z].z;

			data[x][z] = ((float)sum / (float)maxSum) * (float)(latticeHeight - 1);
		}
	}
	//initBuffersOld();
	initBuffers();




	diffuseTexture = new Texture("textures/Ground_Dirt_006_COLOR.jpg");
	normalMap = new Texture("textures/Ground_Dirt_006_NORM.jpg");
	testDiffuse = new Texture("textures/Rock_030_COLOR.jpg");



}


HeightMap::~HeightMap() {
	cout << "DELETING HEIGHTMAP" << endl;
	for (int i = 0; i < width; i++) {
		delete[] data[i];
	}
	delete[] data;
	if (diffuseTexture) {
		delete diffuseTexture;
	}
}

void HeightMap::initBuffers() {

	vector<glm::vec3> vertices;
	vector<glm::vec3> normals;
	vector<glm::vec2> texCoords;
	vector<glm::vec3> tangents;
	vector<glm::vec3> bitangents;

	vector<float> vertexData;

	//int size = 3 * width * height;

	//vertices.reserve(size);
	//normals.reserve(size);
	//texCoords.reserve(size);
	//tangents.reserve(size);
	//bitangents.reserve(size);

	//vertexData.reserve(size * 15);



	numPoints = 0;

	float den = (float)((width >= height) ? width : height);

	for (int z = height - 1; z >= 1; z--) {
		for (int x = 0; x < width - 1; x++) {

			glm::vec3 p1(x, data[x][z], z);
			glm::vec3 p2(x + 1, data[x + 1][z], z);
			glm::vec3 p3(x + 1, data[x + 1][z - 1], z - 1);
			glm::vec3 p4(x, data[x][z - 1], z - 1);


			glm::vec3 normalP1 = computeNormal(x, z);
			glm::vec3 normalP2 = computeNormal(x + 1, z);
			glm::vec3 normalP3 = computeNormal(x + 1, z - 1);
			glm::vec3 normalP4 = computeNormal(x, z - 1);

			vertices.push_back(p1);
			normals.push_back(normalP1);
			texCoords.push_back(glm::vec2(p1.x, p1.z) / den);

			vertices.push_back(p2);
			normals.push_back(normalP2);
			texCoords.push_back(glm::vec2(p2.x, p2.z) / den);

			vertices.push_back(p3);
			normals.push_back(normalP3);
			texCoords.push_back(glm::vec2(p3.x, p3.z) / den);

			vertices.push_back(p3);
			normals.push_back(normalP3);
			texCoords.push_back(glm::vec2(p3.x, p3.z) / den);

			vertices.push_back(p4);
			normals.push_back(normalP4);
			texCoords.push_back(glm::vec2(p4.x, p4.z) / den);

			vertices.push_back(p1);
			normals.push_back(normalP1);
			texCoords.push_back(glm::vec2(p1.x, p1.z) / den);

			numPoints += 6;
		}
	}


	//cout << numPoints << " ... " << size << endl;
	// Based on: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
	for (int i = 0; i < vertices.size() - 2; i += 3) {
		glm::vec3 edge1 = vertices[i + 1] - vertices[i];
		glm::vec3 edge2 = vertices[i + 2] - vertices[i];
		glm::vec2 deltaUV1 = texCoords[i + 1] - texCoords[i];
		glm::vec2 deltaUV2 = texCoords[i + 2] - texCoords[i];
		float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

		glm::vec3 tangent;
		tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
		tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
		tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
		tangent = glm::normalize(tangent);

		glm::vec3 bitangent;
		bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
		bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
		bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
		bitangent = glm::normalize(bitangent);

		tangents.push_back(tangent);
		tangents.push_back(tangent);
		tangents.push_back(tangent);

		bitangents.push_back(bitangent);
		bitangents.push_back(bitangent);
		bitangents.push_back(bitangent);


	}


	cout << vertices.size() << " ... " << numPoints << endl;

	// merge together
	for (int i = 0; i < numPoints; i++) {
		vertexData.push_back(vertices[i].x);
		vertexData.push_back(vertices[i].y);
		vertexData.push_back(vertices[i].z);
		vertexData.push_back(normals[i].x);
		vertexData.push_back(normals[i].y);
		vertexData.push_back(normals[i].z);
		vertexData.push_back(texCoords[i].x);
		vertexData.push_back(texCoords[i].y);
		vertexData.push_back(tangents[i].x);
		vertexData.push_back(tangents[i].y);
		vertexData.push_back(tangents[i].z);
		vertexData.push_back(bitangents[i].x);
		vertexData.push_back(bitangents[i].y);
		vertexData.push_back(bitangents[i].z);
	}

	
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);


	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexData.size(), vertexData.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void *)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void *)(sizeof(float) * 3));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void *)(sizeof(float) * 6));


	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void *)(sizeof(float) * 8));

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void *)(sizeof(float) * 11));


	

	glBindVertexArray(0);
}

void HeightMap::initBuffersOld() {


	vector<glm::vec3> areaPoints;
	vector<float> vertexData; // for texture coordinates (which are vec2)



							  /*
							  vector<glm::vec3> triangles;
							  vector<glm::vec3> normals;
							  */
	bool uploadTextureCoordinates = true;

	bool flatShading = false;

	if (uploadTextureCoordinates && flatShading) {
		cerr << "Upload of texture coordinates not possible with flat shading!" << endl;
		uploadTextureCoordinates = false;
	}

	numPoints = 0;

	// it would be useful to create a DCEL for the terrain so it would be much easier to modify later on
	for (int z = height - 1; z >= 1; z--) {
		for (int x = 0; x < width - 1; x++) {

			float den = (float)((width >= height) ? width : height);

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


				if (uploadTextureCoordinates) {



					vertexData.push_back(p1.x);
					vertexData.push_back(p1.y);
					vertexData.push_back(p1.z);

					vertexData.push_back(normalP1.x);
					vertexData.push_back(normalP1.y);
					vertexData.push_back(normalP1.z);

					vertexData.push_back(p1.x / den);
					vertexData.push_back(p1.z / den);

					vertexData.push_back(p2.x);
					vertexData.push_back(p2.y);
					vertexData.push_back(p2.z);

					vertexData.push_back(normalP2.x);
					vertexData.push_back(normalP2.y);
					vertexData.push_back(normalP2.z);

					vertexData.push_back(p2.x / den);
					vertexData.push_back(p2.z / den);

					vertexData.push_back(p3.x);
					vertexData.push_back(p3.y);
					vertexData.push_back(p3.z);

					vertexData.push_back(normalP3.x);
					vertexData.push_back(normalP3.y);
					vertexData.push_back(normalP3.z);

					vertexData.push_back(p3.x / den);
					vertexData.push_back(p3.z / den);




					vertexData.push_back(p3.x);
					vertexData.push_back(p3.y);
					vertexData.push_back(p3.z);

					vertexData.push_back(normalP3.x);
					vertexData.push_back(normalP3.y);
					vertexData.push_back(normalP3.z);

					vertexData.push_back(p3.x / den);
					vertexData.push_back(p3.z / den);

					vertexData.push_back(p4.x);
					vertexData.push_back(p4.y);
					vertexData.push_back(p4.z);

					vertexData.push_back(normalP4.x);
					vertexData.push_back(normalP4.y);
					vertexData.push_back(normalP4.z);

					vertexData.push_back(p4.x / den);
					vertexData.push_back(p4.z / den);

					vertexData.push_back(p1.x);
					vertexData.push_back(p1.y);
					vertexData.push_back(p1.z);

					vertexData.push_back(normalP1.x);
					vertexData.push_back(normalP1.y);
					vertexData.push_back(normalP1.z);

					vertexData.push_back(p1.x / den);
					vertexData.push_back(p1.z / den);


				} else {
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
				}
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
			numPoints += 6;



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

	for (int i = 0; i < vertexData.size(); i += 8) {

		//glm::vec3 edge1 = 

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

	if (uploadTextureCoordinates) {

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexData.size(), vertexData.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(sizeof(float) * 3));

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(sizeof(float) * 6));


	} else {
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * areaPoints.size(), &areaPoints[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)0);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void *)sizeof(glm::vec3));

	}

	glBindVertexArray(0);

}

void HeightMap::draw() {
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	//glUseProgram(shader->id);
	//glBindVertexArray(VAO);
	//glDrawArrays(GL_TRIANGLES, 0, numPoints);
	draw(shader);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

void HeightMap::draw(ShaderProgram *shader) {
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glUseProgram(shader->id);

	shader->setInt("u_Material.diffuse", 1);
	shader->setInt("u_Material.normalMap", 3);
	shader->setInt("u_TestDiffuse", 4);
	shader->setFloat("u_Material.tiling", vars->terrainTextureTiling);

	glBindTextureUnit(1, diffuseTexture->id);
	glBindTextureUnit(3, normalMap->id);
	glBindTextureUnit(4, testDiffuse->id);

	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, numPoints);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

void HeightMap::drawGeometry(ShaderProgram * shader) {
	glUseProgram(shader->id);
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, numPoints);
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












