#include "HeightMap.h"

#include <iostream>
#include <fstream>
#include <glm\glm.hpp>
#include <vector>
#include "ppmImage.h"
#include "VariableManager.h"
#include "ShaderManager.h"
#include "TextureManager.h"
#include "Utils.h"

HeightMap::HeightMap() {}

HeightMap::HeightMap(string filename, int latticeHeight) {


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


	initMaterials();

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

void HeightMap::initMaterials() {

	CHECK_GL_ERRORS();

	shader = ShaderManager::getShaderPtr("terrain");
	shader->use();

	materials[0].diffuseTexture = TextureManager::loadTexture("textures/Ground_Dirt_006_COLOR.jpg");
	materials[0].normalMap = TextureManager::loadTexture("textures/Ground_Dirt_006_NORM.jpg");
	materials[0].shininess = 2.0f;
	materials[0].textureTiling = 80.0f;

	materials[1].diffuseTexture = TextureManager::loadTexture("textures/ROCK_030_COLOR.jpg");
	materials[1].normalMap = TextureManager::loadTexture("textures/ROCK_030_NORM.jpg");
	materials[1].shininess = 16.0f;
	materials[1].textureTiling = 20.0f;

	//materials[2].diffuseTexture = TextureManager::loadTexture("mossy-ground1-albedo.png");
	//materials[2].normalMap = TextureManager::loadTexture("mossy-ground1-preview.png");
	CHECK_GL_ERRORS();

	for (int i = 0; i < MAX_TERRAIN_MATERIALS; i++) {
		materials[i].setTextureUniformsMultiple(shader, i);
	}
	shader->setInt("u_MaterialMap", 12);
	shader->setInt("u_TerrainNormalMap", 11);
	shader->setFloat("u_UVRatio", (float)width / (float)height);
	
	CHECK_GL_ERRORS();


	diffuseTexture = new Texture("textures/Ground_Dirt_006_COLOR.jpg");
	normalMap = new Texture("textures/Ground_Dirt_006_NORM.jpg");

	secondDiffuseTexture = new Texture("textures/ROCK_030_COLOR.jpg");
	secondNormalMap = new Texture("textures/ROCK_030_NORM.jpg");

	testDiffuse = new Texture("textures/Rock_030_COLOR.jpg");
	terrainNormalMap = new Texture("textures/ROCK_030_NORM.jpg");
	materialMap = new Texture("textures/1200x800_materialMap.png");



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
			texCoords.push_back(glm::vec2(p1.x / width, p1.z / height));
			//texCoords.push_back(glm::vec2(p1.x, p1.z) / den);

			vertices.push_back(p2);
			normals.push_back(normalP2);
			texCoords.push_back(glm::vec2(p2.x / width, p2.z / height));
			//texCoords.push_back(glm::vec2(p2.x, p2.z) / den);

			vertices.push_back(p3);
			normals.push_back(normalP3);
			texCoords.push_back(glm::vec2(p3.x / width, p3.z / height));
			//texCoords.push_back(glm::vec2(p3.x, p3.z) / den);

			vertices.push_back(p3);
			normals.push_back(normalP3);
			texCoords.push_back(glm::vec2(p3.x / width, p3.z / height));
			//texCoords.push_back(glm::vec2(p3.x, p3.z) / den);

			vertices.push_back(p4);
			normals.push_back(normalP4);
			texCoords.push_back(glm::vec2(p4.x / width, p4.z / height));
			//texCoords.push_back(glm::vec2(p4.x, p4.z) / den);

			vertices.push_back(p1);
			normals.push_back(normalP1);
			texCoords.push_back(glm::vec2(p1.x / width, p1.z / height));
			//texCoords.push_back(glm::vec2(p1.x, p1.z) / den);

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


	//cout << vertices.size() << " ... " << numPoints << endl;

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

float HeightMap::getHeight(float x, float z) {

	int leftx = (int)x;
	int rightx = leftx + 1;
	int leftz = (int)z;
	int rightz = leftz + 1;

	leftx = glm::clamp(leftx, 0, width - 1);
	rightx = glm::clamp(rightx, 0, width - 1);
	leftz = glm::clamp(leftz, 0, height - 1);
	rightz = glm::clamp(rightz, 0, height - 1);


	float xRatio = x - leftx;
	float zRatio = z - leftz;

	float y1 = data[leftx][leftz];
	float y2 = data[leftx][rightz];
	float y3 = data[rightx][leftz];
	float y4 = data[rightx][rightz];

	float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
	float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;

	float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;

	return y;

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

	shader->use();


	// Set texture uniforms for unknown shader
	if (shader != this->shader) {
		for (int i = 0; i < MAX_TERRAIN_MATERIALS; i++) {
			materials[i].setTextureUniformsMultiple(shader, i);
		}
		shader->setInt("u_TerrainNormalMap", 11);
		shader->setInt("u_MaterialMap", 12);
		shader->setFloat("u_UVRatio", (float)width / (float)height);
	}



	for (int i = 0; i < MAX_TERRAIN_MATERIALS; i++) {
		materials[i].useMultiple(shader, i);
	}


	shader->setInt("u_TestDiffuse", 9);
	shader->setInt("u_DepthMapTexture", TEXTURE_UNIT_DEPTH_MAP);

	shader->setFloat("u_GlobalNormalMapMixingRatio", vars->globalNormalMapMixingRatio);



	glBindTextureUnit(9, testDiffuse->id);

	glBindTextureUnit(11, terrainNormalMap->id);
	glBindTextureUnit(12, materialMap->id);


	/*
	shader->setInt("u_NumActiveMaterials", 2);

	shader->setInt("u_Materials[0].diffuse", 0);
	shader->setInt("u_Materials[0].specular", 1);
	shader->setInt("u_Materials[0].normalMap", 2);
	shader->setInt("u_TestDiffuse", 6);
	shader->setInt("u_DepthMapTexture", TEXTURE_UNIT_DEPTH_MAP);

	shader->setFloat("u_Materials[0].shininess", 2.0f);
	shader->setFloat("u_Materials[0].tiling", vars->terrainTextureTiling);

	shader->setInt("u_Materials[1].diffuse", 3);
	shader->setInt("u_Materials[1].specular", 4);
	shader->setInt("u_Materials[1].normalMap", 5);

	shader->setFloat("u_Materials[1].shininess", 2.0f);
	shader->setFloat("u_Materials[1].tiling", vars->terrainTextureTiling);


	shader->setInt("u_TerrainNormalMap", 11);
	shader->setFloat("u_GlobalNormalMapMixingRatio", vars->globalNormalMapMixingRatio);


	shader->setInt("u_MaterialMap", 12);

	shader->setFloat("u_UVRatio", (float)width / (float)height);


	glBindTextureUnit(0, diffuseTexture->id);
	glBindTextureUnit(2, normalMap->id);

	glBindTextureUnit(3, secondDiffuseTexture->id);
	glBindTextureUnit(5, secondNormalMap->id);

	glBindTextureUnit(6, testDiffuse->id);

	glBindTextureUnit(11, terrainNormalMap->id);
	glBindTextureUnit(12, materialMap->id);
	*/

	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, numPoints);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

void HeightMap::drawGeometry(ShaderProgram * shader) {
	shader->use();

	shader->setModelMatrix(glm::mat4(1.0f));
	shader->setBool("u_IsInstanced", false);

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











