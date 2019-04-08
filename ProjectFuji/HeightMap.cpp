#include "HeightMap.h"

#include <iostream>
#include <fstream>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

#include <vector>
#include "ppmImage.h"
#include "VariableManager.h"
#include "ShaderManager.h"
#include "TextureManager.h"
#include "Utils.h"
#include <stb_image.h>
#include <limits>
#include <set>


HeightMap::HeightMap() {}


HeightMap::HeightMap(VariableManager * vars) : vars(vars) {

	if (!vars) {
		cerr << "Oh noes - VariableManager not set in HeightMap!" << endl;
		exit(EXIT_FAILURE);
	}
	terrainHeightRange = vars->terrainHeightRange;
	heightMapFilename = SCENES_DIR + vars->sceneFilename;
	//exit(EXIT_FAILURE); // TESTING!!!

	

	//initBuffersOld();
	initBuffers();
	loadHeightMapData();
	createAndUploadMesh();

	initMaterials();

}

void HeightMap::loadHeightMapData() {
	loadHeightMapData(heightMapFilename);
}

void HeightMap::loadHeightMapData(std::string filename) {

	if (data != nullptr) {
		delete[] data;
	}

	typedef unsigned char img_type;

	cout << "Creating Terrain..." << endl;


	unsigned short *imageData = nullptr;


	int numChannels;

	stbi_set_flip_vertically_on_load(true);
	imageData = stbi_load_16((filename).c_str(), &width, &height, &numChannels, NULL);


	if (!imageData) {
		cout << "Error loading texture at " << filename << endl;
		stbi_image_free(imageData);
		exit(EXIT_FAILURE);
	}

	data = new float[width * height]();




	cout << "number of channels = " << numChannels << endl;

	cout << "Width: " << width << ", height: " << height << endl;
	cout << (float)numeric_limits<unsigned short>().max() << endl;

	for (int z = 0; z < height; z++) {
		for (int x = 0; x < width; x++) {
			unsigned short *pixel = imageData + (x * height + z) * numChannels;
			unsigned short val = pixel[0];
			unsigned short a = 0xff;
			if (numChannels >= 1) {
				a = pixel[1];
			}

			data[x + z * width] = (float)val;
			data[x + z * width] /= (float)numeric_limits<unsigned short>().max();
			rangeToRange(data[x + z * width], 0.0f, 1.0f, terrainHeightRange.x, terrainHeightRange.y);
		}
	}

	if (imageData != nullptr) {
		stbi_image_free(imageData);
	}


}

void HeightMap::createAndUploadMesh() {

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



			float tws = vars->texelWorldSize;
			glm::vec3 p1(x * tws, data[x + z * width], z * tws);
			glm::vec3 p2((x + 1) * tws, data[x + 1 + z * width], z * tws);
			glm::vec3 p3((x + 1) * tws, data[x + 1 + (z - 1) * width], (z - 1) * tws);
			glm::vec3 p4(x * tws, data[x + (z - 1) * width], (z - 1) * tws);


			glm::vec3 p1i(x, data[x + z * width], z);
			glm::vec3 p2i(x + 1, data[x + 1 + z * width], z);
			glm::vec3 p3i(x + 1, data[x + 1 + (z - 1) * width], z - 1);
			glm::vec3 p4i(x, data[x + (z - 1) * width], z - 1);

			//glm::vec3 n1 = glm::normalize(glm::cross(p2 - p3, p2 - p1));
			//glm::vec3 n2 = glm::normalize(glm::cross(p4 - p1, p4 - p3)); // flat shading normals



			glm::vec3 normalP1 = computeNormal(x, z);
			glm::vec3 normalP2 = computeNormal(x + 1, z);
			glm::vec3 normalP3 = computeNormal(x + 1, z - 1);
			glm::vec3 normalP4 = computeNormal(x, z - 1);



			vertices.push_back(p1);
			normals.push_back(normalP1);
			texCoords.push_back(glm::vec2(p1i.z / width, p1i.x / height));

			vertices.push_back(p2);
			normals.push_back(normalP2);
			texCoords.push_back(glm::vec2(p2i.z / width, p2i.x / height));

			vertices.push_back(p3);
			normals.push_back(normalP3);
			texCoords.push_back(glm::vec2(p3i.z / width, p3i.x / height));

			vertices.push_back(p3);
			normals.push_back(normalP3);
			texCoords.push_back(glm::vec2(p3i.z / width, p3i.x / height));

			vertices.push_back(p4);
			normals.push_back(normalP4);
			texCoords.push_back(glm::vec2(p4i.z / width, p4i.x / height));

			vertices.push_back(p1);
			normals.push_back(normalP1);
			texCoords.push_back(glm::vec2(p1i.z / width, p1i.x / height));

			numPoints += 6;




			/*

			glm::vec3 p1(x, data[x][z], z);
			glm::vec3 p2(x + 1, data[x + 1][z], z);
			glm::vec3 p3(x + 1, data[x + 1][z - 1], z - 1);
			glm::vec3 p4(x, data[x][z - 1], z - 1);


			glm::vec3 n1 = glm::normalize(glm::cross(p2 - p3, p2 - p1));
			glm::vec3 n2 = glm::normalize(glm::cross(p4 - p1, p4 - p3)); // flat shading normals



			glm::vec3 normalP1 = computeNormal(x, z);
			glm::vec3 normalP2 = computeNormal(x + 1, z);
			glm::vec3 normalP3 = computeNormal(x + 1, z - 1);
			glm::vec3 normalP4 = computeNormal(x, z - 1);

			if (true) {


			vertices.push_back(p1);

			normals.push_back(normalP1);
			//normals.push_back(n1);

			texCoords.push_back(glm::vec2(p1.x / width, p1.z / height));
			//texCoords.push_back(glm::vec2(p1.x, p1.z) / den);

			vertices.push_back(p2);
			normals.push_back(normalP2);
			//normals.push_back(n1);

			texCoords.push_back(glm::vec2(p2.x / width, p2.z / height));
			//texCoords.push_back(glm::vec2(p2.x, p2.z) / den);

			vertices.push_back(p3);
			normals.push_back(normalP3);
			//normals.push_back(n1);

			texCoords.push_back(glm::vec2(p3.x / width, p3.z / height));
			//texCoords.push_back(glm::vec2(p3.x, p3.z) / den);

			vertices.push_back(p3);
			normals.push_back(normalP3);
			//normals.push_back(n2);

			texCoords.push_back(glm::vec2(p3.x / width, p3.z / height));
			//texCoords.push_back(glm::vec2(p3.x, p3.z) / den);

			vertices.push_back(p4);
			normals.push_back(normalP4);
			//normals.push_back(n2);

			texCoords.push_back(glm::vec2(p4.x / width, p4.z / height));
			//texCoords.push_back(glm::vec2(p4.x, p4.z) / den);

			vertices.push_back(p1);
			normals.push_back(normalP1);
			//normals.push_back(n2);

			texCoords.push_back(glm::vec2(p1.x / width, p1.z / height));
			//texCoords.push_back(glm::vec2(p1.x, p1.z) / den);
			} else {

			vertices.push_back(p1);
			normals.push_back(n1);
			texCoords.push_back(glm::vec2(p1.x / width, p1.z / height));

			vertices.push_back(p2);
			normals.push_back(n1);
			texCoords.push_back(glm::vec2(p2.x / width, p2.z / height));

			vertices.push_back(p3);
			normals.push_back(n1);
			texCoords.push_back(glm::vec2(p3.x / width, p3.z / height));

			vertices.push_back(p3);
			normals.push_back(n2);
			texCoords.push_back(glm::vec2(p3.x / width, p3.z / height));

			vertices.push_back(p4);
			normals.push_back(n2);
			texCoords.push_back(glm::vec2(p4.x / width, p4.z / height));

			vertices.push_back(p1);
			normals.push_back(n2);
			texCoords.push_back(glm::vec2(p1.x / width, p1.z / height));
			}

			numPoints += 6;
			*/



		}
	}


	//cout << numPoints << " ... " << size << endl;
	// Based on: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
	for (int i = 0; i < vertices.size() - 2; i += 3) {
		glm::vec3 edge1 = vertices[i + 1] - vertices[i];
		glm::vec3 edge2 = vertices[i + 2] - vertices[i];
		//edge1 *= vars->texelWorldSize;
		//edge2 *= vars->texelWorldSize;
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
		//vertexData.push_back(vertices[i].x * vars->texelWorldSize);


		vertexData.push_back(vertices[i].y);


		vertexData.push_back(vertices[i].z);
		//vertexData.push_back(vertices[i].z * vars->texelWorldSize);


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
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexData.size(), vertexData.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


}




// this can be easily parallelized on GPU
void HeightMap::smoothHeights() {

	const float kernel[] = {
		0.00000067,	0.00002292,	0.00019117,	0.00038771,	0.00019117,	0.00002292,	0.00000067,
		0.00002292,	0.00078633,	0.00655965,	0.01330373,	0.00655965,	0.00078633,	0.00002292,
		0.00019117,	0.00655965,	0.05472157,	0.11098164,	0.05472157,	0.00655965,	0.00019117,
		0.00038771,	0.01330373,	0.11098164,	0.22508352,	0.11098164,	0.01330373,	0.00038771,
		0.00019117,	0.00655965,	0.05472157,	0.11098164,	0.05472157,	0.00655965,	0.00019117,
		0.00002292,	0.00078633,	0.00655965,	0.01330373,	0.00655965,	0.00078633,	0.00002292,
		0.00000067,	0.00002292,	0.00019117,	0.00038771,	0.00019117,	0.00002292,	0.00000067
	};


	for (int x = 3; x < width - 3; x++) {
		for (int z = 3; z < height - 3; z++) {

			float sum = 0.0f;
			for (int j = -3; j <= 3; j++) {
				for (int k = -3; k <= 3; k++) {
					float scale = kernel[(3 + j) * 7 + (3 + k)];
					sum += scale * data[x + j + (z + k) * width];
				}
			}
			data[x + z * width] = sum;
		}
	}



	//for (int z = height - 1; z >= 0; z--) {
	//	for (int x = 0; x < width; x++) {


	//	}
	//}

}


HeightMap::~HeightMap() {
	cout << "Deleting heightmap..." << endl;
	delete[] data;
}

void HeightMap::initMaterials() {

	CHECK_GL_ERRORS();

	shader = ShaderManager::getShaderPtr("terrain");
	//shader = ShaderManager::getShaderPtr("singleColor");
	shader->use();

	materials[0].diffuseTexture = TextureManager::getTexturePtr("textures/Ground_Dirt_006_COLOR.jpg");
	materials[0].normalMap = TextureManager::getTexturePtr("textures/Ground_Dirt_006_NORM.jpg");

	// load two other options
	TextureManager::loadTexture("textures/mossy-ground1-albedo.png");
	TextureManager::loadTexture("textures/mossy-ground1-preview.png");

	//materials[0].diffuseTexture = TextureManager::getTexturePtr("textures/mossy-ground1-albedo.png");
	//materials[0].normalMap = TextureManager::getTexturePtr("textures/mossy-ground1-preview.png");

	materials[0].shininess = 2.0f;
	materials[0].textureTiling = 8000.0f;

	materials[1].diffuseTexture = TextureManager::getTexturePtr("textures/ROCK_030_COLOR.jpg");
	materials[1].normalMap = TextureManager::getTexturePtr("textures/ROCK_030_NORM.jpg");
	materials[1].shininess = 16.0f;
	materials[1].textureTiling = 1000.0f;

	//materials[2].diffuseTexture = TextureManager::getTexturePtr("mossy-ground1-albedo.png");
	//materials[2].normalMap = TextureManager::getTexturePtr("mossy-ground1-preview.png");
	CHECK_GL_ERRORS();

	for (int i = 0; i < MAX_TERRAIN_MATERIALS; i++) {
		materials[i].setTextureUniformsMultiple(shader, i);
	}
	shader->setInt("u_MaterialMap", materialMapTextureUnit);
	shader->setInt("u_TerrainNormalMap", normalMapTextureUnit);
	shader->setFloat("u_UVRatio", (float)width / (float)height);

	CHECK_GL_ERRORS();

	terrainNormalMap = TextureManager::loadTexture("textures/ROCK_030_NORM.jpg");
	//materialMap = TextureManager::loadTexture("textures/1200x800_materialMap.png");
	materialMap = TextureManager::loadTexture("materialMaps/materialMap_1024.png");

	visTexture = materialMap;

	//terrainNormalMap = new Texture("textures/ROCK_030_NORM.jpg");
	//materialMap = new Texture("textures/1200x800_materialMap.png");



}




float HeightMap::getHeight(float x, float z, bool worldPosition) {

	if (worldPosition) {
		x += vars->terrainXOffset;
		z += vars->terrainZOffset;

		x /= vars->texelWorldSize;
		z /= vars->texelWorldSize;
	}



	int leftx = (int)x;
	int rightx = leftx + 1;
	int leftz = (int)z;
	int rightz = leftz + 1;

	// clamp to edges
	leftx = glm::clamp(leftx, 0, width - 1);
	rightx = glm::clamp(rightx, 0, width - 1);
	leftz = glm::clamp(leftz, 0, height - 1);
	rightz = glm::clamp(rightz, 0, height - 1);


	float xRatio = x - leftx;
	float zRatio = z - leftz;

	float y1 = data[leftx + leftz * width];
	float y2 = data[leftx + rightz * width];
	float y3 = data[rightx + leftz * width];
	float y4 = data[rightx + rightz * width];

	float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
	float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;

	float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;

	return y;

}

float HeightMap::getWorldWidth() {
	return width * vars->texelWorldSize;
}

float HeightMap::getWorldDepth() {
	return height * vars->texelWorldSize;
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
		shader->setInt("u_TerrainNormalMap", normalMapTextureUnit);
		shader->setInt("u_MaterialMap", materialMapTextureUnit);
		shader->setFloat("u_UVRatio", (float)width / (float)height);
	}



	for (int i = 0; i < MAX_TERRAIN_MATERIALS; i++) {
		materials[i].useMultiple(shader, i);
	}


	//shader->setInt("u_TestDiffuse", 9);
	shader->setInt("u_DepthMapTexture", TEXTURE_UNIT_DEPTH_MAP);

	shader->setFloat("u_GlobalNormalMapMixingRatio", vars->globalNormalMapMixingRatio);

	shader->setBool("u_NormalsOnly", (bool)showNormalsOnly);
	shader->setInt("u_NormalsMode", normalsShaderMode);

	glm::mat4 modelMatrix(1.0f);
	modelMatrix = glm::translate(modelMatrix, -glm::vec3(vars->terrainXOffset, 0.0f, vars->terrainZOffset));
	//modelMatrix = glm::scale(modelMatrix, glm::vec3(vars->texelWorldSize, 1.0f, vars->texelWorldSize));
	shader->setModelMatrix(modelMatrix);


	glBindTextureUnit(normalMapTextureUnit, terrainNormalMap->id);


	// TODO - Make this global for multiple shaders
	shader->setInt("u_CloudShadowTexture", TEXTURE_UNIT_CLOUD_SHADOW_MAP);
	shader->setBool("u_CloudsCastShadows", (bool)vars->cloudsCastShadows);
	shader->setFloat("u_CloudCastShadowAlphaMultiplier", vars->cloudCastShadowAlphaMultiplier);

	if (visualizeTextureMode && visTexture) {
		shader->setBool("u_VisualizeTextureMode", true);
		glBindTextureUnit(materialMapTextureUnit, visTexture->id);
	} else {
		shader->setBool("u_VisualizeTextureMode", false);
		glBindTextureUnit(materialMapTextureUnit, materialMap->id);
	}




	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, numPoints);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

void HeightMap::drawGeometry(ShaderProgram * shader) {
	shader->use();

	shader->setModelMatrix(glm::mat4(1.0f));
	shader->setBool("u_IsInstanced", false);


	glm::mat4 modelMatrix(1.0f);
	modelMatrix = glm::translate(modelMatrix, -glm::vec3(vars->terrainXOffset, 0.0f, vars->terrainZOffset));
	//modelMatrix = glm::scale(modelMatrix, glm::vec3(vars->texelWorldSize, 1.0f, vars->texelWorldSize));

	shader->setModelMatrix(modelMatrix);

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
	float hLeft = data[xLeft + z * width];
	float hRight = data[xRight + z * width];
	float hBottom = data[x + zBottom * width];
	float hTop = data[x + zTop * width];

	//rangeToRange(hLeft, 0.0f, 1.0f, terrainHeightRange.x, terrainHeightRange.y);
	//rangeToRange(hLeft, terrainHeightRange.x, terrainHeightRange.y, 0.0f, 1.0f);
	//rangeToRange(hRight, terrainHeightRange.x, terrainHeightRange.y, 0.0f, 1.0f);
	//rangeToRange(hBottom, terrainHeightRange.x, terrainHeightRange.y, 0.0f, 1.0f);
	//rangeToRange(hTop, terrainHeightRange.x, terrainHeightRange.y, 0.0f, 1.0f);

	//cout << "hLeft = " << hLeft << endl;

	glm::vec3 normal;
	normal.x = (hLeft - hRight) / vars->texelWorldSize;
	//normal.x = (hLeft - hRight);
	//normal.y = hBottom - hTop;
	//normal.z = -2.0f;
	normal.y = 2.0f/* * vars->texelWorldSize * 10.0f*/;
	//rangeToRange(normal.y, 0.0f, 1.0f, vars->terrainHeightRange.x, vars->terrainHeightRange.y);
	normal.z = (hTop - hBottom) / vars->texelWorldSize;
	//normal.z = (hTop - hBottom);


	return glm::normalize(normal);

	/*

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
	*/
}






void HeightMap::initBuffers() {

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

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



			glm::vec3 p1(x, data[x + z  * width], z);
			glm::vec3 p2(x + 1, data[x + 1 + z * width], z);
			glm::vec3 p3(x + 1, data[x + 1 + (z - 1) * width], z - 1);
			glm::vec3 p4(x, data[x + (z - 1) * width], z - 1);

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
