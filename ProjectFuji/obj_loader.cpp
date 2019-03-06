#include "obj_loader.h"

#include <iostream>
#include <fstream>

using namespace std;


// Based on: https://github.com/opengl-tutorials/ogl/blob/master/common/objloader.cpp
bool loadObj(const char *path, std::vector<MeshVertex> &vertices, std::vector<glm::vec3> &outVertices, std::vector<glm::vec2> &outUVs, std::vector<glm::vec3> &outNormals,
			 std::vector<glm::vec3> &outTangents, std::vector<glm::vec3> &outBitangents) {

	std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
	std::vector<glm::vec3> temp_vertices;
	std::vector<glm::vec2> temp_uvs;
	std::vector<glm::vec3> temp_normals;

	FILE *file = fopen(path, "r");
	if (file == nullptr) {
		std::cout << "Impossible to open file!" << std::endl;
		return false;
	}

	while (true) {
		char lineHeader[128];
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF) {
			break;
		}
		if (strcmp(lineHeader, "v") == 0) {
			glm::vec3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			temp_vertices.push_back(vertex);
		} else if (strcmp(lineHeader, "vt") == 0) {
			glm::vec2 uv;
			fscanf(file, "%f %f\n", &uv.x, &uv.y);
			temp_uvs.push_back(uv);
		} else if (strcmp(lineHeader, "vn") == 0) {
			glm::vec3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			temp_normals.push_back(normal);
		} else if (strcmp(lineHeader, "f") == 0) {
			std::string vertex1, vertex2, vertex3;
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
			if (matches != 9) {
				printf("File can't be read by our simple parser : ( Try exporting with other options\n");
				return false;
			}
			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			uvIndices.push_back(uvIndex[0]);
			uvIndices.push_back(uvIndex[1]);
			uvIndices.push_back(uvIndex[2]);
			normalIndices.push_back(normalIndex[0]);
			normalIndices.push_back(normalIndex[1]);
			normalIndices.push_back(normalIndex[2]);
		}

	}
	// For each vertex of each triangle
	for (unsigned int i = 0; i<vertexIndices.size(); i++) {

		// Get the indices of its attributes
		unsigned int vertexIndex = vertexIndices[i];
		unsigned int uvIndex = uvIndices[i];
		unsigned int normalIndex = normalIndices[i];

		// Get the attributes thanks to the index
		glm::vec3 vertex = temp_vertices[vertexIndex - 1];
		glm::vec2 uv = temp_uvs[uvIndex - 1];
		glm::vec3 normal = temp_normals[normalIndex - 1];

		// Put the attributes in buffers
		outVertices.push_back(vertex);
		outUVs.push_back(uv);
		outNormals.push_back(normal);

	}

	for (unsigned int i = 0; i < outVertices.size(); i++) {
		MeshVertex vert;
		vert.position = outVertices[i];
		vert.normal = outNormals[i];
		vert.texCoords = outUVs[i];
		vertices.push_back(vert);
	}
	for (unsigned int i = 0; i < outVertices.size() - 2; i += 3) {
		glm::vec3 edge1 = outVertices[i + 1] - outVertices[i];
		glm::vec3 edge2 = outVertices[i + 2] - outVertices[i];
		glm::vec2 deltaUV1 = outUVs[i + 1] - outUVs[i];
		glm::vec2 deltaUV2 = outUVs[i + 2] - outUVs[i];

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

		for (unsigned int j = 0; j < 3; j++) {
			outTangents.push_back(tangent);
			outBitangents.push_back(bitangent);
		}
	}

	fclose(file);
	return true;

}


// Taken from: https://github.com/opengl-tutorials/ogl/blob/master/common/objloader.cpp
bool loadObj(const char *path, std::vector<MeshVertex> &vertices, std::vector<glm::vec3> &outVertices, std::vector<glm::vec2> &outUVs, std::vector<glm::vec3> &outNormals) {

	std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
	std::vector<glm::vec3> temp_vertices;
	std::vector<glm::vec2> temp_uvs;
	std::vector<glm::vec3> temp_normals;

	FILE *file = fopen(path, "r");
	if (file == nullptr) {
		std::cout << "Impossible to open file!" << std::endl;
		return false;
	}

	while (true) {
		char lineHeader[128];
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF) {
			break;
		}
		if (strcmp(lineHeader, "v") == 0) {
			glm::vec3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			temp_vertices.push_back(vertex);
		} else if (strcmp(lineHeader, "vt") == 0) {
			glm::vec2 uv;
			fscanf(file, "%f %f\n", &uv.x, &uv.y);
			temp_uvs.push_back(uv);
		} else if (strcmp(lineHeader, "vn") == 0) {
			glm::vec3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			temp_normals.push_back(normal);
		} else if (strcmp(lineHeader, "f") == 0) {
			std::string vertex1, vertex2, vertex3;
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
			if (matches != 9) {
				printf("File can't be read by our simple parser : ( Try exporting with other options\n");
				return false;
			}
			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			uvIndices.push_back(uvIndex[0]);
			uvIndices.push_back(uvIndex[1]);
			uvIndices.push_back(uvIndex[2]);
			normalIndices.push_back(normalIndex[0]);
			normalIndices.push_back(normalIndex[1]);
			normalIndices.push_back(normalIndex[2]);
		}

	}
	// For each vertex of each triangle
	for (unsigned int i = 0; i<vertexIndices.size(); i++) {

		// Get the indices of its attributes
		unsigned int vertexIndex = vertexIndices[i];
		unsigned int uvIndex = uvIndices[i];
		unsigned int normalIndex = normalIndices[i];

		// Get the attributes thanks to the index
		glm::vec3 vertex = temp_vertices[vertexIndex - 1];
		glm::vec2 uv = temp_uvs[uvIndex - 1];
		glm::vec3 normal = temp_normals[normalIndex - 1];

		// Put the attributes in buffers
		outVertices.push_back(vertex);
		outUVs.push_back(uv);
		outNormals.push_back(normal);

	}

	for (unsigned int i = 0; i < outVertices.size(); i++) {
		MeshVertex vert;
		vert.position = outVertices[i];
		vert.normal = outNormals[i];
		vert.texCoords = outUVs[i];
		vertices.push_back(vert);
	}
	fclose(file);
	return true;

}