#pragma once


#include <vector>
#include <glm/glm.hpp>
#include "DataStructures.h"

//bool loadObjNew(const char *path, std::vector<MeshVertex> &outVertices);

//bool loadObjNew(const char * path, std::vector<glm::vec3>& outVertices, std::vector<glm::vec2>& outUVs, std::vector<glm::vec3>& outNormals);


bool loadObj(const char *path, std::vector<MeshVertex>& vertices, std::vector<glm::vec3> &outVertices,
			 std::vector<glm::vec2> &outUVs, std::vector<glm::vec3> &outNormals, std::vector<glm::vec3> &outTangents, std::vector<glm::vec3> &outBitangents);

bool loadObj(const char *path, std::vector<MeshVertex>& vertices, std::vector<glm::vec3> &outVertices,
			 std::vector<glm::vec2> &outUVs, std::vector<glm::vec3> &outNormals);