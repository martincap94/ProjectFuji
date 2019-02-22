///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       DataStructures.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Class containing important data structures for the application.
*
*  Class containing important data structures for the application.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <glm\glm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// Node for LBM 3D
struct Node3D {
	float adj[19];	///< Distribution function for adjacent lattice nodes (in possible streaming directions)
};

/// 3rd ordering enum as proposed by Woodgate et al.
enum eDirection3D {
	DIR_MIDDLE_VERTEX = 0,
	DIR_RIGHT_FACE,
	DIR_LEFT_FACE,
	DIR_BACK_FACE,
	DIR_FRONT_FACE,
	DIR_TOP_FACE,
	DIR_BOTTOM_FACE,
	DIR_BACK_RIGHT_EDGE,
	DIR_BACK_LEFT_EDGE,
	DIR_FRONT_RIGHT_EDGE,
	DIR_FRONT_LEFT_EDGE,
	DIR_TOP_BACK_EDGE,
	DIR_TOP_FRONT_EDGE,
	DIR_BOTTOM_BACK_EDGE,
	DIR_BOTTOM_FRONT_EDGE,
	DIR_TOP_RIGHT_EDGE,
	DIR_TOP_LEFT_EDGE,
	DIR_BOTTOM_RIGHT_EDGE,
	DIR_BOTTOM_LEFT_EDGE
};

/// Direction vectors for LBM 3D
const glm::vec3 directionVectors3D[19] = {
	glm::vec3(0.0f, 0.0f, 0.0f),
	glm::vec3(1.0f, 0.0f, 0.0f),
	glm::vec3(-1.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, 0.0f, -1.0f),
	glm::vec3(0.0f, 0.0f, 1.0f),
	glm::vec3(0.0f, 1.0f, 0.0f),
	glm::vec3(0.0f, -1.0f, 0.0f),
	glm::vec3(1.0f, 0.0f, -1.0f),
	glm::vec3(-1.0f, 0.0f, -1.0f),
	glm::vec3(1.0f, 0.0f, 1.0f),
	glm::vec3(-1.0f, 0.0f, 1.0f),
	glm::vec3(0.0f, 1.0f, -1.0f),
	glm::vec3(0.0f, 1.0f, 1.0f),
	glm::vec3(0.0f, -1.0f, -1.0f),
	glm::vec3(0.0f, -1.0f, 1.0f),
	glm::vec3(1.0f, 1.0f, 0.0f),
	glm::vec3(-1.0f, 1.0f, 0.0f),
	glm::vec3(1.0f, -1.0f, 0.0f),
	glm::vec3(-1.0f, -1.0f, 0.0f)
};

