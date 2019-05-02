///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       DataStructures.h
* \author     Martin Cap
*
*	Header file containing important data structures for the application.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <glm\glm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>


//! Enum of all possible sounding data parameters.
enum eSTLPAttribute {
	PRES,	//!< Pressure;				in [hPa]
	HGHT,	//!< Height;				in [m]
	TEMP,	//!< Temperature;			in [°C]
	DWPT,	//!< Dew-point temperature;	in [°C]
	RELH,	//!< Relative humidity;		in [%]
	MIXR,	//!< Mixing ratio;			in [g/kg]
	DRCT,	//!< Wind direction;		in [°]
	SKNT,	//!< Wind speed;			in [knots]
	TWTB,	//!< [°C]
	TVRT,	//!< [°C]
	THTA,	//!< [K]
	THTE,	//!< [K]
	THTV	//!< [K]
};

//! Single sounding data row item.
struct SoundingDataItem {
	float data[13];

	void print() {
		printf("Item: \n");
		printf("	PRES = %0.2f\n", data[PRES]);
		printf("	HGHT = %0.2f\n", data[HGHT]);
		printf("	TEMP = %0.2f\n", data[TEMP]);
		printf("	DWPT = %0.2f\n", data[DWPT]);
		printf("	RELH = %0.2f\n", data[RELH]);
		printf("	MIXR = %0.2f\n", data[MIXR]);
		printf("	DRCT = %0.2f\n", data[DRCT]);
		printf("	SKNT = %0.2f\n", data[SKNT]);
		printf("	TWTB = %0.2f\n", data[TWTB]);
		printf("	TVRT = %0.2f\n", data[TVRT]);
		printf("	THTA = %0.2f\n", data[THTA]);
		printf("	THTE = %0.2f\n", data[THTE]);
		printf("	THTV = %0.2f\n", data[THTV]);
	}
};


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


struct MeshVertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texCoords;
	glm::vec3 tangent;
	glm::vec3 bitangent;
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


const float quadVertices[] = {
	// positions   // texCoords
	-1.0f,  1.0f,  0.0f, 1.0f,
	-1.0f, -1.0f,  0.0f, 0.0f,
	1.0f, -1.0f,  1.0f, 0.0f,

	-1.0f,  1.0f,  0.0f, 1.0f,
	1.0f, -1.0f,  1.0f, 0.0f,
	1.0f,  1.0f,  1.0f, 1.0f
};


const float cubeVertices[] = {
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
	0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
	-0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

	-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	-0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

	0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
	0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
	0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
	-0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

	-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
	0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
	-0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
	-0.5f,  0.5f, -0.5f,  0.0f, 1.0f
};



const float skyboxVertices[108] = {
	// positions
	-1.0f,  1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f,  1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,

	-1.0f, -1.0f,  1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,
	-1.0f,  1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f,

	1.0f, -1.0f, -1.0f,
	1.0f, -1.0f,  1.0f,
	1.0f,  1.0f,  1.0f,
	1.0f,  1.0f,  1.0f,
	1.0f,  1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,

	-1.0f, -1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	1.0f,  1.0f,  1.0f,
	1.0f,  1.0f,  1.0f,
	1.0f, -1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f,

	-1.0f,  1.0f, -1.0f,
	1.0f,  1.0f, -1.0f,
	1.0f,  1.0f,  1.0f,
	1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f, -1.0f,

	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f,  1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f,  1.0f,
	1.0f, -1.0f,  1.0f
};


// Indexed skybox taken from: https://github.com/benanders/Hosek-Wilkie/blob/master/src/main.rs
const float skyboxVerticesNew[24] = {
		-1.0, -1.0, 1.0, // 0: Left,  bottom, front
		1.0, -1.0, 1.0, // 1: Right, bottom, front
		1.0, 1.0, 1.0, // 2: Right, top,    front
		-1.0, 1.0, 1.0, // 3: Left,  top,    front
		-1.0, -1.0, -1.0, // 4: Left,  bottom, back
		1.0, -1.0, -1.0, // 5: Right, bottom, back
		1.0, 1.0, -1.0, // 6: Right, top,    back
		-1.0, 1.0, -1.0, // 7: Left,  top,    back
};

const unsigned int skyboxIndicesNew[36] = {
		2, 1, 0, 0, 3, 2, // Front
		5, 6, 7, 7, 4, 5, // Back
		3, 0, 4, 4, 7, 3, // Left
		6, 5, 1, 1, 2, 6, // Top
		6, 2, 3, 3, 7, 6, // Right
		0, 1, 5, 5, 4, 0, // Bottom
};