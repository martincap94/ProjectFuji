///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       CUDAUtils.h
* \author     Martin Cap
* \brief      Utility header file that contains helper functions for CUDA usage.
*
*	Utility header file that contains helper functions for CUDA usage.
*	At this moment it contains the error handling only.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/color_space.hpp>

#include <iostream>

//! Handle and print CUDA errors. Adapted from slides.
/*!
	Prints and handles given CUDA errors. This function and its helper macro were adapted from slides
	of GPGPU course: https://cent.felk.cvut.cz/courses/GPU/ by Ing. Jaroslav Sloup.

	\param[in] error	The error to handle.
	\param[in] file		The source file the error may have occured in.
	\param[in] line		The line where the error was processed.
*/
static void handleError(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		exit(EXIT_FAILURE);
	}
}

/// Helper macro function that checks and handles error for the cudaError_t parameter.
#define CHECK_ERROR( error ) ( handleError( error, __FILE__, __LINE__ ) )




