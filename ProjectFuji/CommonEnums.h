///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CommonEnums.h
* \author     Martin Cap
*
*	This file contains some common enums that are used in the framework.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

//! Projection mode of the application.
enum eProjectionMode {
	ORTHOGRAPHIC,	//!< Orthographic projection (parallel)
	PERSPECTIVE		//!< Perspective projection
};

//! --- DEPRECATED --- \deprecated Enum listing all possible LBM types.
enum eLBMType {
	LBM2D,	//!< 2D version of the LBM
	LBM3D	//!< 3D version of the LBM
};

//! Possible sorting policies when using GPU sorting with Thrust.
enum eSortPolicy {
	LESS,		//!< Sort by less than operator (<)
	GREATER,	//!< Sort by greater than operator (>)
	LEQUAL,		//!< Sort by less or equal operator (<=)
	GEQUAL		//!< Sort by greater or equal operator (>=)
};


//! Possible viewport modes of the application.
enum eViewportMode {
	VIEWPORT_3D = 0,	//!< Regular 3D viewport
	DIAGRAM				//!< Viewport showing the STLP diagram
};

//! Possible fog modes.
enum eFogMode {
	LINEAR = 0,			//!< Fog with linear falloff
	EXPONENTIAL,		//!< Fog with exponential falloff
	_NUM_FOG_MODES		//!< Number of fog modes
};