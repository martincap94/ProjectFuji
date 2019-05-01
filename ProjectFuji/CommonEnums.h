///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CommonEnums.h
* \author     Martin Cap
* \brief      Contains common enums used in the framework.
*
*	This file contains some common enums that are used in the framework.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once


enum eProjectionMode {
	ORTHOGRAPHIC,
	PERSPECTIVE
};

/// Enum listing all possible LBM types. LBM2D_reindex and LBM3D_reindexed were deprecated, hence they are absent.
enum eLBMType {
	LBM2D,
	LBM3D
};

enum eSortPolicy {
	LESS,
	GREATER,
	LEQUAL,
	GEQUAL
};


enum eViewportMode {
	VIEWPORT_3D = 0,
	DIAGRAM
};

enum eFogMode {
	LINEAR = 0,
	EXPONENTIAL,
	_NUM_FOG_MODES
};