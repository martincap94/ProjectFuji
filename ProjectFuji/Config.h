///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Config.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Configuration file.
*
*  Configuration header file that contains some of the variables that or not modifiable at runtime 
*  (but at compile time).
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <cuda_runtime.h>



#define TEXTURES_DIR "textures/"		///< Textures directory
#define SCENES_DIR "scenes/"			///< Scenes directory
#define SHADERS_DIR "shaders/"			///< Shaders directory
#define LOG_FILENAME_BASE "logs/"		///< Logs directory

//#define LBM_EXPERIMENTAL // experimental features


//#define DRAW_VELOCITY_ARROWS
//#define DRAW_PARTICLE_VELOCITY_ARROWS

#define CONFIG_FILE "config.ini"		///< Configuration filename


#define MAX_STREAMLINE_LENGTH 200		///< Maximum streamline length


#define LAT_SPEED 1.0f					///< Lattice speed (experimental)
#define LAT_SPEED_SQ (LAT_SPEED * LAT_SPEED)	///< Square root of lattice speed (experimental)


#define DEFAULT_CAMERA_SPEED 60.0f		///< Default camera movement speed


//#define SUBGRID_EXPERIMENTAL
#define SMAG_C 0.3f						///< Smagorinsky constant (experimental)


using namespace std;
