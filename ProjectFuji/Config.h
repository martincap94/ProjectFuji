///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Config.h
* \author     Martin Cap
* \brief      Configuration file.
*
*  Configuration header file that contains some of the variables that are not modifiable at runtime.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <cuda_runtime.h>



#define TEXTURES_DIR "textures/"				//!< Textures directory
#define SCENES_DIR "scenes/"					//!< Scenes directory
#define SHADERS_DIR "shaders/"					//!< Shaders directory
#define LOG_FILENAME_BASE "logs/"				//!< Logs directory
#define SOUNDING_DATA_DIR "sounding_data/"		//!< Directory in which sounding data files reside
#define PARTICLE_DATA_DIR "particle_system/"	//!< Directory in which particle data files reside

#define MIN_X -40.0f							//!< Minimum x value for the diagram
#define MAX_X 40.0f								//!< Maximum x value for the diagram

#define MIN_TEMP MIN_X							//!< Minimum temperature shown in the diagram
#define MAX_TEMP MAX_X							//!< Maximum temperature shown in the diagram

#define MIN_P 100.0f							//!< Minimum pressure shown in the diagram, should be constant for all soundings
#define MAX_P 1025.0f							//!< Maximum pressure shown in the diagram, should be constant for all soundings

#define CURVE_DELTA_P 25.0f						//!< Distance of pressure [hPa] between curve vertices for adiabats


#define TEXTURE_UNIT_DEPTH_MAP 14				//!< Texture unit to be used for depth map binding
#define TEXTURE_UNIT_CLOUD_SHADOW_MAP 15		//!< Texture unit to be used for cloud shadow map binding

#define MAX_TERRAIN_MATERIALS 4					//!< Maximum amount of Blinn-Phong materials used on terrain
#define MAX_TERRAIN_PBR_MATERIALS (MAX_TERRAIN_MATERIALS - 1)		//!< Maximum amount of PBR materials used on terrain

// Old defines from LBM 2D implementation, may be useful in future updates when we incorporate old functions back to the framework
//#define DRAW_VELOCITY_ARROWS
//#define DRAW_PARTICLE_VELOCITY_ARROWS

#define CONFIG_FILE "config.ini"		//!< Name of the configuration file that loads initialization data


#define MAX_STREAMLINE_LENGTH 200		//!< Maximum streamline length for the old framework! Not used anymore


#define LAT_SPEED 1.0f					//!< Lattice speed (experimental) from the old framework - parametrization of the LBM
#define LAT_SPEED_SQ (LAT_SPEED * LAT_SPEED)	//!< Square root of lattice speed (experimental) from the old framework


#define DEFAULT_CAMERA_SPEED 60.0f		//!< Default camera movement speed


#define SMAG_C 0.3f						//!< Smagorinsky constant (experimental) - used in LBM subgrid model


//using namespace std;
