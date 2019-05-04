///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       ShaderProgram.h
* \author     Martin Cap
*
*	Simple ShaderProgram class that helps us with creating, linking and using shader programs.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////


#pragma once

#include <glad\glad.h>

#include <string>
#include "Config.h"

#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>

#include "DirectionalLight.h"


//! Simple shader program representation which supports basic operations.
/*!
	Simple shader program representation which supports basic creation (only vertex and fragment shaders at this moment).
	It holds the identification number of its shader program and provides basic uniform setting functions.
*/
class ShaderProgram {
public:

	//! Possible material types of the shader.
	enum eMaterialType {
		NONE,		//!< No material type set
		COLOR,		//!< The object does not use any shading, just color (it is unlit)
		PHONG,		//!< Use Blinn-Phong shader
		PBR			//!< Use simple PBR shader
	};

	//! Possible ways the material can be lit/unlit.
	enum eLightingType {
		LIT,	//!< The material can be lit
		UNLIT	//!< The material cannot be lit
	};

	GLuint id;	//!< OpenGL ID of the program

	eMaterialType matType = NONE;			//!< Material type
	eLightingType lightingType = UNLIT;		//!< Lighting type

	//! Default constructor.
	ShaderProgram();

	//! Creates and links vertex and fragment shaders.
	/*!
		MeshVertex and fragment shaders are read from files.
		\param[in] vsPath	Path to the vertex shader.
		\param[in] fsPath	Path to the fragment shader.
		\param[in] gsPath	Path to the geometry shader, can be nullptr.
	*/
	ShaderProgram(const GLchar *vsPath, const GLchar *fsPath, const GLchar *gsPath = nullptr);

	//! Default destructor.
	~ShaderProgram();

	//! Use the shader program.
	void use();

	//! Set the boolean uniform variable.
	void setBool(const std::string &name, bool value) const;

	//! Set the integer uniform variable.
	void setInt(const std::string &name, int value) const;

	//! Set the float uniform variable.
	void setFloat(const std::string &name, float value) const;

	//! Set 4x4 float matrix uniform variable.
	void setMat4fv(const std::string &name, glm::mat4 value) const;

	//! Set vec3 uniform with 3 float values.
	void setVec3(const std::string &name, float x, float y, float z) const;

	//! Set vec3 uniform with glm::vec3.
	void setVec3(const std::string &name, glm::vec3 value) const;

	//! Set vec4 unfirom with glm::vec4.
	void setVec4(const std::string &name, glm::vec4 value) const;

	//! Set the projection matrix uniform.
	void setProjectionMatrix(glm::mat4 projectionMatrix, std::string uniformName = "u_Projection");

	//! Set the view matrix uniform.
	void setViewMatrix(glm::mat4 viewMatrix, std::string uniformName = "u_View");

	//! Set the model matrix uniform.
	void setModelMatrix(glm::mat4 modelMatrix, std::string uniformName = "u_Model");

	//! Set the RGB color uniform.
	void setColor(glm::vec3 color, std::string uniformName = "u_Color");

	//! Set the RGBA color uniform.
	void setColorAlpha(glm::vec4 color, std::string uniformName = "u_Color");

	//! Prepares the material uniforms for this material's type.
	/*!
		\param[in] useShader	Use the shader before setting the uniforms (false to prevent unnecessary state changes).
	*/
	void setupMaterialUniforms(bool useShader = true);

	//! Sets the given fog properties.
	/*!
		\param[in] fogIntensity		Intensity [0,1] of the fog.
		\param[in] fogMinDistance	Minimum distance at which the fog appears.
		\param[in] fogMaxDistance	Maximum distance till which the fog hasn't got full opacity.
		\param[in] fogColor			Color of the fog.
		\param[in] fogMode			Whether the fog is linear or exponential.
		\param[in] fogExpFalloff	Exponential falloff of the fog (if exponential fog used).
		\param[in] useShader		Whether to use the shader program before setting these uniforms.
	*/
	void setFogProperties(float fogIntensity, float fogMinDistance, float fogMaxDistance, glm::vec4 fogColor, int fogMode = 0, float fogExpFalloff = 0.1f, bool useShader = true);


	//void setPointLightAttributes(int lightNum, PointLight &pointLight);

	//! Updates uniforms with the given directional light.
	/*!
		\param[in] dirLight		The directional light to be used.
	*/
	void updateDirectionalLightUniforms(DirectionalLight &dirLight);

	//void setSpotlightAttributes(Spotlight &spotlight, Camera &camera, bool spotlightOn);

};

