///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       EVSMShadowMapper.h
* \author     Martin Cap
*
*	Header file containing helper functions for loading cubemap textures for skybox creation.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include <vector>

class ShaderProgram;
class DirectionalLight;
class VariableManager;

//! Helper class for drawing shadow maps.
/*!
	Uses the exponential variance shadow maps (EVSM) based on variance shadow maps (VSM) and layered variance
	shadow maps (LVSM). 
	All these methods were proposed by Andrew Lauritzen;
	VSM is available here: http://www.punkuser.net/vsm/
	LVSM here: http://www.punkuser.net/lvsm/lvsm_web.pdf

*/
class EVSMShadowMapper {
public:

	bool useBlurPass = true;			//!< Whether to use blur pass on the depth map
	const GLint resolution = 1024;		//!< Framebuffer resolution for the depth map

	float shadowBias = 0.001f;			//!< Shadow bias used in shaders
	float lightBleedReduction = 0.2f;	//!< Naive light bleeding reduction parameter
	float varianceMinLimit = 0.0001f;	//!< Minimum variance limit
	float exponent = 42.0f;				//!< Exponent used in EVSM shaders
	float shadowIntensity = 0.95f;		//!< Intensity of the shadows (i.e. how dark the shadows can get)

	glm::mat4 lightProjectionMatrix;	//!< Projection matrix of the light
	glm::mat4 lightViewMatrix;			//!< View matrix of the light

	glm::mat4 lightSpaceMatrix;			//!< Light projection matrix * light view matrix
	glm::mat4 prevLightSpaceMatrix;		//!< Light space matrix from previous frame to prevent cloud cast shadow snapping

	int shadowOnly = 0;					//!< Whether to display only shadows

	//! Constructs and initializes the EVSMShadowMapper instance.
	/*!
		\param[in] vars			VariableManager to be used.
		\param[in] dirLight		The DirectionalLight that lights the scene and casts shadows.
		\see init()
	*/
	EVSMShadowMapper(VariableManager *vars, DirectionalLight *dirLight);

	//! Default destructor.
	~EVSMShadowMapper();

	//! Initializes all necessary OpenGL buffers, textures and prepares all shaders.
	/*!
		The shaders are managed here for the whole application.
	*/
	void init();

	//! Prepares the scene for the first pass of the shadow mapping process.
	/*!
		This includes binding the depthMapFramebuffer, clearing it and setting light projection matrices
		to all appropriate shaders.
	*/
	void preFirstPass();

	//! Runs all post-first pass processes, mainly the blurring pass if it is enabled.
	void postFirstPass();

	//! Prepares the scene for second pass of the shadow mapping process.
	/*!
		Binds the depthMapTexture (or its blurred version) to the correct texture unit and informs 
		shaders about its sampler index.
	*/
	void preSecondPass();

	//! Runs all post-second pass processes, mainly enabling disabled OpenGL flags such as back face culling.
	void postSecondPass();

	//! Returns the currently active depth map texture ID.
	/*!
		\return		OpenGL ID of the depth map texture (or its blurred version if blur pass enabled).
	*/
	GLuint getDepthMapTextureId();

	//! Returns the zBufferTexture ID.
	/*!
		\return		zBufferTexture OpenGL ID.
	*/
	GLuint getZBufferTextureId();


// let it be public for testing now
//private:

	//! Returns whether the EVSMShadowMapper is ready for scene rendering.
	/*!
		\return	Whether the EVSMShadowMapper is ready for scene rendering.
	*/
	bool isReady();

	std::vector<ShaderProgram *> firstPassShaders;		//!< List of shaders that can be used in the first pass
	std::vector<ShaderProgram *> secondPassShaders;		//!< List of shaders that can be used in the second pass

	ShaderProgram *blurShader;			//!< Shader that is used for blurring the depth map

	// initial depth map generation
	GLuint depthMapTexture;				//!< Depth map texture into which we draw color in the first pass
	GLuint zBufferTexture;				//!< z buffer texture into which we draw depths in the first pass
	GLuint depthMapFramebuffer;			//!< Framebuffer for the first pass of the shadow mapping algorithm

	GLuint firstPassBlurTexture;		//!< Blurred depth map texture after the (first) horizontal blur pass
	GLuint firstPassBlurFramebuffer;	//!< Framebuffer for the (first) horizontal blur pass

	GLuint secondPassBlurTexture;		//!< Blurred depth map texture after the (second) vertical blur pass
	GLuint secondPassBlurFramebuffer;	//!< Framebuffer for the (second) vertical blur pass



	GLuint quadVAO;		//!< VAO for the quad that is used for the blur pass
	GLuint quadVBO;		//!< VBO for the quad that is used for the blur pass

private:

	VariableManager *vars = nullptr;		//!< VariableManager for this instance.
	DirectionalLight *dirLight = nullptr;	//!< DirectionalLight that casts the shadows generated by this shadow mapper



};

