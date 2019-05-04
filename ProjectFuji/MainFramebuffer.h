///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       MainFramebuffer.h
* \author     Martin Cap
*
*	Describes the MainFramebuffer class that is used to manage main framebuffer and its multisampled
*	version. Also provides functions to switch and blit these framebuffers.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <string>
#include "VariableManager.h"
#include "ShaderProgram.h"
#include "Texture.h"

#define MAX_MSAA_SAMPLE_COUNT 12	//!< Maximum possible sample count for MSAA

//! Helper class into which the final image is rendered before it is shown in main window.
/*!
	Uses multisampling to provide better image quality.
*/
class MainFramebuffer {
public:


	GLuint framebufferId;				//!< ID of the main framebuffer
	GLuint multisampledFramebufferId;	//!< ID of the multisampled framebuffer

	GLuint colorTex;					//!< Color attachment to the main framebuffer
	GLuint multisampledColorTex;		//!< Color attachment to the multisampled framebuffer
	GLuint depthTex;					//!< Depth attachment to the main framebuffer
	GLuint multisampledDepthTex;		//!< Depth attachment to the multisampled framebuffer

	//! Loads sampling count from the given VariableManager instance and initializes the MainFramebuffer.
	/*!
		\param[in] vars		VariableManager to be used.
		\see init()
	*/
	MainFramebuffer(VariableManager *vars);
	
	//! Default destructor.
	~MainFramebuffer();

	//! Prepares the framebuffer for next frame.
	/*!
		\param[in] clearColor	Color used to clear the bound framebuffer.
	*/
	void prepareForNextFrame(glm::vec4 clearColor);

	//! Draws the active framebuffer to screen.
	void drawToScreen();
	
	//! Draws a quad.
	void drawQuad();

	//! Blits the multisampled framebuffer to the main (regular) framebuffer if multisampling is turned on.
	void blitMultisampledToRegular();

	//! Obtains the shader and initializes OpenGL buffers and the quad that is used for rendering.
	/*!
		\see initBuffers()
		\see initQuad()
	*/
	void init();

	//! Initializes the framebuffer and its color and depth attachments.
	/*!
		If multisampling enabled, their multisampled counterparts are also initialized.
	*/
	void initBuffers();

	//! Uploads a simple quad to the OpenGL buffers.
	void initQuad();

	//! Refreshes the framebuffers by deleting them and loading them once again.
	void refresh();

	//! Binds the active framebuffer.
	void bind();
	
	//! Unbinds the active framebuffer (by binding the default window framebuffer).
	void unbind();

private:

	GLuint activeFramebuffer;	//!< Currently active framebuffer (useful when we want to switch between regular and multisampled)

	const std::string colorTexName = "Main framebuffer COLOR";	//!< Debug name for the color texture attachment
	const std::string depthTexName = "Main framebuffer DEPTH";	//!< Debug name for the depth texture attachment

	int sampleCount = 12;			//!< Number of MSAA samples if multisampling enabled.
	bool useMultisampling = true;	//!< Whether to use multisampling (MSAA)


	VariableManager *vars = nullptr;	//!< VariableManager for this framebuffer
	ShaderProgram *shader = nullptr;	//!< Shader used for drawing the quad (pass-through shader basically)

	GLuint quadVAO;	//!< VAO of the quad
	GLuint quadVBO;	//!< VBO of the quad

	//! Selects active framebuffer based on whether we are currently using multisampling or not.
	void refreshActiveFramebuffer();

};

