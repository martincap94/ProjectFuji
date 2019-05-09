// Particle Renderer based on the sample CUDA project - add proper citations!
///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       ParticleRenderer.h
* \author     Martin Cap
*
*	ParticleRenderer renders particles of the set ParticleSystem using the volumetric half-vector
*	slicing method described in: https://developer.download.nvidia.com/assets/cuda/files/smokeParticles.pdf
*	Based on the CUDA sample provided with CUDA toolkit installation.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>


#include "ShaderProgram.h"
#include "Camera.h"
#include "DirectionalLight.h"
#include "VariableManager.h"
#include "ParticleSystem.h"
#include "Timer.h"

//! Class that is used for volumetric particle rendering.
/*!
	The rendering algorithm is based on CUDA sample implementation of half-vector slicing.
	Article by Simon Green describing the method can be found here: https://developer.download.nvidia.com/assets/cuda/files/smokeParticles.pdf
	Extends the method with phase functions that is crucial in visualizing clouds.
*/
class ParticleRenderer {
public:

	//! Possible phase functions that can be used in the renderer.
	enum ePhaseFunction {
		NONE = 0,					//!< No phase function is used
		RAYLEIGH,					//!< Rayleigh phase function (for tiny particles - not useful for clouds)
		HENYEY_GREENSTEIN,			//!< Henyey-Greenstein (one lobe - either forward or backward based on symmetry parameter g)
		DOUBLE_HENYEY_GREENSTEIN,	//!< Double Henyey-Greenstein (two interpolated lobes with two symmetry parameters)
		SCHLICK,					//!< Fast approximation of Henyey-Greenstein
		CORNETTE_SHANK,				//!< More physically accurate Henyey-Greenstein (and more computationally expensive)
		_NUM_PHASE_FUNCTIONS		//!< Number of phase functions
	};


	glm::vec3 lightVec;			//!< Direction of the light
	glm::vec3 viewVec;			//!< View vector (forward direction of camera)
	glm::vec3 halfVec;			//!< Half vector (depends on angle between light vector and view vector)

	glm::mat4 lightViewMatrix;			//!< View matrix of the light
	glm::mat4 lightProjectionMatrix;	//!< Projection matrix of the light

	bool invertedRendering = false;		//!< Whether we render front to back (true) or back to front

	const int lightBufferResolution = 1024;		//!< Resolution of the light buffer

	int maxNumSlices = 4096;				//!< Maximum number of slices allowed
	int numSlices = 256;					//!< Number of slices used
	int numDisplayedSlices = numSlices;		//!< Number of visible slices

	int srcLightTexture = 0;	//!< Current source light texture (used for blurring)

	int shaderSet = 2;		//!< Used shader set (for debugging/testing)

	int batchSize;			//!< Size of one particle batch (determined by number of particles / number of slices)

	int imageWidth;			//!< Width of the final image
	int imageHeight;		//!< Height of the final image

	int useVolumetricRendering = 1;		//!< Whether to use volumetric rendering or the old system

	int useBlurPass = 1;				//!< Whether to use the blur pass of the light texture
	float blurAmount = 0.4f;			//!< How much should the light texture be blurred if blur pass enabled

	// for easier testing
	int firstPassShaderMode = 0;		//!< Debug mode uniform variable in the first pass
	int numFirstPassShaderModes = 2;	//!< Number of debug modes in the first pass

	int secondPassShaderMode = 0;		//!< Debug mode uniform variable in the second pass
	int numSecondPassShaderModes = 2;	//!< Number of debug modes in the second pass

	float shadowAlpha100x = 0.5f;					//!< Multiplier of the shadow alpha value (x100)
	float shadowAlpha = shadowAlpha100x * 0.01f;	//!< Actual multiplier of the shadow alpha value

	int showParticleTextureIdx = 0;		//!< Show particle indices if texture atlas is used
	int useAtlasTexture = 0;			//!< Whether to use atlas texture
	int showParticlesBelowCCL = 1;		//!< Whether to show particles below CCL/LCL (show hidden particles basically)

	float inversionThreshold = 0.0f;	//!< Threshold of the rendering inversion (compared to the dot product of view vector and light vector)

	ePhaseFunction phaseFunction = ePhaseFunction::HENYEY_GREENSTEIN;	//!< Used phase function
	int multiplyPhaseByShadow = 1;		//!< Whether to multiply phase function intensity by shadow intensity
	float symmetryParameter = 0.58f;	//!< Symmetry parameter g for Henyey-Greenstein function 
	float symmetryParameter2 = -0.6f;	//!< Secondary symmetry parameter for Double Henyey-Greenstein
	float dHenyeyGreensteinInterpolationParameter = 0.5f;	//!< Interpolation parameter for Double Henyey-Greenstein

	Texture *spriteTexture;				//!< Sprite texture used for particle rendering
	Texture *atlasSpriteTexture;		//!< Atlas sprite texture used when atlas drawing enabled

	std::vector<Texture *> spriteTextures;		//!< List of all possible sprite textures

	Timer *timer = nullptr;

	//! Initializes the renderer and prepares all available textures.
	/*!
		\param[in] vars		VariableManager to be used.
		\param[in] ps		ParticleSystem that holds the particles we want to render.
	*/
	ParticleRenderer(VariableManager *vars, ParticleSystem *ps);

	//! Default destructor.
	~ParticleRenderer();

	//! Sets border color for the light textures.
	/*!
		\param[in] dirLight		DirectionalLight whose color is used for setting the color.
	*/
	void setLightTextureBorders(DirectionalLight *dirLight);

	//! Sets all the shader uniforms for the given shader.
	/*!
		\param[in] shader	Shader whose uniforms are to be set.
	*/
	void setShaderUniforms(ShaderProgram *shader);

	//! Draws the particles of the given particle system.
	/*!
		Draws particles in slices and then composites the result to the main framebuffer.
		
		\param[in] ps			ParticleSystem whose particles are to be rendered.
		\param[in] dirLight		DirectionalLight that lights the particles.
		\param[in] cam			Camera used for rendering.
		\see drawSlices()
		\see compositeResult()
	*/
	void draw(ParticleSystem *ps, DirectionalLight *dirLight, Camera *cam);

	//! Recalculates all important vectors that are used in drawing the slices.
	/*!
		Most importantly it determines whether we use inverted rendering or not.
		This depends on the angle between light's direction vector and camera view vector.
		If the angle is an acute angle, inverted rendering is used and half vector is computed as
		normalized view vector + light vector. Otherwise inverted rendering is not used and 
		half vector is computed as -view vector + light vector.
		
		\param[in] cam			Camera used for rendering.
		\param[in] dirLight		DirectionalLight that lights the particles.
	*/
	void recalcVectors(Camera *cam, DirectionalLight *dirLight);

	//! Returns the vector that should be used for projection sorting of the particles.
	/*!
		\return		Vector that determines sorting of the particles.
	*/
	glm::vec3 getSortVec();

	//! Refreshes image buffer.
	/*!
		This should be called when we change screen size.
	*/
	void refreshImageBuffer();

	//! Does pre-scene render preparations.
	void preSceneRenderImage();

	//! Does post-scene render operations.
	void postSceneRenderImage();

	//! Updates shader set that is currently selected.
	void updateShaderSet();

	//! Returns name of the given phase function.
	/*!
		\return Name of the given phase function.
	*/
	const char *getPhaseFunctionName(int phaseFunc);



private:

	// Helper members so we do not have to send them through all the functions
	ParticleSystem *ps = nullptr;			//!< ParticleSystem whose particles are rendered
	DirectionalLight *dirLight = nullptr;	//!< DirectionalLight that lights the particles
	Camera *cam = nullptr;					//!< Camera for which the particles are rendered

	VariableManager *vars = nullptr;		//!< VariableManager for this instance

	ShaderProgram *firstPassShader;		//!< Shader used in the first pass (light view)
	ShaderProgram *secondPassShader;	//!< Shader used in the second pass (camera view)
	ShaderProgram *passThruShader;		//!< Pass-through shader used for compositing the final result
	ShaderProgram *blurShader;			//!< Shader that is used when blurring the light texture

	GLuint imageFramebuffer;	//!< Framebuffer for the final image generated (the image that will be composited to main framebuffer)
	GLuint imageTexture;		//!< Color attachment for the image framebuffer
	GLuint imageDepthTexture;	//!< Depth attachment for the image framebuffer -

	GLuint lightFramebuffer;	//!< Framebuffer for the first pass (light view)
	GLuint lightTexture[2];		//!< Two light textures (second is used when using a blur pass)
	GLuint lightDepthTexture;	//!< Depth texture attachment to the light framebuffer

	int prevShaderSet = -1; //!< Previous shader set used

	GLuint quadVAO;		//!< VAO of the quad that is used for compositing
	GLuint quadVBO;		//!< VBO of the quad that is used for compositing

	//! Initialize framebuffers for first and second pass of the algorithm.
	void initFramebuffers();

	//! Initializes the image framebuffer.
	void initImageBuffer();

	//! Initializes the two light buffers.
	void initLightBuffers();

	//! Draws all visible slices.
	/*!
		Determines batch size and prepares framebuffers for both passes.
		Iteratively draws slice from the second pass and then from the first pass.
		This may be a little confusing since their order is reversed, I use terminology as in shadow mapping
		where the first pass is drawn from light's point of view and second pass is drawn from camera's point of view.
		If blur pass is enabled, we blur the light texture after drawing each batch.
		\see drawSlice()
		\see drawSliceLightView()
		\see blurLightTexture()
	*/
	void drawSlices();

	//! Draws a single slice (size of one batch) from the given offset to the image framebuffer.
	/*!
		Must use different blending function if inverted rendering is enabled.
		\param[in] i		Offset of the slice.
		\see drawPointSprites()
	*/
	void drawSlice(int i);

	//! Draws a single slice to the active light framebuffer.
	/*!
		\param[in] i		Offset of the slice.
		\see drawPointSprites()
	*/
	void drawSliceLightView(int i);

	//! Draws point sprites from the given start.
	/*!
		Does not change depths! Only reads them.
		\param[in] shader		Shader to be used for rendering.
		\param[in] start		Start index of the points to be drawn.
		\param[in] count		Number of points to be drawn.
		\param[in] shadowed		Whether we are drawing the shadowed/shaded particles.
		\see drawPoints()
	*/
	void drawPointSprites(ShaderProgram *shader, int start, int count, bool shadowed);

	//! Simply draws the given number of points from start index.
	/*!
		Uses glDrawElements if points are sorted (use EBO).
		\param[in] start		Start index of the points to be drawn.
		\param[in] count		Number of points to be drawn.
		\param[in] sorted		Whether the points use EBO and should be drawn using glDrawElements call.
	*/
	void drawPoints(int start, int count, bool sorted);

	//! Composites the image result to the main framebuffer.
	void compositeResult();

	//! Blurs the currently active light texture.
	void blurLightTexture();


};

