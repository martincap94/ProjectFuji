///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Curve.h
* \author     Martin Cap
*
*	This file describes a very simple Curve class that is used in the SkewT/LogP diagram.
*	The curve intersection is based on theroy from http://paulbourke.net/geometry/pointlineplane 
*	& http://www.cs.swan.ac.uk/~cssimon/line_intersection.html. 
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>

#include <vector>

#include <glad\glad.h>
#include "ShaderProgram.h"


using namespace std;

//! Very simple STLPDiagram curve implementation.
/*!
	Very simple STLPDiagram curve implementation that mainly stores the curve data.
	Generally, we do not want to draw individual curves, we cluster them into larger VBOs which
	is done manually, that is why this class is so limited in function.
*/
class Curve {
public:


	vector<glm::vec2> vertices;		//!< Vertices of the curve
	glm::vec3 color = glm::vec3(0.1f);

	//! Default constructor. 
	Curve();

	//! Default destructor.
	~Curve();

	//! Initialize OpenGL buffers for curve drawing.
	void initBuffers();

	//! Upload curve vertex data to buffers.
	/*!
		Upload curve vertex data to buffers.
		Warning: Assumes that buffers are initialized and that curve data is not empty!
	*/
	void uploadToBuffers();

	
	
	//! Initializes the curve buffers and uploads the curve vertex data.
	void init();

	//! Draws the curve with the given shader reference.
	void draw(ShaderProgram &shader);

	//! Draws the curve with the given shader.
	void draw(ShaderProgram *shader);

	//! --- NOT USED --- Naively finds intersection with isobar.
	/*!
		--- NOT USED ---
		Naively finds intersection with isobar. 
		A optimized GPU implementation is used instead that uses a binary search.
	*/
	glm::vec2 getIntersectionWithIsobar(float normalizedPressure);

	//! Prints vertices of the curve to the console.
	void printVertices();

private:

	GLuint VAO;
	GLuint VBO;


};


//! Finds intersection between given curves and returns whether intersection found.
/*!
	Based on theory from: http://paulbourke.net/geometry/pointlineplane
						& http://www.cs.swan.ac.uk/~cssimon/line_intersection.html
	\param[in] first				First curve.
	\param[in] second				Second curve.
	\param[out] outIntersection		Intersection of the two curves.
	\param[in] intersectionNumber	The number of the intersection (e.g. if we are interested in second found intersection it equals 2).
	\param[in] skipStartingPoints	Whether to skip intersection between either curve and other's starting point (with small epsilon offset).
	\param[in] reverseFirst			Whether to reverse the order of vertices for the first curve.
	\param[in] reverseSecond		Whether to reverse the order of vertices for the second curve.
	\return Whether the two curves intersect.
*/

bool findIntersectionNew(const Curve &first, const Curve &second, glm::vec2 &outIntersection, unsigned int intersectionNumber = 1, bool skipStartingPoints = false, bool reverseFirst = false, bool reverseSecond = false);

//! Finds intersection between given curves and returns the found intersection.
/*!
	Finds intersection between given curves and returns the found intersection (or zero vector if none found).
	Based on theory from: http://paulbourke.net/geometry/pointlineplane
						& http://www.cs.swan.ac.uk/~cssimon/line_intersection.html
	\param[in] first			First curve.
	\param[in] second			Second curve.
	\param[in] reverseFirst		Whether to reverse the order of vertices for the first curve.
	\param[in] reverseSecond	Whether to reverse the order of vertices for the second curve.
	\return The curve intersection.
*/
glm::vec2 findIntersection(const Curve &first, const Curve &second, bool reverseFirst = false, bool reverseSecond = false);

//! --- DEPRECATED --- Finds intersection between two curves.
/*!
	--- DEPRECATED ---
	Finds intersection between two curves.
	Based on: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

	\param[in] first	Reference to the first curve.
	\param[in] second	Reference to the second curve.

	\deprecated		Ineffective approach that is not used anymore.
*/
glm::vec2 findIntersectionOld(const Curve &first, const Curve &second);

