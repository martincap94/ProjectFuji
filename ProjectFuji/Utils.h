///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Utils.h
* \author     Martin Cap
*
*	Contains general utility functions to be used across the application such as rangeToRange, 
*	intersections, normalizations, printing of vectors, etc.
*	Citations to outer resources are in function descriptions.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>
#include <string>
#include <glad\glad.h>

constexpr double PI = 3.14159265358979323846;	//!< Double value of pi



/*
--- THIS LINE INTERSECTION APPROACH IS NO LONGER USED ---
Line intersection taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/
created by Martin Thoma
*/

//! Checks whether bounding boxes (amin, amax) and (bmin, bmax) intersect.
/*!
	Checks whether bounding boxes (amin, amax) and (bmin, bmax) intersect.
	Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

	\deprecated No longer used due to more performant line intersection algorithm.
	\author Martin Thoma

	\param[in] amin		min extents of BBox A
	\param[in] amax		max extents of BBox A
	\param[in] bmin		min extents of BBox B
	\param[in] bmax		max extents of BBox B
	\return				true if bounding boxes intersect, false otherwise
*/
bool doBoundingBoxesIntersect(const glm::vec2 &amin, const glm::vec2 &amax, const glm::vec2 &bmin, const glm::vec2 &bmax);

//! Checks whether a given point is on a line defined by first and second.
/*!
	Checks whether a given point is on a line defined by first and second.
	Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

	\deprecated No longer used due to more performant line intersection algorithm.
	\author Martin Thoma

	\param[in] first	first point defining the line
	\param[in] second	second point defining the line
	\param[in] point	point to check
	\return				true if point is on line, false otherwise
*/
bool isPointOnLine(glm::vec2 first, glm::vec2 second, glm::vec2 point);

//! Checks whether a point is to the right of a line (orientation predicate - float precision with epsilon).
/*!
	Checks whether a point is to the right of a line.
	Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

	\deprecated No longer used due to more performant line intersection algorithm.
	\author Martin Thoma

	\param[in] first	first point defining the line
	\param[in] second	second point defining the line
	\param[in] point	point to check
	\return				true if point is to the right from the line, false otherwise
*/
bool isPointRightOfLine(glm::vec2 first, glm::vec2 second, glm::vec2 point);

//! Checks whether a line segment touches or crosses a line using "point on line" and "is point right of line" tests.
/*!
	Checks whether a line segment touches or crosses a line using "point on line" and "is point right of line" tests.
	Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

	\deprecated No longer used due to more performant line intersection algorithm.
	\author Martin Thoma

	\param[in] afirst	first point of line
	\param[in] asecond	second point of line
	\param[in] bfirst	first point of line segment
	\param[in] bsecond	second point of line segment
	\return				true if line segment touches or crosses the line, false otherwise
*/
bool lineSegmentTouchesOrCrossesLine(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond);

//! Checks whether two line segments intersect.
/*!
	Checks whether two line segments intersect.
	Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

	\deprecated No longer used due to more performant line intersection algorithm.
	\author Martin Thoma

	\param[in] afirst	first point of line segment A
	\param[in] asecond	second point of line segment A
	\param[in] bfirst	first point of line segment B
	\param[in] bsecond	second point of line segment B
	\return				true if line segments intersect, false otherwise
*/
bool doLineSegmentsIntersect(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond);


//! Returns an intersection of two lines. Assumes that the lines intersect!!!
/*!
	Returns an intersection point of two lines. Assumes that the lines intersect!!!
	Taken from: http://paulbourke.net/geometry/pointlineplane/javascript.txt

	\author Paul Bourke

	\param[in] afirst	first point of line segment A
	\param[in] asecond	second point of line segment A
	\param[in] bfirst	first point of line segment B
	\param[in] bsecond	second point of line segment B
	\return				point of intersecton
*/
glm::vec2 getIntersectionPoint(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond);

//! Returns an intersection of two lines. Assumes that the lines intersect!!!
/*!
	Returns an intersection point of two lines. Assumes that the lines intersect!!!
	Taken from: http://paulbourke.net/geometry/pointlineplane/javascript.txt

	\author Paul Bourke

	\param[in] afirst			first point of line segment A
	\param[in] asecond			second point of line segment A
	\param[in] bfirst			first point of line segment B
	\param[in] bsecond			second point of line segment B
	\param[in] intersection		out reference to intersection point
	\return						whether the two segments intersect
*/
bool getIntersectionPoint(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond, glm::vec2 &intersection);


/*
Trimming functions taken from:
 https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string
 -> answer by user Galik
*/


//! Trim from left.
inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v") {
	s.erase(0, s.find_first_not_of(t));
	return s;
}

//! Trim from right.
inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v") {
	s.erase(s.find_last_not_of(t) + 1);
	return s;
}

//! Trim from left & right.
inline std::string& trim(std::string& s, const char* t = " \t\n\r\f\v") {
	return ltrim(rtrim(s, t), t);
}

// copying versions
//! Trim from left.
inline std::string ltrim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return ltrim(s, t);
}

//! Trim from right.
inline std::string rtrim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return rtrim(s, t);
}

//! Trim from left & right.
inline std::string trim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return trim(s, t);
}

//! Whether string starts with given prefix (needle).
inline bool string_starts_with(const std::string &haystack, const std::string &needle) {
	return (haystack.find_first_of(needle) == 0);
}


//! Normalize input value from range to [0,1].
void normalizeFromRange(float &val, float min = 0.0f, float max = 1.0f);

//! Normalize input vec3 from range to [0,1] (per member).
void normalizeFromRange(glm::vec3 &val, float min = 0.0f, float max = 1.0f);

//! Get normalized vec3 from range to [0,1] (per member). Does not modify input vec3.
glm::vec3 getNormalizedFromRange(glm::vec3 val, float min = 0.0f, float max = 1.0f);

//! Transform input value from original range [origMin, origMax] to new range [newMin, newMax].
void rangeToRange(float &val, float origMin = 0.0f, float origMax = 1.0f, float newMin = 0.0f, float newMax = 1.0f);

//! Transform input vec3 (per member) from original range [origMin, origMax] to new range [newMin, newMax].
void rangeToRange(glm::vec3 &val, float origMin = 0.0f, float origMax = 1.0f, float newMin = 0.0f, float newMax = 1.0f);

//! Transform input vec3 (per member) from original vec3 range to new vec3 range (per member).
void rangeToRange(glm::vec3 &val, glm::vec3 origMin = glm::vec3(0.0f), glm::vec3 origMax = glm::vec3(1.0f), glm::vec3 newMin = glm::vec3(0.0f), glm::vec3 newMax = glm::vec3(1.0f));

//! Transform input vec2 (per member) from original vec2 range to new vec2 range (per member).
void rangeToRange(glm::vec2 &val, glm::vec2 origMin = glm::vec2(0.0f), glm::vec2 origMax = glm::vec2(1.0f), glm::vec2 newMin = glm::vec2(0.0f), glm::vec2 newMax = glm::vec2(1.0f));



//! Get error string based on OpenGL error code.
std::string getGLErrorString(unsigned int err);

//! Reports OpenGL errors with given message appended.
void reportGLErrors(std::string message = "");

//! Reports OpenGL errors with file and line parameters appended.
void reportGLErrors(const char *file, int line);

//! Checks for all OpenGL errors and prints them with file and line information to console.
#define CHECK_GL_ERRORS() ( reportGLErrors( __FILE__, __LINE__ ) )

//! Prints an error message about unimplemented function for given file, function and line.
void reportUnimplementedFunction(const char *file, const char *function, int line);

//! Reports that a function is not implemented with information about the file, function and line.
#define REPORT_NOT_IMPLEMENTED() ( reportUnimplementedFunction( __FILE__, __FUNCTION__, __LINE__ ) )

//! Returns random float in specified range. Uses regular rand() call to generate random distribution.
float getRandFloat(float min = 0.0f, float max = 1.0f);




//! Finds extension of filename, returns whether any extension exists.
/*!
	Finds extension of filename, returns whether any extension exists.
	Only suitable for filenames, not paths! 
	(PathFindExtensionA for windows is suitable for paths e.g.)
	Based on: https://stackoverflow.com/questions/51949/how-to-get-file-extension-from-string-in-c/51993#51993
	\param[in] filename			Filename to be examined.
	\param[out] outExtension	The found extension, not set if does not exist.
	\param[in] to_lower			Whether to convert the extension to lower case.
	\return	True if extension exists, false otherwise.
*/
bool getFileExtension(const std::string &filename, std::string &outExtension, bool to_lower = true);


//! Simple texture creation helper.
/*!
	Creates (simple) OpenGL texture based on given parameters.
	\param target		Texture target (e.g. GL_TEXTURE_2D).
	\param w			Width of the texture.
	\param h			Height of the texture.
	\param internalFormat	Internal format for OpenGL.
	\param format			Format of the data.
	\return Texture id.
*/
GLuint createTextureHelper(GLenum target, int w, int h, GLint internalFormat, GLenum format);

//! Prints glm::vec2 to console.
void printVec2(const glm::vec2 &v);
//! Prints glm::vec3 to console.
void printVec3(const glm::vec3 &v);
//! Prints glm::vec4 to console.
void printVec4(const glm::vec4 &v);


//! Returns a formatted time current time string.
/*!
	Taken from: https://stackoverflow.com/questions/17223096/outputting-date-and-time-in-c-using-stdchrono
*/
std::string getTimeStr();



