#pragma once

#include <glm\glm.hpp>
#include <string>
//#include <glad\glad.h>

constexpr double PI = 3.14159265358979323846;



/*
Line intersection taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/
created by Martin Thoma
*/

/// Checks whether bounding boxes (amin, amax) and (bmin, bmax) intersect.
/**
Checks whether bounding boxes (amin, amax) and (bmin, bmax) intersect.
Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

\author Martin Thoma

\param[in] amin		min extents of BBox A
\param[in] amax		max extents of BBox A
\param[in] bmin		min extents of BBox B
\param[in] bmax		max extents of BBox B
\return				true if bounding boxes intersect, false otherwise
*/
bool doBoundingBoxesIntersect(const glm::vec2 &amin, const glm::vec2 &amax, const glm::vec2 &bmin, const glm::vec2 &bmax);

/// Checks whether a given point is on a line defined by first and second.
/**
Checks whether a given point is on a line defined by first and second.
Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

\author Martin Thoma

\param[in] first	first point defining the line
\param[in] second	second point defining the line
\param[in] point	point to check
\return				true if point is on line, false otherwise
*/
bool isPointOnLine(glm::vec2 first, glm::vec2 second, glm::vec2 point);

/// Checks whether a point is to the right of a line (orientation predicate - float precision with epsilon).
/**
Checks whether a point is to the right of a line.
Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

\author Martin Thoma

\param[in] first	first point defining the line
\param[in] second	second point defining the line
\param[in] point	point to check
\return				true if point is to the right from the line, false otherwise
*/
bool isPointRightOfLine(glm::vec2 first, glm::vec2 second, glm::vec2 point);

/// Checks whether a line segment touches or crosses a line using "point on line" and "is point right of line" tests.
/**
Checks whether a line segment touches or crosses a line using "point on line" and "is point right of line" tests.
Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

\author Martin Thoma

\param[in] afirst	first point of line
\param[in] asecond	second point of line
\param[in] bfirst	first point of line segment
\param[in] bsecond	second point of line segment
\return				true if line segment touches or crosses the line, false otherwise
*/
bool lineSegmentTouchesOrCrossesLine(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond);

/// Checks whether two line segments intersect.
/**
Checks whether two line segments intersect.
Taken from: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

\author Martin Thoma

\param[in] afirst	first point of line segment A
\param[in] asecond	second point of line segment A
\param[in] bfirst	first point of line segment B
\param[in] bsecond	second point of line segment B
\return				true if line segments intersect, false otherwise
*/
bool doLineSegmentsIntersect(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond);


/// Returns an intersection of two lines. Assumes that the lines intersect!!!
/**
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

/// Returns an intersection of two lines. Assumes that the lines intersect!!!
/**
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


// Trimming functions taken from: https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string
// answer by user Galik

/// Trim from left.
inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v") {
	s.erase(0, s.find_first_not_of(t));
	return s;
}

/// Trim from right.
inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v") {
	s.erase(s.find_last_not_of(t) + 1);
	return s;
}

/// Trim from left & right.
inline std::string& trim(std::string& s, const char* t = " \t\n\r\f\v") {
	return ltrim(rtrim(s, t), t);
}

// copying versions
/// Trim from left.
inline std::string ltrim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return ltrim(s, t);
}

/// Trim from right.
inline std::string rtrim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return rtrim(s, t);
}

/// Trim from left & right.
inline std::string trim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return trim(s, t);
}



void normalizeFromRange(float &val, float min = 0.0f, float max = 1.0f);
void rangeToRange(float &val, float origMin = 0.0f, float origMax = 1.0f, float newMin = 0.0f, float newMax = 1.0f);
void rangeToRange(glm::vec3 &val, float origMin = 0.0f, float origMax = 1.0f, float newMin = 0.0f, float newMax = 1.0f);
void rangeToRange(glm::vec3 &val, glm::vec3 origMin = glm::vec3(0.0f), glm::vec3 origMax = glm::vec3(1.0f), glm::vec3 newMin = glm::vec3(0.0f), glm::vec3 newMax = glm::vec3(1.0f));

void normalizeFromRange(glm::vec3 &val, float min = 0.0f, float max = 1.0f);

glm::vec3 getNormalizedFromRange(glm::vec3 val, float min = 0.0f, float max = 1.0f);


std::string getGLErrorString(unsigned int err);
void reportGLErrors(std::string message = "");
void reportGLErrors(const char *file, int line);


#define CHECK_GL_ERRORS() ( reportGLErrors( __FILE__, __LINE__ ) )


float getRandFloat(float min, float max);
