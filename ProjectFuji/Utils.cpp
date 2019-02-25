#include "Utils.h"

#include "DataStructures.h"

// Line intersection based on: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/
bool doBoundingBoxesIntersect(const glm::vec2 &amin, const glm::vec2 &amax, const glm::vec2 &bmin, const glm::vec2 &bmax) {
	return amin.x <= bmax.x &&
		amax.x >= bmin.x &&
		amin.y <= bmax.y &&
		amax.y >= bmin.y;
}

bool isPointOnLine(glm::vec2 first, glm::vec2 second, glm::vec2 point) {
	glm::vec2 tmp(0.0f);
	glm::vec2 tmp2(second.x - first.x, second.y - first.y);
	glm::vec2 tmpPoint(point.x - first.x, point.y - first.y);
	float r = tmp2.x * tmpPoint.y - tmpPoint.x * tmp2.y;
	return abs(r) < FLT_EPSILON;

}

bool isPointRightOfLine(glm::vec2 first, glm::vec2 second, glm::vec2 point) {
	glm::vec2 tmp(0.0f);
	glm::vec2 tmp2(second.x - first.x, second.y - first.y);
	glm::vec2 tmpPoint(point.x - first.x, point.y - first.y);
	return ((tmp2.x * tmpPoint.y - tmpPoint.x * tmp2.y) < 0.0f);
}

bool lineSegmentTouchesOrCrossesLine(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond) {
	return isPointOnLine(afirst, asecond, bfirst) ||
		isPointOnLine(afirst, asecond, bsecond) ||
		(isPointRightOfLine(afirst, asecond, bfirst) ^ isPointRightOfLine(afirst, asecond, bsecond));
}

bool doLineSegmentsIntersect(glm::vec2 afirst, glm::vec2 asecond, glm::vec2 bfirst, glm::vec2 bsecond) {
	glm::vec2 amin;
	glm::vec2 amax;
	glm::vec2 bmin;
	glm::vec2 bmax;

	amin = glm::vec2(afirst.x <= asecond.x ? afirst.x : asecond.x, afirst.y <= asecond.y ? afirst.y : asecond.y);
	amax = glm::vec2(afirst.x > asecond.x ? afirst.x : asecond.x, afirst.y > asecond.y ? afirst.y : asecond.y);

	bmin = glm::vec2(bfirst.x <= bsecond.x ? bfirst.x : bsecond.x, bfirst.y <= bsecond.y ? bfirst.y : bsecond.y);
	bmax = glm::vec2(bfirst.x > bsecond.x ? bfirst.x : bsecond.x, bfirst.y > bsecond.y ? bfirst.y : bsecond.y);

	return doBoundingBoxesIntersect(amin, amax, bmin, bmax) && lineSegmentTouchesOrCrossesLine(afirst, asecond, bfirst, bsecond) && lineSegmentTouchesOrCrossesLine(bfirst, bsecond, afirst, asecond);
}

// Based on: http://paulbourke.net/geometry/pointlineplane
glm::vec2 getIntersectionPoint(glm::vec2 P1, glm::vec2 P2, glm::vec2 P3, glm::vec2 P4) {

	//if (((P1.x == P2.x) && (P1.y == P2.y)) || ((P3.x == P4.x) && (P3.y == P4.y))) {
	//	return glm::vec2(-1.0f);
	//}

	//float denominator = ((P4.y - P3.y) * (P2.x - P1.x) - (P4.x - P3.x) * (P2.y - P1.y));
	float denominator = ((P4.x - P3.x) * (P2.y - P1.y) - (P2.x - P1.x) * (P4.y - P3.y));

	if (fabs(denominator) <= FLT_EPSILON) {
		return glm::vec2(-1.0f); // this can also mean that they intersect in infinite number of points (add some flag later)
	}

	//float t_a = ((P4.x - P3.x) * (P1.y - P3.y) - (P4.y - P3.y) * (P1.x - P3.x)) / denominator;
	//float t_b = ((P2.x - P1.x) * (P1.y - P3.y) - (P2.y - P1.y) * (P1.y - P3.x)) / denominator;

	float t_a = ((P4.x - P3.x) * (P3.y - P1.y) - (P3.x - P1.x) * (P4.y - P3.y)) / denominator;
	float t_b = ((P2.x - P1.x) * (P3.y - P1.y) - (P3.x - P1.x) * (P2.y - P1.y)) / denominator;


	if (t_a < 0.0f || t_a > 1.0f || t_b < 0.0f || t_b > 1.0f) {
		return glm::vec2(-1.0f);
	}
	float x = P1.x + t_a * (P2.x - P1.x);
	float y = P1.y + t_a * (P2.y - P1.y);

	return glm::vec2(x, y);
}

// Based on: http://paulbourke.net/geometry/pointlineplane
bool getIntersectionPoint(glm::vec2 P1, glm::vec2 P2, glm::vec2 P3, glm::vec2 P4, glm::vec2 &intersection) {

	float denominator = ((P4.x - P3.x) * (P2.y - P1.y) - (P2.x - P1.x) * (P4.y - P3.y));

	if (fabs(denominator) <= FLT_EPSILON) {
		return false; // this can also mean that they intersect in infinite number of points (add some flag later)
	}

	float t_a = ((P4.x - P3.x) * (P3.y - P1.y) - (P3.x - P1.x) * (P4.y - P3.y)) / denominator;
	float t_b = ((P2.x - P1.x) * (P3.y - P1.y) - (P3.x - P1.x) * (P2.y - P1.y)) / denominator;


	if (t_a < 0.0f || t_a > 1.0f || t_b < 0.0f || t_b > 1.0f) {
		return false;
	}
	float x = P1.x + t_a * (P2.x - P1.x);
	float y = P1.y + t_a * (P2.y - P1.y);

	intersection = glm::vec2(x, y);
	return true;
}

void normalizeFromRange(float &val, float min, float max) {
	val = (val - min) / (max - min);
}

void rangeToRange(float &val, float origMin, float origMax, float newMin, float newMax) {
	normalizeFromRange(val, origMin, origMax);
	val *= (newMax - newMin);
	val += newMin;
}

void normalizeFromRange(glm::vec3 &val, float min, float max) {
	val.x = (val.x - min) / (max - min);
	val.y = (val.y - min) / (max - min);
	val.z = (val.z - min) / (max - min);
}

glm::vec3 getNormalizedFromRange(glm::vec3 val, float min, float max) {
	normalizeFromRange(val, min, max);
	return val;
}


