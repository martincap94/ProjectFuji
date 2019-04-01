#pragma once

#include "Transform.h"

class HeightMap;

class Actor {
public:

	Transform transform;

	Actor();
	~Actor();

	void snapToGround(HeightMap *heightMap);


};

