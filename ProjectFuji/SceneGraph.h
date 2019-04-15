#pragma once


#include "Actor.h"

class SceneGraph {
public:

	Actor *root = nullptr;

	SceneGraph();
	~SceneGraph();
};

