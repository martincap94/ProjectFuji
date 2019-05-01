///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Actor.h
* \author     Martin Cap
* \brief      Contains Actor class.
*
*	This file describes the Actor class. An actor is any object that is part of the scene hierarchy
*	of our engine (i.e. it is the same as Actor in Unreal Engine or GameObject in Unity).
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>
#include <string>

#include <glm\glm.hpp>


#include "Transform.h"
#include "ShaderProgram.h"

class HeightMap;

class Actor {
public:

	Transform transform;
	//glm::mat4 modelMatrix;
	std::string name;

	int visible = 1;
	int castShadows = 1;
	int selected = 0;


	Actor *parent = nullptr;
	int pidx = -1; // idx in its parent's children vector
	std::vector<Actor *> children;

	Actor();
	Actor(std::string name);
	~Actor();

	virtual void update();
	virtual bool draw();
	virtual bool draw(ShaderProgram *shader);
	virtual bool drawGeometry(ShaderProgram *shader);
	virtual bool drawShadows(ShaderProgram *shader);
	virtual bool drawWireframe(ShaderProgram *shader);

	virtual void addChild(Actor *child);
	virtual void unparent(bool keepWorldPosition = true);

	bool isRootChild();

	void snapToGround(HeightMap *heightMap);

protected:

	virtual bool shouldDraw();


};

