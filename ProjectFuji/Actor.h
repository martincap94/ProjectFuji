///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Actor.h
* \author     Martin Cap
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

//! Basic Actor class that represents any object in the scene.
/*!
	Basic Actor class representing any object in the scene.
	The actor can be drawn in multiple ways (normally, geometry, wireframe) and can be updated.
*/
class Actor {
public:

	Transform transform;		//!< Holds all transformation data about this actor
	std::string name;			//!< Name of the actor (not unique at the moment)

	int visible = 1;			//!< Whether the actor is visible or not
	int castShadows = 1;		//!< Whether the actor casts shadows
	int selected = 0;			//!< Whether the actor is selected in the hierarchy browser


	Actor *parent = nullptr;	//!< Parent of the actor
	int pidx = -1;				//!< Index in its parent's children vector
	std::vector<Actor *> children;	//!< List of its children

	//! Basic constructor.
	/*!
		Basic constructor of the actor. Initializes the transform member.
	*/
	Actor();

	//! Basic constructor with given name.
	/*!
		Constructs the actors with the given name and initializes the transform member.
	*/
	Actor(std::string name);

	//! Basic destructor. No heap data is destroyed.
	~Actor();

	//! Updates the actor for next draw call.
	/*!
		Updates the actor for next draw call.
		The model matrix and all children of this actor are updated.
	*/
	virtual void update();

	//! Draws the actor using its shader member.
	virtual bool draw();

	//! Draws the actor using the given shader.
	virtual bool draw(ShaderProgram *shader);

	//! Draws only the geometry of this actor with the given shader.
	virtual bool drawGeometry(ShaderProgram *shader);

	//! Draws only the geometry of this actor if castShadows is enabled.
	virtual bool drawShadows(ShaderProgram *shader);

	//! Draws the actor's wireframe using the given shader.
	virtual bool drawWireframe(ShaderProgram *shader);

	//! Adds a child for this actor.
	virtual void addChild(Actor *child);

	//! Moves up in the hierarchy by unparenting from current parent.
	/*!
		Moves up in the hierarchy by unparenting from current parent.
		The pidx (parent idx) is used for faster removal. If pidx not set,
		this actor must be manually found in its parent's vector.
	*/
	virtual void unparent(bool keepWorldPosition = true);

	//! Returns whether this actor is a child of root.
	bool isChildOfRoot();

	//! Snaps the parent to ground of the given terrain.
	void snapToGround(HeightMap *heightMap);

protected:

	//! Returns whether the actor should be drawn.
	virtual bool shouldDraw();


};

