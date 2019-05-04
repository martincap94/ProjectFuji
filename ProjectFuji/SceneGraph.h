///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       SceneGraph.h
* \author     Martin Cap
*
*	A very simple SceneGraph that holds a DAG hierarchy of Actor objects.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once


#include "Actor.h"

//! Very simple graph of scene.
class SceneGraph {
public:

	Actor *root = nullptr;	//!< Root of the scene.

	//! Default constructor. 
	SceneGraph();

	//! Default destructor.
	~SceneGraph();
};

