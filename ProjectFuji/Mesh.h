///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Mesh.h
* \author     Martin Cap
*
*	Describes the Mesh class that is used to represent a loaded mesh. We assume that objects may
*	be formed from multiple meshes, hence the mesh class is wrapped with Model class which can
*	be composited from multiple meshes and is the general representation of a model in our engine.
*	Based on Joey de Vries's tutorials: https://learnopengl.com/Model-Loading/Mesh
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <vector>

#include "DataStructures.h"
#include "Texture.h"
#include "ShaderProgram.h"
#include "Transform.h"

//! Simple mesh representation.
/*!
	Based on Joey de Vries's tutorials: https://learnopengl.com/Model-Loading/Mesh
*/
class Mesh {
public:

	std::vector<MeshVertex> vertices;	//!< Vertices of the mesh
	std::vector<GLuint> indices;		//!< Indices of the mesh
	std::vector<Texture> textures;		//!< --- NOT USED --- Textures used by the mesh

	//! Creates a mesh using the given parameters.
	Mesh(std::vector<MeshVertex> vertices, std::vector<GLuint> indices, std::vector<Texture> textures);

	//! Default destructor.
	~Mesh();

	//! Draws the mesh using the given shader.
	void draw(ShaderProgram *shader);

	//! Makes the mesh instanced given an array of transforms for all instances.
	void makeInstanced(std::vector<Transform> &instanceTransforms);

	//! Updates the instance transforms if the mesh is instanced.
	void updateInstanceTransforms(std::vector<Transform> &instanceTransforms);

	//! Updates model matrices of all instances.
	void updateInstanceModelMatrices(std::vector<glm::mat4> &instanceModelMatrices);

private:

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;

	bool instanced = false;		//!< Whether the mesh is instanced or not
	int numInstances = 0;		//!< Number of instances
	GLuint instancesVBO;		//!< VBO for instance model matrices

	//! Initializes the necessary buffers for the mesh.
	void setupMesh();

	//! Initializes the instance buffers.
	void initInstancedMeshBuffers();

};

