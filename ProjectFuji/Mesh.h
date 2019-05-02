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

class Mesh {
public:

	std::vector<MeshVertex> vertices;
	std::vector<GLuint> indices;
	std::vector<Texture> textures;

	Mesh(std::vector<MeshVertex> vertices, std::vector<GLuint> indices, std::vector<Texture> textures);
	~Mesh();

	void draw(ShaderProgram *shader);

	void makeInstanced(std::vector<Transform> &instanceTransforms);
	void updateInstanceTransforms(std::vector<Transform> &instanceTransforms);
	void updateInstanceModelMatrices(std::vector<glm::mat4> &instanceModelMatrices);

private:

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;

	bool instanced = false;
	int numInstances = 0;
	GLuint instancesVBO;


	void setupMesh();
	void initInstancedMeshBuffers();

};

