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


private:

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;

	bool instanced = false;
	int numInstances = 0;
	GLuint instancesVBO;


	void setupMesh();

};

