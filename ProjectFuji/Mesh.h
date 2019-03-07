#pragma once

#include <glad\glad.h>
#include <vector>

#include "DataStructures.h"
#include "Texture.h"
#include "ShaderProgram.h"

class Mesh {
public:

	std::vector<MeshVertex> vertices;
	std::vector<GLuint> indices;
	std::vector<Texture> textures;

	Mesh(std::vector<MeshVertex> vertices, std::vector<GLuint> indices, std::vector<Texture> textures);
	~Mesh();

	void draw(ShaderProgram &shader);

private:

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;

	void setupMesh();

};

