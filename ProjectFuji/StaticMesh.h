#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "ShaderProgram.h"
#include "DataStructures.h"
#include "Transform.h"
#include "Texture.h"
#include "Material.h"



class StaticMesh {

public:
	unsigned int id;
	const char *meshPath;
	std::vector<MeshVertex> vertices;
	std::vector<Texture> textures;
	Material *material;
	ShaderProgram *shader;
	Transform transform;


	StaticMesh();
	StaticMesh(const char *meshPath, ShaderProgram *shader, Material *material);
	~StaticMesh();

	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void draw(const glm::mat4 &ownerGlobalTransformMatrix);
	virtual void drawSimple(const glm::mat4 &globalTransformMatrix);
	virtual void drawShadow(const glm::mat4 &globalTransformMatrix, ShaderProgram &shader);

	virtual void draw(const Transform &transform, ShaderProgram &shader);
	virtual void draw(const Transform &transform);

protected:
	unsigned int VAO, VBO, EBO, tangentVBO, bitangentVBO;

	virtual bool setupMesh(const char *meshPath);

	//StaticMesh(StaticMesh const &staticMesh) = delete;
	//void operator=(StaticMesh const &staticMesh) = delete;

private:
	static unsigned int idCounter;

};


