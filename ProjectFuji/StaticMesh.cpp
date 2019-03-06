#include "StaticMesh.h"

#include "obj_loader.h"

unsigned int StaticMesh::idCounter = 0;

StaticMesh::StaticMesh() {

}


StaticMesh::StaticMesh(const char *meshPath, ShaderProgram *shader, Material *material)
 : meshPath(meshPath), shader(shader), material(material), id(idCounter++) {
	bool success = setupMesh(meshPath);
	if (!success) {
		std::cout << " *** ERROR *** " << std::endl;
		std::cout << " *** Failed to load mesh at '" << meshPath << "' !" << std::endl;
		std::cout << " ************* " << std::endl;
	}
}

StaticMesh::~StaticMesh() {
}


void StaticMesh::draw() {
	shader->use();
	shader->setModelMatrix(transform.getModelMatrix());
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

void StaticMesh::draw(const glm::mat4 &ownerGlobalTransformMatrix) {
	shader->use();
	shader->setModelMatrix(ownerGlobalTransformMatrix * transform.getModelMatrix());
	material->use(*shader);
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

void StaticMesh::drawSimple(const glm::mat4 &globalTransformMatrix) {
	shader->use();
	shader->setModelMatrix(globalTransformMatrix);
	//material->use(*shader);
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

void StaticMesh::drawShadow(const glm::mat4 &globalTransformMatrix, ShaderProgram &shader) {
	shader.setModelMatrix(globalTransformMatrix);
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

void StaticMesh::draw(const Transform &transform, ShaderProgram &shader) {
	shader.use();
	shader.setModelMatrix(transform.getModelMatrix());
	for (unsigned int i = 0; i < textures.size(); i++) {
		textures[i].useTexture();
	}
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

void StaticMesh::draw(const Transform &transform) {
	shader->use();
	shader->setModelMatrix(transform.getModelMatrix());

	for (unsigned int i = 0; i < textures.size(); i++) {
		textures[i].useTexture();
	}
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

bool StaticMesh::setupMesh(const char *meshPath) {

	std::vector<glm::vec3> verticePositions;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec3> tangents;
	std::vector<glm::vec3> bitangents;

	bool success = loadObj(meshPath, vertices, verticePositions, uvs, normals, tangents, bitangents);
	if (!success) {
		return false;
	}

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &tangentVBO);
	glGenBuffers(1, &bitangentVBO);
	//glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(MeshVertex), &vertices[0], GL_STATIC_DRAW);

	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, normal));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, texCoords));

	glBindBuffer(GL_ARRAY_BUFFER, tangentVBO);
	glBufferData(GL_ARRAY_BUFFER, tangents.size() * sizeof(glm::vec3), &tangents[0], GL_STATIC_DRAW);
	
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, bitangentVBO);
	glBufferData(GL_ARRAY_BUFFER, bitangents.size() * sizeof(glm::vec3), &bitangents[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);


	glBindVertexArray(0);

	return true;

}