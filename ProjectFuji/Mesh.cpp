#include "Mesh.h"



Mesh::Mesh(std::vector<MeshVertex> vertices, std::vector<GLuint> indices, std::vector<Texture> textures) : vertices(vertices), indices(indices), textures(textures) {
	setupMesh();
}

Mesh::~Mesh() {
}

void Mesh::draw(ShaderProgram *shader) {
	// Currently - one model, one material

	// TODO textures (naming and binding conventions)

	//shader.use();
	glBindVertexArray(VAO);

	if (instanced) {
		glDrawElementsInstanced(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0, numInstances);
	} else {
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
	}
	glBindVertexArray(0);

}

void Mesh::makeInstanced(std::vector<Transform> &instanceTransforms) {
	if (instanced) {
		cout << "This mesh is already instanced... Updating instance transforms!" << endl;
		updateInstanceTransforms(instanceTransforms);
		return;
	}
	initInstancedMeshBuffers();
	updateInstanceTransforms(instanceTransforms);
	instanced = true;
}

void Mesh::updateInstanceTransforms(std::vector<Transform>& instanceTransforms) {
	numInstances = instanceTransforms.size();
	vector<glm::mat4> instanceModelMatrices;
	for (int i = 0; i < numInstances; i++) {
		instanceModelMatrices.push_back(instanceTransforms[i].getModelMatrix());
	}
	glBindBuffer(GL_ARRAY_BUFFER, instancesVBO);
	glBufferData(GL_ARRAY_BUFFER, numInstances * sizeof(glm::mat4), instanceModelMatrices.data(), GL_STATIC_DRAW);
}

void Mesh::updateInstanceModelMatrices(std::vector<glm::mat4> &instanceModelMatrices) {
	numInstances = instanceModelMatrices.size();

	glBindBuffer(GL_ARRAY_BUFFER, instancesVBO);
	glBufferData(GL_ARRAY_BUFFER, numInstances * sizeof(glm::mat4), instanceModelMatrices.data(), GL_STATIC_DRAW);
}



void Mesh::setupMesh() {
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(MeshVertex), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
				 &indices[0], GL_STATIC_DRAW);

	// vertex positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)0);
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, normal));
	// vertex texture coords
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, texCoords));

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, tangent));

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, bitangent));
	glBindVertexArray(0);
}

void Mesh::initInstancedMeshBuffers() {

	glBindVertexArray(VAO);
	glGenBuffers(1, &instancesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, instancesVBO);

	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);

	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));

	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));

	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));

	glVertexAttribDivisor(5, 1);
	glVertexAttribDivisor(6, 1);
	glVertexAttribDivisor(7, 1);
	glVertexAttribDivisor(8, 1);

	glBindVertexArray(0);
}
