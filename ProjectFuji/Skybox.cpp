#include "Skybox.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Cubemap.h"
#include "DataStructures.h"

Skybox::Skybox() {
	setupSkybox();
}


Skybox::~Skybox() {
}

void Skybox::draw(ShaderProgram &shader) {
	glDepthMask(GL_FALSE);
	shader.use();

	/*view = glm::mat4(glm::mat3(camera.GetViewMatrix()));
	skyboxShader.setMat4fv("view", view);*/

	glBindVertexArray(VAO);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
	//glDrawArrays(GL_TRIANGLES, 0, 36);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);


	glDepthMask(GL_TRUE);
}

void Skybox::setupSkybox() {

	skyboxTexture = loadCubemap(faces);

	/*
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	*/

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVerticesNew), skyboxVerticesNew, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyboxIndicesNew), skyboxIndicesNew, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);


}