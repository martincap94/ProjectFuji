#include "Skybox.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Cubemap.h"

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
	glDrawArrays(GL_TRIANGLES, 0, 36);

	glDepthMask(GL_TRUE);
}

void Skybox::setupSkybox() {

	skyboxTexture = loadCubemap(faces);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);


}