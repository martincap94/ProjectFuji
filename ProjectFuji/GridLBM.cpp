#include "GridLBM.h"

#include "LBM3D_1D_indices.h"
#include "ShaderProgram.h"
#include "ShaderManager.h"

using namespace std;

GridLBM::GridLBM(LBM3D_1D_indices *owner, glm::vec3 boxColor, glm::vec3 stepSize) : lbm(owner), boxColor(boxColor), stepSize(stepSize) {

	shader = ShaderManager::getShaderPtr("singleColorModel");

	// use model matrix instead
	//float bw = lbm->getWorldWidth();
	//float bh = lbm->getWorldHeight();
	//float bd = lbm->getWorldDepth();
	int bw = lbm->latticeWidth;
	int bh = lbm->latticeHeight;
	int bd = lbm->latticeDepth;

	vector<glm::vec3> bData;


	bData.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
	bData.push_back(glm::vec3(bw, 0.0f, 0.0f));

	bData.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
	bData.push_back(glm::vec3(0.0f, bh, 0.0f));

	bData.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
	bData.push_back(glm::vec3(0.0f, 0.0f, bd));

	bData.push_back(glm::vec3(0.0f, 0.0f, bd));
	bData.push_back(glm::vec3(bw, 0.0f, bd));

	bData.push_back(glm::vec3(bw, 0.0f, bd));
	bData.push_back(glm::vec3(bw, 0.0f, 0.0f));

	bData.push_back(glm::vec3(bw, 0.0f, 0.0f));
	bData.push_back(glm::vec3(bw, bh, 0.0f));

	bData.push_back(glm::vec3(bw, bh, bd));
	bData.push_back(glm::vec3(bw, bh, 0.0f));

	bData.push_back(glm::vec3(bw, 0.0f, bd));
	bData.push_back(glm::vec3(bw, bh, bd));

	bData.push_back(glm::vec3(0.0f, bh, 0.0f));
	bData.push_back(glm::vec3(0.0f, bh, bd));

	bData.push_back(glm::vec3(0.0f, bh, 0.0f));
	bData.push_back(glm::vec3(bw, bh, 0.0f));

	bData.push_back(glm::vec3(0.0f, bh, bd));
	bData.push_back(glm::vec3(bw, bh, bd));

	bData.push_back(glm::vec3(0.0f, 0.0f, bd));
	bData.push_back(glm::vec3(0.0f, bh, bd));

	// use model matrix instead
	//for (int i = 0; i < bData.size(); i++) {
	//	bData[i] += lbm->position;
	//}


	glGenVertexArrays(1, &boxVAO);
	glBindVertexArray(boxVAO);
	glGenBuffers(1, &boxVBO);
	glBindBuffer(GL_ARRAY_BUFFER, boxVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bData.size(), &bData[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);

}

GridLBM::~GridLBM() {
}

void GridLBM::draw() {
	draw(lbm->getModelMatrix());
}

void GridLBM::draw(glm::mat4 modelMatrix) {
	shader->use();

	shader->setColor(boxColor);
	shader->setModelMatrix(modelMatrix);

	glBindVertexArray(boxVAO);
	glDrawArrays(GL_LINES, 0, 24);
}
