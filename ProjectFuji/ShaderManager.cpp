#include "ShaderManager.h"

#include <iostream>
#include <map>
#include "ShaderProgram.h"

using namespace std;

namespace {
	
	bool initFlag = false;
	map<string, ShaderProgram *> shadersStr;
	map<GLuint, ShaderProgram *> shaders;
	map<GLuint, string> shaderIdName;

	void addShader(string sName, string vertShader, string fragShader) {
		ShaderProgram *sPtr = new ShaderProgram(vertShader.c_str(), fragShader.c_str());
		addShader(sPtr, sName, sPtr->id);
	}

	void addShader(ShaderProgram *sPtr, string sName, GLuint sId) {
		shadersStr.insert(make_pair(sName, sPtr));
		shaders.insert(make_pair(sId, sPtr));
		shaderIdName.insert(make_pair(sId, sName));
	}


	void loadShaders() {
		cout << "Loading shaders" << endl;

		//singleColorShader = ShaderManager::getShaderPtr("singleColor");
		//singleColorShaderAlpha = new ShaderProgram("singleColor.vert", "singleColor_alpha.frag");
		//singleColorShaderVBO = new ShaderProgram("singleColor_VBO.vert", "singleColor_VBO.frag");

		//unlitColorShader = new ShaderProgram("unlitColor.vert", "unlitColor.frag");
		//dirLightOnlyShader = new ShaderProgram("dirLightOnly.vert", "dirLightOnly.frag");
		//pointSpriteTestShader = new ShaderProgram("pointSpriteTest.vert", "pointSpriteTest.frag");
		//coloredParticleShader = new ShaderProgram("coloredParticle.vert", "coloredParticle.frag");
		//diagramShader = new ShaderProgram("diagram.vert", "diagram.frag");

		//textShader = new ShaderProgram("text.vert", "text.frag");
		//curveShader = new ShaderProgram("curve.vert", "curve.frag");

		addShader("singleColor", "singleColor.vert", "singleColor.frag");
		addShader("singleColorAlpha", "singleColor.vert", "singleColor_alpha.frag");
		addShader("singleColor_VBO", "singleColor_VBO.vert", "singleColor_VBO.frag");
		addShader("unlitColor", "unlitColor.vert", "unlitColor.frag");
		addShader("dirLightOnly", "dirLightOnly.vert", "dirLightOnly.frag");
		addShader("pointSpriteTest", "pointSpriteTest.vert", "pointSpriteTest.frag");
		addShader("coloredParticle", "coloredParticle.vert", "coloredParticle.frag");
		addShader("diagram", "diagram.vert", "diagram.frag");
		addShader("text", "text.vert", "text.frag");
		addShader("curve", "curve.vert", "curve.frag");
		addShader("skybox", "skybox.vert", "skybox.frag");

	}

}


namespace ShaderManager {

	bool init() {
		if (initFlag) {
			cout << "ShaderManager has already been initialized!" << endl;
			return false;
		}
		cout << "Initializing ShaderManager" << endl;
		loadShaders();


		initFlag = true;
		return true;
	}

	bool tearDown() {
		for (const auto& kv : shadersStr) {
			delete kv.second;
		}
		return true;
	}

	ShaderProgram *getShaderPtr(std::string shaderName) {
		if (shadersStr.count(shaderName) == 0) {
			cout << "No shader with name " << shaderName << " found!" << endl;
			return nullptr;
		}
		return shadersStr[shaderName];
	}

	ShaderProgram *getShaderPtr(GLuint shaderId) {
		if (shaders.count(shaderId) == 0) {
			cout << "No shader with id" << shaderId << " found!" << endl;
			return nullptr;
		}
		return shaders[shaderId];
	}

	GLuint getShaderId(std::string shaderName) {
		ShaderProgram *ptr = getShaderPtr(shaderName);
		return (ptr == nullptr) ? 0 : ptr->id;
	}


	std::string getShaderName(GLuint shaderId) {
		if (shaders.count(shaderId) == 0) {
			cout << "No shader with id" << shaderId << " found!" << endl;
			return nullptr;
		}
		return shaderIdName[shaderId];
	}

	void updatePVMMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix, glm::mat4 modelMatrix) {
		for (const auto& kv : shaders) {
			kv.second->use();
			kv.second->setMat4fv("u_Projection", projectionMatrix);
			kv.second->setMat4fv("u_View", viewMatrix);
			kv.second->setMat4fv("u_Model", modelMatrix);
		}
	}

	void updatePVMatrixUniforms(glm::mat4 projectionMatrix, glm::mat4 viewMatrix) {
		for (const auto& kv : shaders) {
			kv.second->use();
			kv.second->setMat4fv("u_Projection", projectionMatrix);
			kv.second->setMat4fv("u_View", viewMatrix);
		}
	}

	void updateProjectionMatrixUniforms(glm::mat4 projectionMatrix) {
		for (const auto& kv : shaders) {
			kv.second->use();
			kv.second->setMat4fv("u_Projection", projectionMatrix);
		}
	}

	void updateViewMatrixUniforms(glm::mat4 viewMatrix) {
		for (const auto& kv : shaders) {
			kv.second->use();
			kv.second->setMat4fv("u_View", viewMatrix);
		}
	}

	void updateModelMatrixUniforms(glm::mat4 modelMatrix) {
		for (const auto& kv : shaders) {
			kv.second->use();
			kv.second->setMat4fv("u_Model", modelMatrix);
		}
	}



}