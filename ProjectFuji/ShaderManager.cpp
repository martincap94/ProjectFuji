#include "ShaderManager.h"

#include <iostream>
#include <map>

using namespace std;


namespace ShaderManager {



	namespace {

		bool initFlag = false;
		map<string, ShaderProgram *> shadersStr;
		map<GLuint, ShaderProgram *> shaders;
		map<GLuint, string> shaderIdName;
		VariableManager *vars;

		void addShader(string sName, string vertShader, string fragShader, ShaderProgram::eLightingType lightingType, ShaderProgram::eMaterialType matType) {
			ShaderProgram *sPtr = new ShaderProgram(vertShader.c_str(), fragShader.c_str());
			sPtr->lightingType = lightingType;
			sPtr->matType = matType;
			addShader(sPtr, sName, sPtr->id);
		}

		void addShader(ShaderProgram *sPtr, string sName, GLuint sId) {
			shadersStr.insert(make_pair(sName, sPtr));
			shaders.insert(make_pair(sId, sPtr));
			shaderIdName.insert(make_pair(sId, sName));
		}

		void initShaders() {
			for (const auto& kv : shaders) {
				kv.second->use();
				kv.second->setupMaterialUniforms(false);

				// fog testing
				if (vars) {
					kv.second->setFogProperties(vars->fogIntensity, vars->fogMinDistance, vars->fogMaxDistance, vars->fogColor);
				} else {
					cout << "oopsie" << endl;
					kv.second->use();
					kv.second->setVec4("u_Fog.color", glm::vec4(0.1f, 0.1f, 0.1f, 1.0f));
					kv.second->setFloat("u_Fog.minDistance", 20.0f);
				}
			}
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
			addShader("singleColorModel", "singleColorModel.vert", "singleColor.frag");

			addShader("singleColorAlpha", "singleColor.vert", "singleColor_alpha.frag");
			addShader("singleColor_VBO", "singleColor_VBO.vert", "singleColor_VBO.frag");
			addShader("unlitColor", "unlitColor.vert", "unlitColor.frag");
			addShader("dirLightOnly", "dirLightOnly.vert", "dirLightOnly.frag", ShaderProgram::LIT, ShaderProgram::PHONG);
			addShader("pointSpriteTest", "pointSpriteTest.vert", "pointSpriteTest.frag");
			addShader("coloredParticle", "coloredParticle.vert", "coloredParticle.frag");
			addShader("diagram", "diagram.vert", "diagram.frag");
			addShader("text", "text.vert", "text.frag");
			addShader("curve", "curve.vert", "curve.frag");
			addShader("skybox", "skybox.vert", "skybox.frag");
			addShader("gaussianBlur", "basic_blur.vert", "gblur_9x9_separated.frag");
			addShader("evsm_1st_pass", "evsm_1st_pass.vert", "evsm_1st_pass.frag");
			addShader("evsm_2nd_pass", "evsm_2nd_pass.vert", "evsm_2nd_pass.frag", ShaderProgram::LIT, ShaderProgram::PHONG);

			addShader("dirLightOnly_evsm", "dirLightOnly_evsm.vert", "dirLightOnly_evsm.frag", ShaderProgram::LIT, ShaderProgram::PHONG);

			addShader("vsm_1st_pass", "vsm_1st_pass.vert", "vsm_1st_pass.frag");
			addShader("vsm_2nd_pass", "vsm_2nd_pass.vert", "vsm_2nd_pass.frag");

			addShader("shadow_mapping_1st_pass", "shadow_mapping_1st_pass.vert", "shadow_mapping_1st_pass.frag");
			addShader("shadow_mapping_2nd_pass", "shadow_mapping_2nd_pass.vert", "shadow_mapping_2nd_pass.frag");

			addShader("normals", "normals.vert", "normals.frag", ShaderProgram::LIT, ShaderProgram::PHONG);

			addShader("terrain", "terrain.vert", "terrain.frag", ShaderProgram::LIT, ShaderProgram::PHONG);

			addShader("normals_instanced", "normals_instanced.vert", "normals_instanced.frag", ShaderProgram::LIT, ShaderProgram::PHONG);

		}
	}


	bool init(VariableManager *vars) {
		ShaderManager::vars = vars; // not very nice
		if (initFlag) {
			cout << "ShaderManager has already been initialized!" << endl;
			return false;
		}
		cout << "Initializing ShaderManager" << endl;
		loadShaders();
		initShaders();

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

	void updateDirectionalLightUniforms(DirectionalLight &dirLight) {
		for (const auto& kv : shaders) {
			kv.second->updateDirectionalLightUniforms(dirLight);
		}
	}

	void updateViewPositionUniforms(glm::vec3 viewPos) {
		for (const auto& kv : shaders) {
			kv.second->use();
			kv.second->setVec3("u_ViewPos", viewPos);
		}
	}

	void updateFogUniforms() {
		if (!vars) {
			return;
		}
		for (const auto &kv : shaders) {
			kv.second->setFogProperties(vars->fogIntensity, vars->fogMinDistance, vars->fogMaxDistance, vars->fogColor);
		}
	}





}