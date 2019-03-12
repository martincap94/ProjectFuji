#include "ShaderProgram.h"

#include <fstream>
#include <sstream>
#include <iostream>

ShaderProgram::ShaderProgram() {




}

ShaderProgram::ShaderProgram(const GLchar *vsPath, const GLchar *fsPath) {

	string vsCode;
	string fsCode;
	ifstream vsFile(SHADERS_DIR + string(vsPath));
	ifstream fsFile(SHADERS_DIR + string(fsPath));

	stringstream vsStream;
	stringstream fsStream;

	vsStream << vsFile.rdbuf();
	fsStream << fsFile.rdbuf();

	vsCode = vsStream.str();
	fsCode = fsStream.str();

	vsFile.close();
	fsFile.close();

	const GLchar *vsArr = vsCode.c_str();
	const GLchar *fsArr = fsCode.c_str();

	GLuint vs;
	GLuint fs;
	GLint result;

	char infoLog[1024];


	vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsArr, nullptr);
	glCompileShader(vs);

	glGetShaderiv(vs, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED" << endl;
		glGetShaderInfoLog(vs, 1024, NULL, infoLog);
		cout << infoLog << endl;

		/*GLint logLen;
		glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char *log = (char *)malloc(logLen);
			GLsizei written;
			glGetShaderInfoLog(vs, logLen, &written, log);
			cerr << "Shader Log: " << endl << log << endl;
			free(log);
		}*/
	}

	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsArr, nullptr);
	glCompileShader(fs);

	glGetShaderiv(fs, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED" << endl;
		glGetShaderInfoLog(fs, 1024, NULL, infoLog);
		cout << infoLog << endl;
		/*GLint logLen;
		glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char *log = (char *)malloc(logLen);
			GLsizei written;
			glGetShaderInfoLog(vs, logLen, &written, log);
			cerr << "Shader Log: " << endl << log << endl;
			free(log);
		}*/
	}

	id = glCreateProgram();
	glAttachShader(id, vs);
	glAttachShader(id, fs);
	glLinkProgram(id);

	glGetProgramiv(id, GL_LINK_STATUS, &result);
	if (result == GL_FALSE) {
		cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED" << endl;
		glGetProgramInfoLog(id, 1024, NULL, infoLog);
		cout << infoLog << endl;
		/*GLint logLen;
		glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logLen);
		cout << "Log length = " << logLen << endl;
		if (logLen > 0) {
			char *log = (char *)malloc(logLen);
			GLsizei written;
			glGetProgramInfoLog(vs, logLen, &written, log);
			cerr << "Shader Log: " << endl << log << endl;
			free(log);
		}*/
	}

	glDeleteShader(vs);
	glDeleteShader(fs);


}


ShaderProgram::~ShaderProgram() {
}

void ShaderProgram::use() {
	glUseProgram(id);
}

/*

Important note from: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glUniform.xhtml
If location is a value other than -1 and it does not represent a valid uniform variable location in the current program object, an error will be generated, and no changes will be made to the uniform variable storage of the current program object. If location is equal to -1, the data passed in will be silently ignored and the specified uniform variable will not be changed.
*/

void ShaderProgram::setBool(const std::string &name, bool value) const {
	glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}

void ShaderProgram::setInt(const std::string &name, int value) const {
	glUniform1i(glGetUniformLocation(id, name.c_str()), value);
}

void ShaderProgram::setFloat(const std::string &name, float value) const {
	glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}

void ShaderProgram::setMat4fv(const string &name, glm::mat4 value) const {
	glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}


void ShaderProgram::setVec3(const std::string &name, float x, float y, float z) const {
	glUniform3f(glGetUniformLocation(id, name.c_str()), x, y, z);
}

void ShaderProgram::setVec3(const std::string &name, glm::vec3 value) const {
	glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}

void ShaderProgram::setVec4(const std::string &name, glm::vec4 value) const {
	glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}


void ShaderProgram::setProjectionMatrix(glm::mat4 projectionMatrix, string uniformName) {
	setMat4fv(uniformName, projectionMatrix);
}

void ShaderProgram::setViewMatrix(glm::mat4 viewMatrix, string uniformName) {
	setMat4fv(uniformName, viewMatrix);
}

void ShaderProgram::setModelMatrix(glm::mat4 modelMatrix, string uniformName) {
	setMat4fv(uniformName, modelMatrix);
}

void ShaderProgram::setupMaterialUniforms(bool useShader) {
	if (matType == PHONG) {
		if (useShader) {
			use();
		}
		setInt("u_Material.diffuse", 0);
		setInt("u_Material.specular", 1);
		setInt("u_Material.normalMap", 2);
		setFloat("u_Material.shininess", 4.0f);
	}
}

void ShaderProgram::setFogProperties(float fogIntensity, float fogMinDistance, float fogMaxDistance, glm::vec4 fogColor, bool useShader) {
	if (useShader) {
		use();
	}
	setFloat("u_Fog.intensity", fogIntensity);
	setFloat("u_Fog.minDistance", fogMinDistance);
	setFloat("u_Fog.maxDistance", fogMaxDistance);
	setVec4("u_Fog.color", fogColor);

}


//void ShaderProgram::setPointLightAttributes(int lightNum, PointLight &pointLight) {
//	std::string pointLightName = "pointLights[" + std::to_string(lightNum) + "]";
//
//	//setVec3(pointLightName + ".position", pointLight.transform.position);
//	//std::cout << "(" << pointLight.getWorldPosition().x << ", " << pointLight.getWorldPosition().y << ", " << pointLight.getWorldPosition().z << ")" << std::endl;
//	setVec3(pointLightName + ".position", pointLight.getWorldPosition());
//
//	setVec3(pointLightName + ".ambient", pointLight.ambientColor);
//	setVec3(pointLightName + ".diffuse", pointLight.diffuseColor);
//	setVec3(pointLightName + ".specular", pointLight.specularColor);
//	setFloat(pointLightName + ".constant", pointLight.attenuationConstant);
//	setFloat(pointLightName + ".linear", pointLight.attenuationLinear);
//	setFloat(pointLightName + ".quadratic", pointLight.attenuationQuadratic);
//}


void ShaderProgram::updateDirectionalLightUniforms(DirectionalLight &dirLight) {
	if (lightingType == LIT) {
		use();
		setVec3("u_DirLight.direction", dirLight.getDirection());
		setVec3("u_DirLight.color", dirLight.color);
		setFloat("u_DirLight.intensity", dirLight.intensity);
	}
}

//void ShaderProgram::setSpotlightAttributes(Spotlight &spotlight, Camera &camera, bool spotlightOn) {
//	if (spotlightOn) {
//		setVec3("spotLight.ambient", spotlight.ambientColor);
//		setVec3("spotLight.diffuse", spotlight.diffuseColor);
//		setVec3("spotLight.specular", spotlight.specularColor);
//		setFloat("spotLight.constant", spotlight.attenuationConstant);
//		setFloat("spotLight.linear", spotlight.attenuationLinear);
//		setFloat("spotLight.quadratic", spotlight.attenuationQuadratic);
//		setVec3("spotLight.position", camera.Position);
//		setVec3("spotLight.direction", camera.Front);
//		setFloat("spotLight.cutoff", glm::cos(glm::radians(spotlight.cutoff)));
//		setFloat("spotLight.outerCutoff", glm::cos(glm::radians(spotlight.outerCutoff)));
//	} else {
//		setVec3("spotLight.ambient", glm::vec3(0.0f));
//		setVec3("spotLight.diffuse", glm::vec3(0.0f));
//		setVec3("spotLight.specular", glm::vec3(0.0f));
//	}
//}