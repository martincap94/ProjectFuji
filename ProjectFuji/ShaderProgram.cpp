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


	vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsArr, nullptr);
	glCompileShader(vs);

	glGetShaderiv(vs, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED" << endl;
		GLint logLen;
		glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char *log = (char *)malloc(logLen);
			GLsizei written;
			glGetShaderInfoLog(vs, logLen, &written, log);
			cerr << "Shader Log: " << endl << log << endl;
			free(log);
		}
	}

	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsArr, nullptr);
	glCompileShader(fs);

	glGetShaderiv(fs, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED" << endl;
		GLint logLen;
		glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char *log = (char *)malloc(logLen);
			GLsizei written;
			glGetShaderInfoLog(vs, logLen, &written, log);
			cerr << "Shader Log: " << endl << log << endl;
			free(log);
		}
	}

	id = glCreateProgram();
	glAttachShader(id, vs);
	glAttachShader(id, fs);
	glLinkProgram(id);

	glGetProgramiv(id, GL_LINK_STATUS, &result);
	if (result == GL_FALSE) {
		cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED" << endl;
		GLint logLen;
		glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char *log = (char *)malloc(logLen);
			GLsizei written;
			glGetProgramInfoLog(vs, logLen, &written, log);
			cerr << "Shader Log: " << endl << log << endl;
			free(log);
		}
	}

	glDeleteShader(vs);
	glDeleteShader(fs);


}


ShaderProgram::~ShaderProgram() {
}

void ShaderProgram::setMat4fv(const string &name, glm::mat4 value) const {
	glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}