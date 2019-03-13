#include "InstancedModel.h"




InstancedModel::InstancedModel(const char * path) : Model(path) {
}

InstancedModel::InstancedModel(const char * path, Material * material, ShaderProgram * shader) : Model(path, material, shader) {
}

InstancedModel::~InstancedModel() {
}

void InstancedModel::draw() {
}

void InstancedModel::draw(ShaderProgram * shader) {
}
