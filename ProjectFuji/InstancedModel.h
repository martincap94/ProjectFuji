#pragma once
#include "Model.h"

#include "Transform.h"


class InstancedModel : public Model {
public:

	int numInstances;
	vector<Transform> instanceTransforms;

	InstancedModel(const char *path);
	InstancedModel(const char *path, Material *material, ShaderProgram *shader = nullptr);
	~InstancedModel();
	
	virtual void draw();
	virtual void draw(ShaderProgram *shader);

};

