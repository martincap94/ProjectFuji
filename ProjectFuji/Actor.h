#pragma once

#include <vector>
#include <string>

#include <glm\glm.hpp>


#include "Transform.h"
#include "ShaderProgram.h"

class HeightMap;

class Actor {
public:

	Transform transform;
	//glm::mat4 modelMatrix;
	std::string name;

	Actor *parent = nullptr;
	std::vector<Actor *> children;

	Actor();
	Actor(std::string name);
	~Actor();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void drawGeometry(ShaderProgram *shader);
	virtual void drawWireframe(ShaderProgram *shader);

	virtual void addChild(Actor *child);

	void snapToGround(HeightMap *heightMap);

protected:


};

