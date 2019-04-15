#include "Actor.h"

#include "HeightMap.h"


Actor::Actor() {
}


Actor::~Actor() {
}

void Actor::update() {
	
}

void Actor::draw() {
	for (int i = 0; i < children.size(); i++) {
		children[i]->draw();
	}
}

void Actor::draw(ShaderProgram * shader) {
	for (int i = 0; i < children.size(); i++) {
		children[i]->draw(shader);
	}
}

void Actor::drawGeometry(ShaderProgram * shader) {
	for (int i = 0; i < children.size(); i++) {
		children[i]->drawGeometry(shader);
	}
}

void Actor::drawWireframe(ShaderProgram * shader) {
	for (int i = 0; i < children.size(); i++) {
		children[i]->drawWireframe(shader);
	}
}

void Actor::addChild(Actor *child) {
	children.push_back(child);
	child->parent = this;
}

void Actor::snapToGround(HeightMap *heightMap) {
	transform.position.y = heightMap->getHeight(transform.position.x, transform.position.z);
}
