#include "Actor.h"

#include "HeightMap.h"
#include <algorithm>


Actor::Actor() {
	transform.setOwner(this);
}

Actor::Actor(std::string name) : name(name) {
	transform.setOwner(this);
}


Actor::~Actor() {
}

void Actor::update() {
	transform.updateModelMatrix();
	for (int i = 0; i < children.size(); i++) {
		children[i]->update();
	}
}

bool Actor::draw() {
	if (!shouldDraw()) {
		return false;
	}
	for (int i = 0; i < children.size(); i++) {
		children[i]->draw();
	}
	return true;
}

bool Actor::draw(ShaderProgram * shader) {
	if (!shouldDraw()) {
		return false;
	}
	for (int i = 0; i < children.size(); i++) {
		children[i]->draw(shader);
	}
	return true;
}

bool Actor::drawGeometry(ShaderProgram * shader) {
	if (!shouldDraw()) {
		return false;
	}
	for (int i = 0; i < children.size(); i++) {
		children[i]->drawGeometry(shader);
	}
	return true;
}

bool Actor::drawShadows(ShaderProgram * shader) {
	if (!(shouldDraw() && castShadows)) {
		return false;
	}
	for (int i = 0; i < children.size(); i++) {
		children[i]->drawShadows(shader);
	}
	return true;
}


bool Actor::drawWireframe(ShaderProgram * shader) {
	if (!shouldDraw()) {
		return false;
	}
	for (int i = 0; i < children.size(); i++) {
		children[i]->drawWireframe(shader);
	}
	return true;
}

void Actor::addChild(Actor *child) {
	children.push_back(child);
	child->parent = this;
	child->pidx = children.size() - 1;

}

void Actor::unparent(bool keepWorldPosition) {
	if (isRootChild()) {
		cerr << "Cannot unparent since the current parent is root." << endl;
		return;
	}

	transform.unparent(keepWorldPosition);
	
	if (pidx == -1) {
		parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), this), parent->children.end());
	} else {
		parent->children.erase(parent->children.begin() + pidx);
	}
	
	parent = parent->parent;
	parent->addChild(this);



}

bool Actor::isRootChild() {
	return (parent->parent == nullptr);
}

void Actor::snapToGround(HeightMap *heightMap) {
	transform.position.y = heightMap->getHeight(transform.position.x, transform.position.z);
}

bool Actor::shouldDraw() {
	return visible;
}
