#include "Actor.h"

#include "HeightMap.h"


Actor::Actor() {
}


Actor::~Actor() {
}

void Actor::snapToGround(HeightMap *heightMap) {
	transform.position.y = heightMap->getHeight(transform.position.x, transform.position.z);
}
