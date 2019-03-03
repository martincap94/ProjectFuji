#include "ppmImage.h"

#include <iostream>
#include <fstream>

using namespace std;

ppmImage::ppmImage(std::string filename) {

	if (filename.find(".ppm") == string::npos) {
		cerr << "ppmImage only accepts .ppm files!" << endl;
		return;
	}
	ifstream inFile(filename);
	string line;

	getline(inFile, line);
	if (line != "P3") {
		cerr << "ppmImage requires .ppm files in P3 ASCII format!" << endl;
		return;
	}

	getline(inFile, line);

	inFile >> width;
	inFile >> height;
	inFile >> maxIntensity;
	cout << "Width = " << width << ", height = " << height << ", max intesity = " << maxIntensity << endl;



	data = new glm::vec3*[width]();
	for (int i = 0; i < width; i++) {
		data[i] = new glm::vec3[height]();
	}

	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			inFile >> data[x][y].x;
			inFile >> data[x][y].y;
			inFile >> data[x][y].z;
		}
	}


}

ppmImage::~ppmImage() {
	for (int i = 0; i < width; i++) {
		delete[] data[i];
	}
	delete[] data;
}
