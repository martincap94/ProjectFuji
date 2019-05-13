#include "VariableManager.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>


#include "Config.h"
#include "Utils.h"
//#include "Timer.h"

// use namespace alias as not to type out the whole name
namespace fs = std::experimental::filesystem;

using namespace std;







VariableManager::VariableManager() {
}

VariableManager::~VariableManager() {
	if (heightMap) {
		delete heightMap;
	}
}

void VariableManager::init(int argc, char **argv) {

	loadConfigFile();
	parseArguments(argc, argv);

	loadSceneFilenames();
	loadSoundingDataFilenames();

	ready = true;
}




void VariableManager::loadConfigFile() {

	cout << "Parsing configuration parameters..." << endl;

	ifstream infile(CONFIG_FILE);

	string line;

	while (infile.good()) {

		getline(infile, line);

		// ignore comments
		if (line.find("//") == 0 || line.length() == 0) {
			continue;
		}
		// get rid of comments at the end of the line
		int idx = (int)line.find("//");
		line = line.substr(0, idx);

		// delete whitespace
		trim(line);
		//line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

		idx = (int)line.find(":");

		string param = line.substr(0, idx);
		string val = line.substr(idx + 1, line.length() - 1);
		trim(param);
		trim(val);

		//cout << "param = " << param << ", val = " << val << endl;
		cout <<  "   | " << param << ": " << val << endl;

		saveConfigParam(param, val);

	}
}

void VariableManager::loadSceneFilenames() {

	string path = SCENES_DIR;
	string ext = "";
	for (const auto &entry : fs::directory_iterator(path)) {
		//cout << entry.path() << endl;

		if (getFileExtension(entry.path().string(), ext)) {
			if (ext == "png" || ext == "jpg" || ext == "jpeg") {
				sceneFilenames.push_back(entry.path().string());
			}
		}
	}
	cout << "Possible Scenes:" << endl;
	for (int i = 0; i < sceneFilenames.size(); i++) {
		cout << " | " << sceneFilenames[i] << endl;
	}


}

void VariableManager::loadSoundingDataFilenames() {

	string path = SOUNDING_DATA_DIR;
	string ext = "";
	for (const auto &entry : fs::directory_iterator(path)) {
		//cout << entry.path() << endl;
		if (getFileExtension(entry.path().string(), ext)) {
			if (ext == "txt") {
				soundingDataFilenames.push_back(entry.path().string());
			}
		}
	}
	cout << "Possible Sounding Data Files:" << endl;
	for (int i = 0; i < soundingDataFilenames.size(); i++) {
		cout << " | " << soundingDataFilenames[i] << endl;
	}


}

std::string VariableManager::getFogModeString(int fogMode) {
	switch (fogMode) {
		case LINEAR:
			return "LINEAR";
		case EXPONENTIAL:
			return "EXPONENTIAL";
	}
	return "NONE";
}


void VariableManager::printHelpMessage(string errorMsg) {

	if (errorMsg == "") {
		cout << "Lattice Boltzmann command line argument options:" << endl;
	} else {
		cout << "Incorrect usage of parameter: " << errorMsg << ". Please refer to the options below." << endl;
	}
	cout << " -h, -help, --help:" << endl << "  show this help message" << endl;
	cout << " -t:" << endl << "  LBM type: 2D (or 2) and 3D (or 3)" << endl;
	cout << " -s" << endl << "  scene filename: *.ppm" << endl;
	cout << " -c:" << endl << "   use CUDA: 'true' or 'false'" << endl;
	cout << " -lh: " << endl << "   lattice height (int value)" << endl;
	cout << " -m: " << endl << "   measure times (true or false)" << endl;
	cout << " -p: " << endl << "   number of particles (int value)" << endl;
	cout << " -mavg: " << endl << "   number of measurements for average time" << endl;
	cout << " -mexit: " << endl << "   exit after first average measurement finished (true or false)" << endl;
	cout << " -autoplay, -auto, -a: " << endl << "   start simulation right away (true or false)" << endl;
	cout << " -tau:" << endl << "   value of tau (float between 0.51 and 10.0)" << endl;
	cout << " -sf:" << endl << "  Sounding filename (with extension)" << endl;

}

void VariableManager::parseArguments(int argc, char **argv) {
	if (argc <= 1) {
		return;
	}
	cout << "Parsing command line arguments..." << endl;
	string arg;
	string val;
	string vallw;
	for (int i = 1; i < argc; i++) {
		arg = (string)argv[i];
		if (arg == "-h" || arg == "-help" || arg == "--help") {
			printHelpMessage();
		} else if (arg == "-t") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				if (val == "2D" || val == "2" || val == "3D" || val == "3") {
					saveConfigParam(arg, val);
				} else {
					printHelpMessage("-t");
				}
				i++;
			}
		} else if (arg == "-s") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				saveConfigParam(arg, val);
				i++;
			}
		} else if (arg == "-c") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				transform(val.begin(), val.end(), val.begin(), [](char c) { return tolower(c); });
				if (val == "true" || val == "false") {
					saveConfigParam(arg, val);
				} else {
					printHelpMessage("-c");
				}
				i++;
			}
		} else if (arg == "-m") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				transform(val.begin(), val.end(), val.begin(), [](char c) { return tolower(c); });
				if (val == "true" || val == "false") {
					saveConfigParam(arg, val);
				} else {
					printHelpMessage("-m");
				}
				i++;
			}
		} else if (arg == "-lh") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				saveConfigParam(arg, val);
				i++;
			}
		} else if (arg == "-p") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				saveConfigParam(arg, val);
				i++;
			}
		} else if (arg == "-mavg") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				saveConfigParam(arg, val);
				i++;
			}
		} else if (arg == "-mexit") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				transform(val.begin(), val.end(), val.begin(), [](char c) { return tolower(c); });
				if (val == "true" || val == "false") {
					saveConfigParam(arg, val);
				} else {
					printHelpMessage("-mexit");
				}
				i++;
			}
		} else if (arg == "-autoplay" || arg == "-auto" || arg == "-a") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				transform(val.begin(), val.end(), val.begin(), [](char c) { return tolower(c); });
				if (val == "true" || val == "false") {
					saveConfigParam(arg, "autoplay");
				} else {
					printHelpMessage("-autoplay");
				}
				i++;
			}
		} else if (arg == "-tau") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				saveConfigParam(arg, val);
				i++;
			}
		} else if (arg == "-sf") {
			if (i + 1 < argc) {
				val = argv[i + 1];
				saveConfigParam(arg, val);
				i++;
			}
		}
	}


}


void VariableManager::saveConfigParam(string param, string val) {

	if (param == "VSync") {
		saveIntParam(vsync, val);
	} else if (param == "num_particles" || param == "-p") {
		saveIntParam(numParticles, val);
	} else if (param == "scene_filename" || param == "-s") {
		saveStringParam(sceneFilename, val);
	} else if (param == "window_width") {
		saveIntParam(windowWidth, val);
	} else if (param == "window_height") {
		saveIntParam(windowHeight, val);
	} else if (param == "lattice_width") {
		saveIntParam(latticeWidth, val);
	} else if (param == "lattice_height" || param == "-lh") {
		saveIntParam(latticeHeight, val);
	} else if (param == "lattice_depth") {
		saveIntParam(latticeDepth, val);
	} else if (param == "tau" || param == "-tau") {
		saveFloatParam(tau, val);
	} else if (param == "draw_streamlines") {
		saveBoolParam(drawStreamlines, val);
	} else if (param == "autoplay") {
		saveIntBoolParam(paused, val);
	} else if (param == "camera_speed") {
		saveFloatParam(cameraSpeed, val);
	} else if (param == "block_dim_2D") {
		saveIntParam(blockDim_2D, val);
	} else if (param == "block_dim_3D_x") {
		saveIntParam(blockDim_3D_x, val);
	} else if (param == "block_dim_3D_y") {
		saveIntParam(blockDim_3D_y, val);
	} else if (param == "measure_time" || param == "-m") {
		saveBoolParam(measureTime, val);
	/*} else if (param == "avg_frame_count" || param == "-mavg") {
		timer.numMeasurementsForAvg = stoi(val);
	} else if (param == "log_measurements_to_file") {
		timer.logToFile = (val == "true") ? true : false;
	} else if (param == "print_measurements_to_console") {
		timer.printToConsole = (val == "true") ? true : false;*/
	} else if (param == "exit_after_first_avg" || param == "-mexit") {
		exitAfterFirstAvg = (val == "true") ? true : false;
	} else if (param == "sounding_file" || param == "-sf") {
		soundingFile = val;
	} else if (param == "terrain_x_offset") {
		terrainXOffset = stoi(val);
	} else if (param == "terrain_z_offset") {
		terrainZOffset = stoi(val);
	} else if (param == "lattice_position") {
		saveVec3Param(latticePosition, val);
	} else if (param == "texel_world_size") {
		saveFloatParam(texelWorldSize, val);
	} else if (param == "terrain_height_range") {
		saveVec2Param(terrainHeightRange, val);
	} else if (param == "lattice_scale") {
		saveFloatParam(latticeScale, val);
	} else if (param == "use_monitor_resolution") {
		saveBoolParam(useMonitorResolution, val);
	} else if (param == "fullscreen") {
		saveBoolParam(fullscreen, val);
	} else if (param == "multisampling") {
		saveIntParam(multisamplingAmount, val);
	} else if (param == "startup_particle_file") {
		saveStringParam(startupParticleSaveFile, val);
	} else if (param == "particle_opacity_multiplier") {
		saveFloatParam(opacityMultiplier, val);
	} else if (param == "use_blur_pass") {
		saveBoolParam(volumetricUseBlurPass, val);
	} else if (param == "blur_amount") {
		saveFloatParam(volumetricBlurAmount, val);
	} else if (param == "show_particles_below_CCL") {
		saveBoolParam(showParticlesBelowCCL, val);
	}



}

void VariableManager::saveIntParam(int & target, std::string stringVal) {
	// unsafe
	target = stoi(stringVal);
}

void VariableManager::saveFloatParam(float & target, std::string stringVal) {
	// unsafe
	target = stof(stringVal);
}

void VariableManager::saveVec2Param(glm::vec2 & target, std::string line) {
	size_t idx;
	if (idx = line.find("vec2") == 0) {
		line = line.substr(5, line.length() - 1);
		rtrim(line, ")");
	}

	istringstream iss(line);
	string tmp;
	int counter = 0;
	while (getline(iss, tmp, ',')) {
		target[counter++] = stof(tmp);
		if (counter > 1) {
			break;
		}
	}
}

void VariableManager::saveVec3Param(glm::vec3 & target, std::string line) {
	size_t idx;
	if (idx = line.find("vec3") == 0) {
		line = line.substr(5, line.length() - 1);
		rtrim(line, ")");
	}

	istringstream iss(line);
	string tmp;
	int counter = 0;
	while (getline(iss, tmp, ',')) {
		target[counter++] = stof(tmp);
		if (counter > 2) {
			break;
		}
	}
	/*
	idx = line.find_first_of(" ");

	target.x = stof(line.substr(0, idx));

	string tmp = line.substr(idx + 1, line.length() - 1);
	idx = tmp.find_first_of(" ");

	target.y = stof(tmp.substr(0, idx));
	target.z = stof(tmp.substr(idx + 1, tmp.length() - 1));
	*/

}

void VariableManager::saveVec4Param(glm::vec4 & target, std::string line) {
	size_t idx;
	if (idx = line.find("vec4") == 0) {
		line = line.substr(5, line.length() - 1);
		rtrim(line, ")");
	}

	istringstream iss(line);
	string tmp;
	int counter = 0;
	while (getline(iss, tmp, ',')) {
		target[counter++] = stof(tmp);
		if (counter > 3) {
			break;
		}
	}
}

void VariableManager::saveBoolParam(bool & target, std::string stringVal) {
	transform(stringVal.begin(), stringVal.end(), stringVal.begin(), ::tolower);
	if (stringVal == "true" || stringVal == "t") {
		target = true;
	} else if (stringVal == "false" || stringVal == "f") {
		target = false;
	} else {
		// unsafe - try integer
		int tmp = stoi(stringVal);
		target = (tmp != 0);
	}

}

void VariableManager::saveIntBoolParam(int & target, std::string stringVal) {
	transform(stringVal.begin(), stringVal.end(), stringVal.begin(), ::tolower);
	if (stringVal == "true" || stringVal == "t") {
		target = 1;
	} else if (stringVal == "false" || stringVal == "f") {
		target = 0;
	} else {
		// unsafe - try integer
		int tmp = stoi(stringVal);
		target = tmp;
	}
}

void VariableManager::saveStringParam(string & target, std::string stringVal) {
	target = stringVal;
}
