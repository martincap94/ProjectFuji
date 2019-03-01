#include "VariableManager.h"

#include <fstream>
#include <iostream>
#include <algorithm>


#include "Config.h"
#include "Utils.h"
//#include "Timer.h"

using namespace std;







VariableManager::VariableManager() {
}

VariableManager::~VariableManager() {
}

void VariableManager::init(int argc, char **argv) {

	loadConfigFile();
	parseArguments(argc, argv);

	ready = true;
}




void VariableManager::loadConfigFile() {

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
		cout << param << ": " << val << endl;

		saveConfigParam(param, val);

	}
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

	if (param == "LBM_type" || param == "-t") {
		if (val == "2D" || val == "2") {
			VariableManager::lbmType = LBM2D;
		} else if (val == "3D" || val == "3") {
			VariableManager::lbmType = LBM3D;
		}
	} else if (param == "VSync") {
		VariableManager::vsync = stoi(val);
	} else if (param == "num_particles" || param == "-p") {
		VariableManager::numParticles = stoi(val);
	} else if (param == "scene_filename" || param == "-s") {
		VariableManager::sceneFilename = val;
	} else if (param == "window_width") {
		VariableManager::windowWidth = stoi(val);
	} else if (param == "window_height") {
		VariableManager::windowHeight = stoi(val);
	} else if (param == "lattice_width") {
		VariableManager::latticeWidth = stoi(val);
	} else if (param == "lattice_height" || param == "-lh") {
		VariableManager::latticeHeight = stoi(val);
	} else if (param == "lattice_depth") {
		VariableManager::latticeDepth = stoi(val);
	} else if (param == "use_CUDA" || param == "-c") {
		VariableManager::useCUDA = (val == "true") ? true : false;
		VariableManager::useCUDACheckbox = (int)VariableManager::useCUDA;
	} else if (param == "tau" || param == "-tau") {
		VariableManager::tau = stof(val);
	} else if (param == "draw_streamlines") {
		VariableManager::drawStreamlines = (val == "true") ? true : false;
	} else if (param == "autoplay") {
		VariableManager::paused = (val == "true") ? 0 : 1;
	} else if (param == "camera_speed") {
		VariableManager::cameraSpeed = stof(val);
	} else if (param == "block_dim_2D") {
		VariableManager::blockDim_2D = stoi(val);
	} else if (param == "block_dim_3D_x") {
		VariableManager::blockDim_3D_x = stoi(val);
	} else if (param == "block_dim_3D_y") {
		VariableManager::blockDim_3D_y = stoi(val);
	} else if (param == "measure_time" || param == "-m") {
		VariableManager::measureTime = (val == "true") ? true : false;
	} else if (param == "avg_frame_count" || param == "-mavg") {
		//avgFrameCount = stoi(val);
		VariableManager::timer.numMeasurementsForAvg = stoi(val);
	} else if (param == "log_measurements_to_file") {
		VariableManager::timer.logToFile = (val == "true") ? true : false;
	} else if (param == "print_measurements_to_console") {
		VariableManager::timer.printToConsole = (val == "true") ? true : false;
	} else if (param == "exit_after_first_avg" || param == "-mexit") {
		VariableManager::exitAfterFirstAvg = (val == "true") ? true : false;
	} else if (param == "sounding_file" || param == "-sf") {
		VariableManager::soundingFile = val;
	}
}