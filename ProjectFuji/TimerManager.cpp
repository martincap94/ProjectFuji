#include "TimerManager.h"


#include <map>
#include <fstream>
#include "Utils.h"

using namespace std;

namespace TimerManager {


	namespace {

		map<string, Timer *> timers;

		int globalIdx = 0;
		int globalNumMeasurementsForAverage = 100;

		ofstream benchmarkFile;
		int benchmarking = 1;
		int benchmarkFrame = 0;
		
		void startBenchmarking() {
			if (benchmarking) {
				benchmarkFrame = 0;
				benchmarkFile.open(LOG_FILENAME_BASE + getTimeStr() + " BENCHMARK" + ".csv");
				benchmarkFile << "Frame,";
				for (const auto& kv : timers) {
					benchmarkFile << kv.second->name << ",";
				}
				
				benchmarkFile << endl;
			}
		}

	}


	void init() {

	}

	void tearDown() {
		for (const auto& kv : timers) {
			delete kv.second;
		}
		benchmarkFile.close();
	}

	Timer * createTimer(std::string name, bool callsGLFinish, bool callsCudaDeviceSynchronize, bool logToFile, bool printToConsole, int numMeasurementsForAvg) {
		timers.insert(make_pair(name, new Timer(name, callsGLFinish, callsCudaDeviceSynchronize, logToFile, printToConsole, numMeasurementsForAvg)));
		timers[name]->index = globalIdx++;
		return timers[name];
	}

	Timer * getTimer(std::string name) {
		if (timers.count(name) > 0) {
			return timers[name];
		}
		return nullptr;
	}

	void startAllTimers() {
		for (const auto& kv : timers) {
			kv.second->start();
		}
		startBenchmarking();
	}

	void resetAllTimers() {
		for (const auto& kv : timers) {
			kv.second->reset();
		}
		startBenchmarking();
	}

	void endAllTimers() {
		for (const auto& kv : timers) {
			kv.second->end();
		}
	}

	void writeToBenchmarkFile() {
		if (benchmarking) {
			benchmarkFrame++;
			benchmarkFile << benchmarkFrame << ",";
			for (const auto& kv : timers) {
				benchmarkFile << kv.second->frameTime << ",";
			}
			benchmarkFile << endl;
		}

	}

	void constructTimersUITab(nk_context * ctx, UserInterface * ui) {
		//if (timers.empty()) {
		//	return;
		//}
		ui->nk_label_header(ctx, "Timers", true);

		if (nk_button_label(ctx, "Start All")) {
			startAllTimers();
		}
		if (nk_button_label(ctx, "Reset All")) {
			resetAllTimers();
		}
		if (nk_button_label(ctx, "End All")) {
			endAllTimers();
		}
		nk_checkbox_label(ctx, "Benchmarking Enabled", &benchmarking);

		nk_property_int(ctx, "Num Measurement for Avg", 1, &globalNumMeasurementsForAverage, 100000, 1, 0.2f);
		if (nk_button_label(ctx, "Set Global Num Measurements for Avg")) {
			for (const auto& kv : timers) {
				kv.second->numMeasurementsForAvg = globalNumMeasurementsForAverage;
			}
		}


		nk_layout_row_dynamic(ctx, 45.0f, 1);
		nk_label_colored_wrap(ctx, "Timers that synchronize the GPU may have negative impact on the performance of the whole application.", nk_rgba(240, 170, 170, 255));
		nk_layout_row_dynamic(ctx, 15.0f, 1);
		for (const auto& kv : timers) {
			kv.second->constructUITab(ctx, ui);
		}

	}

}