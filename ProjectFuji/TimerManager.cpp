#include "TimerManager.h"


#include <map>


using namespace std;

namespace TimerManager {


	namespace {

		map<string, Timer *> timers;


	}


	void tearDown() {
		for (const auto& kv : timers) {
			delete kv.second;
		}
	}

	Timer * createTimer(std::string name, bool callsGLFinish, bool callsCudaDeviceSynchronize, bool logToFile, bool printToConsole, int numMeasurementsForAvg) {
		timers.insert(make_pair(name, new Timer(name, callsGLFinish, callsCudaDeviceSynchronize, logToFile, printToConsole, numMeasurementsForAvg)));
		return timers[name];
	}

	Timer * getTimer(std::string name) {
		if (timers.count(name) > 0) {
			return timers[name];
		}
		return nullptr;
	}

	void constructTimersUITab(nk_context * ctx, UserInterface * ui) {
		//if (timers.empty()) {
		//	return;
		//}
		ui->nk_label_header(ctx, "Timers", false);

		for (const auto& kv : timers) {
			kv.second->constructUITab(ctx, ui);
		}

	}

}