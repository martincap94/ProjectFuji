#pragma once

#include "Timer.h"

#include "UserInterface.h"
#include <nuklear.h>

namespace TimerManager {
	
	namespace {
		void startBenchmarking();
	}

	void init();
	void tearDown();

	Timer *createTimer(std::string name, bool callsGLFinish = false, bool callsCudaDeviceSynchronize = false, bool logToFile = true, bool printToConsole = false, int numMeasurementsForAvg = 1000);

	Timer *getTimer(std::string name);

	void startAllTimers();
	void resetAllTimers();
	void endAllTimers();

	void writeToBenchmarkFile();

	void constructTimersUITab(struct nk_context *ctx, UserInterface *ui);


}

