#include "Timer.h"




Timer::Timer(bool logToFile, bool printToConsole, int numMeasurementsForAvg) 
	: logToFile(logToFile), printToConsole(printToConsole), numMeasurementsForAvg(numMeasurementsForAvg) {
}


Timer::~Timer() {
	if (logToFile) {
		logFile.close();
	}
}

void Timer::start() {
	if (running) {
		return;
	}
	if (logToFile && !logFile.is_open()) {
		logFile.open(LOG_FILENAME_BASE + configString + ".txt");
		logFile << "number of measurements for avg = " << numMeasurementsForAvg << endl;
	}
	startTime = chrono::high_resolution_clock::now();
	accumulatedTime = 0.0;
	avgTime = 0.0;
	measurementCount = 0;
	running = true;
}

void Timer::clockAvgStart() {
	if (!running) {
		cerr << "The timer is not running, cannot use clock()!" << endl;
		return;
	}
	lastClockTime = chrono::high_resolution_clock::now();
}

bool Timer::clockAvgEnd() {
	if (!running) {
		cerr << "The timer is not running, cannot use clock()!" << endl;
		return false;
	}
	auto endTime = chrono::high_resolution_clock::now();
	double duration = chrono::duration<double, milli>(endTime - lastClockTime).count();
	accumulatedTime += duration;
	measurementCount++;
	if (measurementCount >= numMeasurementsForAvg) {
		avgTime = accumulatedTime / (double)numMeasurementsForAvg;
		if (logToFile) {
			logFile << avgTime << endl;
		}
		if (printToConsole) {
			cout << "Avg. time for " << numMeasurementsForAvg << " measurements is " << avgTime << " [ms]." << endl;
		}
		measurementCount = 0;
		accumulatedTime = 0.0;
		return true;
	}
	return false;
}

void Timer::end(bool printResults) {
	if (!running) {
		cerr << "The timer is not running, cannot use end()!" << endl;
		return;
	}
	accumulatedTime = 0.0;
	avgTime = 0.0;
	measurementCount = 0;
	running = false;
	if (printResults) {
		auto endTime = chrono::high_resolution_clock::now();
		double duration = chrono::duration<double, milli>(endTime - startTime).count();
		if (logToFile) {
			logFile << duration;
		}
		if (printToConsole) {
			cout << "Total (start to end) duration = " << duration << " [ms]." << endl;
		}
	}
}

