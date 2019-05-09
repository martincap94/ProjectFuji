#include "Timer.h"

#include <glm\glm.hpp>

#include <glad\glad.h>

Timer::Timer(string name, bool callsGLFinish, bool callsCudaDeviceSynchronizebool, 
			 bool logToFile, bool printToConsole, int numMeasurementsForAvg)
	: name(name), callsGLFinish(callsGLFinish), callsCudaDeviceSynchronize(callsCudaDeviceSynchronize),
	logToFile(logToFile), printToConsole(printToConsole), numMeasurementsForAvg(numMeasurementsForAvg) 
{
	logFilename = name;
}


Timer::~Timer() {
	if (logToFile && logFile.is_open()) {
		logFile.close();
	}
}


void Timer::start() {
	if (running) {
		return;
	}
	if (logToFile && !logFile.is_open()) {
		logFile.open(LOG_FILENAME_BASE + logFilename + ".txt");
		logFile << "Number of measurements for average = " << numMeasurementsForAvg << endl;
	}
	startTime = chrono::high_resolution_clock::now();
	resetValues();
	running = true;
}


void Timer::clockAvgStart() {
	if (!running) {
		return;
	}
	startTime = chrono::high_resolution_clock::now();
}

bool Timer::clockAvgEnd() {
	if (!running) {
		return false;
	}
	if (callsGLFinish) {
		glFinish();
	}
	if (callsCudaDeviceSynchronize) {
		cudaDeviceSynchronize();
	}

	auto endTime = chrono::high_resolution_clock::now();
	double duration = chrono::duration<double, milli>(endTime - startTime).count();

	accumulatedTime += duration;
	measurementCount++;

	minTime = glm::min(duration, minTime);
	maxTime = glm::max(duration, maxTime);

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

void Timer::end() {
	if (!running) {
		return;
	}
	resetValues();
	running = false;
	if (logToFile && logFile.is_open()) {
		logFile.close();
	}
}

void Timer::reset() {
	end();
	start();
}

void Timer::constructUITab(nk_context * ctx, UserInterface * ui) {

	if (nk_tree_push(ctx, NK_TREE_NODE, name.c_str(), NK_MAXIMIZED)) {
		//ui->nk_property_string(ctx, logFilename, )
		//nk_label(ctx, name.c_str(), NK_TEXT_LEFT);
		//ui->nk_val_bool(ctx, "Call glFinish", callsGLFinish);

		nk_checkbox_label(ctx, "Calls glFinish()", &callsGLFinish);
		nk_checkbox_label(ctx, "Calls cudaDeviceSynchronize()", &callsCudaDeviceSynchronize);

		//ui->nk_val

		nk_tree_pop(ctx);
	}
}

void Timer::resetValues() {
	accumulatedTime = 0.0;
	avgTime = 0.0;
	minTime = DBL_MAX;
	maxTime = 0.0;
	measurementCount = 0;
}
