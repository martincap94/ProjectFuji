#include "Timer.h"

#include <glm\glm.hpp>

#include <glad\glad.h>

#include "Utils.h"

Timer::Timer(string name, bool callsGLFinish, bool callsCudaDeviceSynchronize, 
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
		string filename = LOG_FILENAME_BASE + getTimeStr() + " " + logFilename + ".txt";
		logFile.open(filename);
		//logFile << "Number of measurements for average = " << numMeasurementsForAvg << endl;
	}
	startTime = chrono::high_resolution_clock::now();
	resetValues();
	running = true;
}


void Timer::clockAvgStart() {
	if (!running) {
		return;
	}
	//if (callsGLFinish) {
	//	glFinish();
	//}
	//if (callsCudaDeviceSynchronize) {
	//	cudaDeviceSynchronize();
	//}
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
	frameTime = duration;

	accumulatedTime += duration;
	measurementCount++;

	avgTime = accumulatedTime / (double)measurementCount;


	minTime = glm::min(duration, minTime);
	maxTime = glm::max(duration, maxTime);

	if (measurementCount >= numMeasurementsForAvg) {
		lastAvgTime = avgTime;
		lastMinTime = minTime;
		lastMaxTime = maxTime;
		if (logToFile) {
			logFile << avgTime << endl;
		}
		if (printToConsole) {
			cout << "Avg. time for " << numMeasurementsForAvg << " measurements is " << avgTime << " [ms]." << endl;
		}
		resetValues();
		return true;
	}
	return false;
}

void Timer::end() {
	if (!running) {
		return;
	}
	//resetValues();
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
	
	const static int bufferLength = 32;
	//static char nameBuffer[bufferLength];
	//static int nameLength;

	static string note;
	static char noteBuffer[bufferLength];
	static int noteLength;

	if (!running) {
		if (nk_tree_push_id(ctx, NK_TREE_NODE, name.c_str(), NK_MAXIMIZED, index)) {

			nk_layout_row_dynamic(ctx, 15.0f, 1);
			if (nk_button_label(ctx, "Start Timer")) {
				start();
			}
			//nk_label(ctx, "Log Filename:", NK_TEXT_LEFT);
			//ui->nk_property_string(ctx, logFilename, nameBuffer, bufferLength, nameLength);

			nk_tree_pop(ctx);
		}
	}

	// do not use else/else if (running) to switch the pane in the same frame (instead of in the next frame)
	if (running) {
		if (nk_tree_push_id(ctx, NK_TREE_NODE, name.c_str(), NK_MAXIMIZED, index)) {
			//ui->nk_property_string(ctx, logFilename, )
			//nk_label(ctx, name.c_str(), NK_TEXT_LEFT);
			//ui->nk_val_bool(ctx, "Call glFinish", callsGLFinish);

			nk_checkbox_label(ctx, "Calls glFinish()", &callsGLFinish);
			nk_checkbox_label(ctx, "Calls cudaDeviceSynchronize()", &callsCudaDeviceSynchronize);

			nk_property_int(ctx, "Num Measurements for Avg", 1, &numMeasurementsForAvg, 100000, 1, 0.2f);

			ui->nk_label_time(ctx, frameTime, 2, "Frame Time: ");
			ui->nk_label_time(ctx, avgTime, 2, "Average Time: ");
			ui->nk_label_time(ctx, accumulatedTime, 2, "Accumulated Time: ");
			nk_label(ctx, ("Frame Count: " + to_string(measurementCount)).c_str(), NK_TEXT_LEFT);
			ui->nk_label_time(ctx, minTime, 2, "Min Time: ");
			ui->nk_label_time(ctx, maxTime, 2, "Max Time: ");

			ui->nk_label_time(ctx, lastAvgTime, 2, "Last Average Time: ");
			ui->nk_label_time(ctx, lastMinTime, 2, "Last Min Time: ");
			ui->nk_label_time(ctx, lastMaxTime, 2, "Last Max Time: ");

			if (nk_tree_push(ctx, NK_TREE_NODE, "Log File", NK_MAXIMIZED)) {

				nk_label(ctx, "Note:", NK_TEXT_LEFT);
				ui->nk_property_string(ctx, note, noteBuffer, bufferLength, noteLength);
				nk_layout_row_dynamic(ctx, 15.0f, 1);
				if (nk_button_label(ctx, "Push Note to Log File")) {
					pushNoteToLogFile(note);
				}
				if (nk_button_label(ctx, "Push Num Measurements for Avg Note To Log File")) {
					pushNumMeasurementsForAvgToLogFile();
				}
				nk_checkbox_label(ctx, "Log to File", &logToFile);
				nk_tree_pop(ctx);
			}

			if (nk_button_label(ctx, "End Timer")) {
				end();
			}

			nk_tree_pop(ctx);
		}
	}
}

void Timer::resetValues() {
	accumulatedTime = 0.0;
	avgTime = 0.0;
	minTime = DBL_MAX;
	maxTime = 0.0;
	measurementCount = 0;
}

void Timer::pushNoteToLogFile(std::string note) {
	if (logFile.is_open()) {
		logFile << endl << note << endl;
	}
}

void Timer::pushNumMeasurementsForAvgToLogFile() {
	if (logFile.is_open()) {
		logFile << "Number of measurements for average = " << numMeasurementsForAvg << endl;
	}
}


