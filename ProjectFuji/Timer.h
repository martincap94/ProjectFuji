///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Timer.h
* \author     Martin Cap
* \date       2018/12/23
*	
*	--- DEPRECATED (at this moment - requires update) ---
*	\deprecated
*	Defines the Timer class which is used for basic timing and logging.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

#include <nuklear.h>
#include "UserInterface.h"

#include "Config.h"

using namespace std;

typedef chrono::time_point<chrono::high_resolution_clock> Timepoint;

//! Timer for basic measurements.
/*!
	Timer that provides functionality for basic measurements.
	Can accumulate time and average it for specified number of clock() calls (frames).
*/
class Timer {
public:

	string name;				//!< Name of the timer

	Timepoint startTime;		//!< Start time of the timer

	string logFilename;			//!< Name of the log file

	double accumulatedTime;		//!< Currently accumulated time that is used for average measurements
	double frameTime;			//!< Last measured frame time
	double avgTime;				//!< Average time measured
	double lastAvgTime;			//!< Last measured average time
	double maxTime;				//!< Maximum time measured
	double minTime;				//!< Minimum time measured
	double lastMaxTime = 0.0;	//!< Last measured maximum time
	double lastMinTime = 0.0;	//!< Last measured minimum time

	int numMeasurementsForAvg;	//!< Number of measurements (clock calls) for average time computation
	int measurementCount;		//!< Counter of current measurement count

	int logToFile;				//!< Whether the timer should log to file or not
	int printToConsole;		//!< Whether the timer should print to console

	int callsGLFinish = 0;	//!< This gives us the option to (roughly) measure OpenGL rendering
								//!< If precise timing necessary, glQueries should be used instead		
	int callsCudaDeviceSynchronize = 0;	//!< This gives us the option to (roughly) measure CUDA operations/blocks of code


	int index = -1;

	//! Constructs timer that can log to file, print to console and measure averages.
	/*!
		Constructs timer that can log to file, print to console and measure averages.
		\param[in] callGLFinish				Whether glFinish() should be called after each clock call.
		\param[in] callsCudaDeviceSynchronize	Whether cudaDeviceSynchronize() should be called after each clock call.
		\param[in] logToFile				Whether the timer should log to file.
		\param[in] printToConsole			Whether the timer should print average time measurements to console.
		\param[in] numMeasurementsForAvg	Number of measurements needed for average time computation.
	*/
	Timer(string name, bool callsGLFinish = false, bool callsCudaDeviceSynchronize = false,
		  bool logToFile = true, bool printToConsole = false, int numMeasurementsForAvg = 1000);

	//! Default destructor.
	~Timer();

	//! Start the timer and reset counters.
	void start();

	//! Notify the timer about the start of the timed section for average.
	void clockAvgStart();

	//! Notify the timer about the end of the timed section for average.
	/*!
		Notify the timer about the end of the timed section for average.
		Prints to console and logs to file based on the logToFile and printToConsole member variables.
		\return Whether the average was measured in this step.
	*/
	bool clockAvgEnd();



	//! Ends the timer.
	/*!
		Ends the timer and resets all counters.
	*/
	void end();

	//! Resets the timer.
	/*!
		This is necessary if we want to open a new log file.
		Uses end() and start() in this order.
	*/
	void reset();


	void constructUITab(struct nk_context *ctx, UserInterface *ui);


private:

	ofstream logFile;			//!< Output file stream for the log file

	bool running = false;

	void resetValues();

	void pushNoteToLogFile(std::string note);
	void pushNumMeasurementsForAvgToLogFile();


};

