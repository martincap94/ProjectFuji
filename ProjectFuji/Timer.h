///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Timer.h
* \author     Martin Cap
* \date       2018/12/23
*	
*	--- DEPRECATED (at this moment - requires update) ---
*	Defines the Timer class which is used for basic timing and logging.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

#include "Config.h"

using namespace std;

typedef chrono::time_point<chrono::high_resolution_clock> Timepoint;

/// Timer for basic measurements.
/**
	Timer that provides functionality for basic measurements.
	Can accumulate time and average it for specified number of clock() calls (frames).
*/
class Timer {
public:

	Timepoint startTime;		///< Start time of the timer
	Timepoint lastClockTime;	///< Last time the clock function was called

	string configString;		///< Configuration string for naming log files

	double accumulatedTime;		///< Currently accumulated time that is used for average measurements
	double avgTime;				///< Average time measured
	int numMeasurementsForAvg;	///< Number of measurements (clock calls) for average time computation
	int measurementCount;		///< Counter of current measurement count

	bool running;				///< Whether the timer is running or not

	bool logToFile;				///< Whether the timer should log to file or not
	bool printToConsole;		///< Whether the timer should print to console

	ofstream logFile;			///< Output file stream for the log file

	/// Constructs timer that can log to file, print to console and measure averages.
	/**
		Constructs timer that can log to file, print to console and measure averages.
		\param[in] logToFile				Whether the timer should log to file.
		\param[in] printToConsole			Whether the timer should print average time measurements to console.
		\param[in] numMeasurementsForAvg	Number of measurements needed for average time computation.
	*/
	Timer(bool logToFile = false, bool printToConsole = true, int numMeasurementsForAvg = 1000);

	/// Default destructor.
	~Timer();

	/// Start the timer and reset counters.
	void start();

	/// Notify the timer about the start of the timed section for average.
	void clockAvgStart();

	/// Notify the timer about the end of the timed section for average.
	/**
		Notify the timer about the end of the timed section for average.
		Prints to console and logs to file based on the logToFile and printToConsole member variables.
		\return Whether the average was measured in this step.
	*/
	bool clockAvgEnd();

	/// Ends the timer.
	/**
		Ends the timer and resets all counters.
		\param[in] printResults		Prints total time if true.
	*/
	void end(bool printResults = false);


};

