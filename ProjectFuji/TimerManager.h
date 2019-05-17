///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       TimerManager.h
* \author     Martin Cap
*
*	Namespace that provides utility functions for timer management across the whole application.
*	Must be initialized and torn down (destroyed) before and after use, respectively!
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Timer.h"

#include "UserInterface.h"
#include <nuklear.h>

//! Namespace that acts as a singleton and provides easily accessible management of all timers for the application.
/*!
	The namespace must be initialized and torn down before and after use, respectively!
*/
namespace TimerManager {
	
	namespace {
		//! Starts the benchmarking process.
		void startBenchmarking();
	}

	//! Initializes the TimerManager singleton.
	void init();

	//! Tears down the TimerManager singleton.
	void tearDown();


	//! Creates a timer with the given properties.
	/*!
		Constructs timer that can log to file, print to console and measure averages.
		\param[in] name						Name of the timer to be used in UI.
		\param[in] callGLFinish				Whether glFinish() should be called after each clock call.
		\param[in] callsCudaDeviceSynchronize	Whether cudaDeviceSynchronize() should be called after each clock call.
		\param[in] logToFile				Whether the timer should log to file.
		\param[in] printToConsole			Whether the timer should print average time measurements to console.
		\param[in] numMeasurementsForAvg	Number of measurements needed for average time computation.
		\return								Pointer to the created timer.
	*/
	Timer *createTimer(std::string name, bool callsGLFinish = false, bool callsCudaDeviceSynchronize = false, bool logToFile = true, bool printToConsole = false, int numMeasurementsForAvg = 1000);

	//! Return pointer to a timer with the given name.
	/*!
		\param[in]		Name of the timer we want a pointer to.
		\return			Pointer to a timer with the given name.
	*/
	Timer *getTimer(std::string name);

	//! Starts all timers managed by the TimerManager.
	void startAllTimers();

	//! Resets all timers managed by the TimerManager.
	void resetAllTimers();

	//! Ends all timers managed by the TimerManager.
	void endAllTimers();

	//! Writes frame times of all timers to the benchmark file.
	void writeToBenchmarkFile();

	//! Constructs the user interface tab for all timers.
	/*!
		\param[in] ctx		Nuklear context that is currently in use.
		\parma[in] ui		The UserInterface instance.
	*/
	void constructTimersUITab(struct nk_context *ctx, UserInterface *ui);


}

