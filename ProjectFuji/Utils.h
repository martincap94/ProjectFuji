///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Utils.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines utility functions that may be used anywhere in the application.
*
*  Defines utility functions that may be used anywhere in the application.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////


#pragma once

#include <string>


// Trimming functions taken from: https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string
// answer by user Galik

/// Trim from left.
inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v") {
	s.erase(0, s.find_first_not_of(t));
	return s;
}

/// Trim from right.
inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v") {
	s.erase(s.find_last_not_of(t) + 1);
	return s;
}

/// Trim from left & right.
inline std::string& trim(std::string& s, const char* t = " \t\n\r\f\v") {
	return ltrim(rtrim(s, t), t);
}

// copying versions
/// Trim from left.
inline std::string ltrim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return ltrim(s, t);
}

/// Trim from right.
inline std::string rtrim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return rtrim(s, t);
}

/// Trim from left & right.
inline std::string trim_copy(std::string s, const char* t = " \t\n\r\f\v") {
	return trim(s, t);
}

