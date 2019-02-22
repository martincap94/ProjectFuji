#pragma once

#include <cmath>
#include <algorithm>

#define R_d 287.05307f		///< Dry air gas constant
#define R_m 461.5f			///< Moist air gas constant
#define R_v R_m
#define EPS (R_d / R_m)


// testing values for 300 K (27 degC) ////////////////////////////////////////
// https://www.engineeringtoolbox.com/dry-air-properties-d_973.html
//#define c_pd 0.7178f
//#define c_pd 1003.5f

//#define c_pd 1.0057f
#define c_pd 1005.7f

// https://www.engineeringtoolbox.com/water-vapor-d_979.html
//#define c_pv 1.864f
//#define c_pv 4218.0f

//#define c_pv 1.875f
#define c_pv 1875.0f;
//////////////////////////////////////////////////////////////////////////////


float getSaturationVaporPressure(float T);

float getMixingRatioOfWaterVapor(float T, float P);

float computeRho(float T, float P);
float computeLatentHeatOfVaporisation(float T);
float computePseudoadiabaticLapseRate(float T, float P);
float getMoistAdiabatIntegralVal(float T, float P);

float getKelvin(float T);
float getCelsius(float T);

void toKelvin(float &T);
void toCelsius(float &T);

float computeThetaFromAbsolute(float T, float P);





///////////////////////////////////////////////////////////////////////////////////////////////////////
// Moist adiabats and utility functions taken from: https://github.com/NCAR/skewt
///////////////////////////////////////////////////////////////////////////////////////////////////////

double theta_dry(double t, double p);

float getLCL(float T, float dewPoint);
float computeEquivalentTheta(float T, float dewPoint, float P);

# define E_3	6.1078
# define T_3	273.15

float getSaturatedAirTemperature(float ept, float P);



///////////////////////////////////////////////////////////////////////////////////////////////////////
// According to Bahkshaii: https://journals.ametsoc.org/doi/pdf/10.1175/JAMC-D-12-062.1
///////////////////////////////////////////////////////////////////////////////////////////////////////

/// T is in degC! P is in hPa, will be converted to kPa automatically...
float getWetBulbPotentialTemperature(float T, float P);

// According to Bahkshaii: https://journals.ametsoc.org/doi/pdf/10.1175/JAMC-D-12-062.1
/// theta_w is in degC! P is in hPa, will be converted to kPa automatically...
float getPseudoadiabatTemperature(double theta_w, double P);




///////////////////////////////////////////////////////////////////////////////////////////////////////
// According to pyMeteo implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////


float e_s(float T);
float w_vs(float T, float pd);
float dTdz_moist(float T, float p);
float dTdp_moist(float T, float p);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Others
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Pressure in hPa, altitude in meters
float getPressureFromAltitude(float altitude);

// Pressure in hPa, altitude in meters
float getAltitudeFromPressure(float pressure);









