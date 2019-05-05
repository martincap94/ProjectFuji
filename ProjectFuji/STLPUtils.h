///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       STLPUtils.h
* \author     Martin Cap
*
*	Contains utility SkewT/LogP (STLP) functions and definitions that are used in STLP diagram creation.
*	The functions are mainly used in adiabat creation (dry and moist adiabats).
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cmath>
#include <algorithm>

#define R_d 287.05307f		//!< Dry air gas constant
#define R_m 461.5f			//!< Moist air gas constant
#define R_v R_m				//!< Moist air gas constant (alternate notation)
#define EPS (R_d / R_m)		//!< Epsilon used for computing saturation mixing ratio
#define c_pd 1005.7f		//!< Specific heat capacity for dry air at constant pressure
#define c_pv 1875.0f		//!< Specific heat capacity for water vapor at constant pressure
//#define k_ratio 0.286f
#define k_ratio (R_d / c_pd)	//!< Poisson's constant: http://glossary.ametsoc.org/wiki/Poisson_constant
								//!< Dimensionless ratio of dry air gas constant and specific heat capacity for dry air at constant pressure


//! Computes the latent heat of vaporisation/condensation from the given temperature in celsius.
/*!
	\param[in] T	Absolute temperature in celsius [C].
	\return			Laten heat of vaporisation/condensation.
*/
float computeLatentHeatOfVaporisationC(float T);

//! Computes the latent heat of vaporisation/condensation from the given temperature in celsius.
/*!
	\param[in] T	Absolute temperature in kelvin [K].
	\return			Laten heat of vaporisation/condensation.
*/
float computeLatentHeatOfVaporisationK(float T);

//! Get temperature converted to kelvins.
/*!
	\param[in] T	Temperature in celsius to be converted.
	\return			Temperature in kelvins.
*/
float getKelvin(float T);

//! Get temperature converted to celsius.
/*!
	\param[in] T	Temperature in kelvins to be converted.
	\return			Temperature in celsius.
*/
float getCelsius(float T);

//! Convert given temperature to kelvins.
/*!
	\param[in] T	Temperature in celsius to be converted.
*/
void toKelvin(float &T);

//! Convert given temperature to celsius.
/*!
	\param[in] T	Temperature in kelvins to be converted.
*/
void toCelsius(float &T);

//! Computes potential temperature (in kelvin [K]) from absolute temperature (in kelvin [K]).
/*!
	Please note that the unit of the pressure parameters is not important.
	The two parameters just need to use the same units. We only need their ratio.

	\param[in] T	Absolute temperature in kelvins [K].
	\param[in] P	Pressure (assumed to be in hectopascals [hPa]).
	\param[in] P0	Ground pressure (default in hectopascals [hPa]).
	\return			Potential temperature in kelvin [K].
*/
float computeThetaFromAbsoluteK(float T, float P, float P0 = 1000.0f);

//! Computes potential temperature (in celsius [C]) from absolute temperature (in celsius [C]).
/*!
	Please note that the unit of the pressure parameters is not important.
	The two parameters just need to use the same units. We only need their ratio.

	\param[in] T	Absolute temperature in kelvins [C].
	\param[in] P	Pressure (assumed to be in hectopascals [hPa]).
	\param[in] P0	Ground pressure (default in hectopascals [hPa]).
	\return			Potential temperature in kelvin [C].
*/
float computeThetaFromAbsoluteC(float T, float P, float P0 = 1000.0f);

//! Computes absolute temperature (in kelvin [K]) from potential temperature (in kelvin [K]).
/*!
	Please note that the unit of the pressure parameters is not important.
	The two parameters just need to use the same units. We only need their ratio.

	\param[in] theta	Potential temperature in kelvins [K].
	\param[in] P		Pressure (assumed to be in hectopascals [hPa]).
	\param[in] P0		Ground pressure (default in hectopascals [hPa]).
	\return				Absolute temperature in kelvin [K].
*/
float computeAbsoluteFromThetaK(float theta, float P, float P0 = 1000.0f);

//! Computes absolute temperature (in celsius [C]) from potential temperature (in celsius [C]).
/*!
	Please note that the unit of the pressure parameters is not important.
	The two parameters just need to use the same units. We only need their ratio.

	\param[in] theta	Potential temperature in celsius [C].
	\param[in] P		Pressure (assumed to be in hectopascals [hPa]).
	\param[in] P0		Ground pressure (default in hectopascals [hPa]).
	\return				Absolute temperature in celsius [C].
*/
float computeAbsoluteFromThetaC(float theta, float P, float P0 = 1000.0f);




///////////////////////////////////////////////////////////////////////////////////////////////////////
// According to pyMeteo implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////

//! Computes the saturation vapor pressure approximation.
/*!
	Uses Bolton's approximation from his article:
	https://journals.ametsoc.org/doi/abs/10.1175/1520-0493(1980)108%3C1046:TCOEPT%3E2.0.CO%3B2

	\param[in] T	Absolute temperature in celsius [C].
	\return			Saturation vapor pressure approximation in Pascal [Pa].
*/
float e_s_degC(float T);

//! Computes the saturation vapor pressure approximation.
/*!
	Uses Bolton's approximation from his article:
	https://journals.ametsoc.org/doi/abs/10.1175/1520-0493(1980)108%3C1046:TCOEPT%3E2.0.CO%3B2

	\param[in] T	Absolute temperature in kelvin [K].
	\return			Saturation vapor pressure approximation in Pascal [Pa].
*/
float e_s_degK(float T);


//! Computes the saturation mixing ratio [g/kg] of water vapor that air can hold for given pressure and temperature.
/*!
	\param[in] T	Absolute temperature in kelvin [K].
	\param[in] P	Pressure in pascals [Pa].
	\return			Saturation mixing ratio [g/kg].
*/
float w_degK(float T, float P);


//! Computes the saturation mixing ratio [g/kg] of water vapor that air can hold for given pressure and temperature.
/*!
	\param[in] T	Absolute temperature in celsius [C].
	\param[in] P	Pressure in pascals [Pa].
	\return			Saturation mixing ratio [g/kg].
*/
float w_degC(float T, float P);

//! Computes moist/saturated adiabatic lapse rate.
/*!
	Based on pyMeteo implementation: https://github.com/cwebster2/pyMeteo
	\param[in] T	Absolute temperature in kelvin [K].
	\param[in] P	Pressure in pascals [Pa].
	\return			Moist/saturated adiabatic lapse rate.
*/
float dTdz_moist_degK(float T, float P);

//! Computes vertical temperature gradient for moist/saturated adiabatic lapse rate.
/*!
	Based on pyMeteo implementation: https://github.com/cwebster2/pyMeteo
	\param[in] T	Absolute temperature in kelvin [K].
	\param[in] P	Pressure in pascals [Pa].
	\return			Vertical temperature gradient.
*/
float dTdP_moist_degK(float T, float P);

//! Computes vertical temperature gradient for moist/saturated adiabatic lapse rate.
/*!
	Based on Bakhshaii's article about her non-iterative method:
	https://journals.ametsoc.org/doi/full/10.1175/JAMC-D-12-062.1

	\param[in] T	Absolute temperature in kelvin [K].
	\param[in] P	Pressure in pascals [Pa].
	\return			Vertical temperature gradient.
*/
float dTdP_moist_degK_Bakhshaii(float T, float P);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Others
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Pressure in hPa, altitude in meters
//! Returns pressure [hPa] computed from altitude [m].
/*!
	\param[in] altitude		Altitude in meters[m].
	\return					Pressure in hectopascals [hPa].
*/
float getPressureFromAltitude(float altitude);


// Pressure in hPa, altitude in meters
//! Returns altitude [m] computed from pressure [hPa]
/*!
	\param[in] pressure		Pressure in hectopascals [hPa].
	\return					Altitude in meters [m].
*/
float getAltitudeFromPressure(float pressure);


//! Compute density (rho) from saturation water vapor pressure.
/*!
	\param[in] T	Absolute temperature in kelvin [K].
	\param[in] P	Pressure in pascals [Pa].
	\return			Density computed for moist adiabat integration.
*/
float computeRho(float T, float P);


//! Computes pseudoadiabatic lapse rate according to Duarte's approach & AMS glossary.
/*!
	\param[in] T	Absolute temperature in kelvin [K].
	\param[in] P	Pressure in pascals [Pa].
	\return			Pseudoadiabatic lapse rate.
*/
float computePseudoadiabaticLapseRate(float T, float P);

//! Computes pseudoadiabatic lapse rate according to Duarte's approach & AMS glossary.
/*!
	\param[in] T	Absolute temperature in kelvin [K].
	\param[in] P	Pressure in pascals [Pa].
	\return			Integration value for moist adiabat creation.
*/
float getMoistAdiabatIntegralVal(float T, float P);





////////////////////////////////////////////////////////////////////////////////////////////////////////
// Non-Iterative According to Bahkshaii: https://journals.ametsoc.org/doi/pdf/10.1175/JAMC-D-12-062.1 //
////////////////////////////////////////////////////////////////////////////////////////////////////////

//! Computes the pseudoadiabat (moist adiabat, saturated adiabat) wet bulb temperature.
/*!
	Uses Bakhshaii's non-iterative approach:
	https://journals.ametsoc.org/doi/pdf/10.1175/JAMC-D-12-062.1
	Note that P is converted from hPa to kPa automatically.

	\param[in] T			Absolute temperature in celsius [C].
	\param[in] P			Pressure in hectopascals [hPa].
	\return					Wet bulb potentail temperature in celsius [C].
*/
float getWetBulbPotentialTemperature_degC_hPa(float T, float P);


//! Computes the pseudoadiabat (moist adiabat, saturated adiabat) temperature non-iteratively.
/*!
	Uses Bakhshaii's non-iterative approach:
	https://journals.ametsoc.org/doi/pdf/10.1175/JAMC-D-12-062.1
	Note that P is converted from hPa to kPa automatically.

	\param[in] theta_w		Potential temperature in celsius [C].
	\param[in] P			Pressure in hectopascals [hPa].
	\return					Pseudoadiabat temperature.
*/
float getPseudoadiabatTemperature_degC_hPa(double theta_w, double P);








