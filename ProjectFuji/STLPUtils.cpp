#include "STLPUtils.h"

#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;


//#define USE_APPROX_LAT_HEAT



#define LH_A (-6.14342f * 0.00001f)
#define LH_B (1.58927 * 0.001f)
#define LH_C (-2.36418f)
#define LH_D 2500.79f

// expects temperature in Celsius
float computeLatentHeatOfVaporisationC(float T) {
#ifdef USE_APPROX_LAT_HEAT
	return 2.501 * 1000000.0f;
#else
	return (LH_A*T*T*T + LH_B*T*T + LH_C*T + LH_D) * 1000.0f;
#endif
}

// expects temperature in Kelvin
float computeLatentHeatOfVaporisationK(float T) {
	return computeLatentHeatOfVaporisationC(getCelsius(T));
}


//////////////////////////////////////////////////////////////////////////////////////
// Duarte's description of moist adiabats - DOES NOT WORK!
//////////////////////////////////////////////////////////////////////////////////////
float computeRho(float T, float P) {
	float satVP = e_s_degK(T);
	return (P - satVP) / (R_d * T) + satVP / (R_m * T);
}


float computePseudoadiabaticLapseRate(float T, float P) {

	float w = w_degK(T, P);
	float L_v = computeLatentHeatOfVaporisationK(T);


	float res = (1.0f + w) * (1.0f + (L_v * w) / (R_d * T));
	res /= c_pd + w * c_pv + (L_v * L_v * w * (EPS + w)) / (R_d * T * T);
	res *= (-9.81f);

	return res;
}

float getMoistAdiabatIntegralVal(float T, float P) {
	return (computePseudoadiabaticLapseRate(T, P) / (computeRho(T, P) * (-9.81f)));
}
//////////////////////////////////////////////////////////////////////////////////////


float getKelvin(float T) {
	return T + 273.15f;
}

float getCelsius(float T) {
	return T - 273.15f;
}

void toKelvin(float &T) {
	T += 273.15f;
}

void toCelsius(float &T) {
	T -= 273.15f;
}


float computeThetaFromAbsoluteK(float T, float P, float P0) {
	float tmp = (P == P0) ? 1.0f : pow(P0 / P, k_ratio);
	return T * tmp;
}

float computeThetaFromAbsoluteC(float T, float P, float P0) {
	float tmp = (P == P0) ? 1.0f : pow(P0 / P, k_ratio);
	return getCelsius(getKelvin(T) * tmp);
}

float computeAbsoluteFromThetaK(float theta, float P, float P0) {
	float tmp = (P == P0) ? 1.0f : pow(P / P0, k_ratio);
	return (theta * tmp);
}


float computeAbsoluteFromThetaC(float theta, float P, float P0) {
	float tmp = (P == P0) ? 1.0f : pow(P / P0, k_ratio);
	return getCelsius(getKelvin(theta) * tmp);
}






///////////////////////////////////////////////////////////////////////////////////////////////////////
// According to Bahkshaii: https://journals.ametsoc.org/doi/pdf/10.1175/JAMC-D-12-062.1
///////////////////////////////////////////////////////////////////////////////////////////////////////

/// T is in degC! P is in hPa, will be converted to kPa automatically...
float getWetBulbPotentialTemperature_degC_hPa(float T, float P) {

	// theta_w and T must be in Celsius!!!
	// P must be in kiloPascals
	P *= 0.1f;

	float g1, g2, g3, g4, g5, g6;
	float tmp;

	g1 = atan(-0.0141748f * (sqrt(P) * (8.114196f + T) + 65.8402));
	cout << "g1 = " << g1 << endl;

	tmp = (6.558563f + (8.3237 / P));
	g2 = sqrt(69.2840f + sqrt(P)) + tmp * tmp;
	cout << "g2 = " << g2 << endl;

	g3 = exp(17.850425f / P) * sin(0.0510f * (T - P));
	cout << "g3 = " << g3 << endl;

	g4 = 0.00740425f * (T - 23.9263f) * P;
	cout << "g4 = " << g4 << endl;

	g5 = -0.355695f * (0.5997f + P - T + atan(T));
	cout << "g5 = " << g5 << endl;

	g6 = 0.357635f * (0.0922f + atan(T)) * sin(sqrt((3.877869f + P)));
	cout << "g6 = " << g6 << endl;


	float theta_w = g1 + g2 + g3 + g4 + g5 + g6;
	return theta_w;
}


// According to Bahkshaii: https://journals.ametsoc.org/doi/pdf/10.1175/JAMC-D-12-062.1
/// theta_w is in degC! P is in hPa, will be converted to kPa automatically...
float getPseudoadiabatTemperature_degC_hPa(double theta_w, double P) {
	if (theta_w <= -30.0) {
		cout << "theta_w cannot be <= 30 degC in getPseudoadiabatTemperature!" << endl;
		return 0.0f;
	} else if (theta_w >= 45.0) {
		cout << "theta_w cannot be >= 45 degC in getPseudoadiabatTemperature!" << endl;
		return 0.0f;
	}

	// theta_w and T must be in Celsius!!!
	// P must be in kiloPascals
	P *= 0.1;

	double g1, g2, g3, g4, g5, g6, g7;
	double tmp;
	double T = 0.0;

	if (theta_w <= 4.0) {
		// cold subdomain (-30, 4]

		g1 = -20.3313 - 0.0253 * P;
		g2 = sin(sqrt(theta_w + P)) + (theta_w / P) + P - 2.8565;
		g3 = cos(19.6836 + pow(1 + exp(-theta_w), -1.0 / 3.0) + P / 15.0252);
		g4 = 4.4653 * sin(sqrt(P)) - 71.9358;
		g5 = pow(exp(theta_w - 2.71828 * cos(P / 18.5219)), 1.0 / 6.0);
		g6 = theta_w - sin(sqrt(P + theta_w + atan(theta_w) + 6.6165));

		T = g1 + g2 + g3 + g4 + g5 + g6;

	} else if (theta_w <= 21.0) {
		// warm subdomain (4, 21]
		g1 = -9.6285 + cos(log(atan(atan(exp(-9.2121 * theta_w / P)))));
		g2 = theta_w - (19.9563 / P) * atan(theta_w) + (theta_w * theta_w) / (5.47162 * P);
		g3 = sin(log(8 * P * P * P)) * log(2 * pow(P, 3.0 / 2.0));
		g4 = theta_w + (P * theta_w - P + theta_w) / (P - 190.2578);
		g5 = P - (P - 383.0292) / (15.4014 * P - P * P);
		g6 = (1.0 / 3.0) * log(339.0316 - P) + atan(theta_w - P + 95.9839);
		g7 = -log(P) * (298.2909 + 16.5109 * P) / (P - 2.2183);

		T = g1 + g2 + g3 + g4 + g5 + g6 + g7;

	} else {
		// hot subdomain (21, 45)

		g1 = 0.3919 * pow(theta_w, 7.0 / 3.0) * (1.0 / (P * (P + 15.8148)));
		g2 = (19.9724 + (797.7921 / P)) * sin(-19.9724 / theta_w);
		g3 = pow(log(-3.927765 + theta_w + P) * cos(log(theta_w + P)), 3);
		g4 = sqrt(exp(sqrt(theta_w + (1.0 / (1.0 + exp(-P)))) - 1.5603));
		g5 = sqrt(P + theta_w) * exp(atan((P + theta_w) / 7.9081));
		g6 = ((P / (theta_w * theta_w)) * min(9.6112, P - theta_w)) - 13.7300;
		g7 = sin(pow(sin(min(P, 17.3170)), 3) - sqrt(P) + 25.5113 / theta_w);

		T = g1 + g2 + g3 + g4 + g5 + g6 + g7;

	}

	//cout << "g1 = " << g1 << endl;
	//cout << "g2 = " << g2 << endl;
	//cout << "g3 = " << g3 << endl;
	//cout << "g4 = " << g4 << endl;
	//cout << "g5 = " << g5 << endl;
	//cout << "g6 = " << g6 << endl;
	//cout << "g7 = " << g7 << endl;


	return (float)T;
}






///////////////////////////////////////////////////////////////////////////////////////////////////////
// According to pyMeteo implementation & Bakhshaii's article
///////////////////////////////////////////////////////////////////////////////////////////////////////

// expects temperature in Celsius
float e_s_degC(float T) {
	return 611.2f * exp((17.67f * T) / (T + 243.04f));
}

// expects temperature in Kelvin
float e_s_degK(float T) {
	return e_s_degC(getCelsius(T));
}

float w_degK(float T, float P) {
	float e_s = e_s_degK(T);
	return EPS * (e_s / (P - e_s));
}

float w_degC(float T, float P) {
	float e_s = e_s_degC(T);
	return EPS * (e_s / (P - e_s));
}

float dTdz_moist_degK(float T, float P) {
	float L_v = computeLatentHeatOfVaporisationK(T);
	float w = w_degK(T, P);

	float num = 1.0f + (L_v * w) / (R_d * T);
	float den = 1.0f + (L_v * L_v * w) / (c_pd * R_m * T * T);
	return (-9.81f / c_pd) * (num / den);
}

float dTdP_moist_degK(float T, float P) {
	return dTdz_moist_degK(T, P) * -(R_d * T) / (P * 9.81f);
}


// Bakhshaii iterative
float dTdP_moist_degK_Bakhshaii(float T, float P) {
	float L_v = computeLatentHeatOfVaporisationK(T);
	float w = w_degK(T, P);

	float res = 1.0f / P;
	res *= (R_d * T + L_v * w);
	res /= (c_pd + (L_v * L_v * w * EPS / (R_d * T * T)));
	return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Others
///////////////////////////////////////////////////////////////////////////////////////////////////////

float getPressureFromAltitude(float altitude) {
	// based on CRC Handbook of Chemistry and Physics - 96th edition
	return /*100.0f **/ pow(((44331.514f - altitude) / 11880.516f), 1.0f / 0.1902632f);
}


float getAltitudeFromPressure(float pressure) {
	// based on CRC Handbook of Chemistry and Physics - 96th edition
	return (44331.5f - 4946.62f * pow(pressure * 100.0f, 0.190263f));
}












///////////////////////////////////////////////////////////////////////////////////////////////////////
// Moist adiabats and utility functions taken from: https://github.com/NCAR/skewt
// NOT USED!!!
///////////////////////////////////////////////////////////////////////////////////////////////////////

double theta_dry(double t, double p) {
	double u, diff, diff2, diff3;

	u = 1000.0 / p;
	/*
	* Use Taylor series expansion about 700mb for pressures down to 500mb
	*/
	if (p > 500.) {
		diff = u - 1000. / 700.;
		diff2 = diff * diff;
		diff3 = diff * diff2;

		return (t * (1.10714599 +
					 0.22116497 * diff +
					 -0.05531763 * diff2 +
					 0.02213146 * diff3 +
					 -0.01051376 * diff * diff3 +
					 0.00546766 * diff2 * diff3 +
					 -0.00300743 * diff3 * diff3));
	}
	/*
	* Use Taylor series expansion about 350mb for pressures down to 250mb
	*/
	else if (p > 250.) {
		diff = u - 1000. / 350.;
		diff2 = diff * diff;
		diff3 = diff * diff2;

		return (t * (1.34930719 +
					 0.13476972 * diff +
					 -0.01685425 * diff2 +
					 0.00337152 * diff3 +
					 -0.00080084 * diff * diff3 +
					 0.00020824 * diff2 * diff3 +
					 -0.00005727 * diff3 * diff3));

	}
	/*
	* Use Taylor series expansion about 175mb for pressures down to 125mb
	*/
	else if (p > 125.) {
		diff = u - 1000. / 175.;
		diff2 = diff * diff;
		diff3 = diff * diff2;

		return (t * (1.64443524 +
					 0.08212365 * diff +
					 -0.00513518 * diff2 +
					 0.00051362 * diff3 +
					 -0.00006100 * diff * diff3 +
					 0.00000793 * diff2 * diff3 +
					 -0.00000109 * diff3 * diff3));
	}
	/*
	* Otherwise, use the exact form
	*/
	else {
		return (t * pow(u, .28537338));
	}
}


float getLCL(float T, float dewPoint) {
	if (dewPoint > T) {
		cout << "dewPoint > T in getLCL!" << endl;
		return 0.0f;
	}
	return (1.0f / (1.0f / (dewPoint - 56.0f) + log(T / dewPoint) / 800.0f) + 56.0f);
}

float computeEquivalentTheta(float T, float dewPoint, float P) {

	float w = w_degC(dewPoint, P);
	//float theta = computeThetaFromAbsolute(T, P);
	float theta = (float)theta_dry(T, P);
	float t_l = getLCL(T, dewPoint);

	return (theta * exp((3.376f / t_l - 0.00254f) * w * (1.0f + 0.00081f * w)));

}

# define E_3	6.1078
# define T_3	273.15

float getSaturatedAirTemperature(float ept, float P) {
	float t_s = T_3;
	float delta = 60.0f;
	float x = ept - computeEquivalentTheta(t_s, t_s, P);

	while (x > 0.01f || x < -0.01f) {
		t_s += x > 0.0f ? delta : -delta;
		delta /= 2.0f;
		if (delta == 0.0f) {
			delta = 60.0f;
		}
		x = ept - computeEquivalentTheta(t_s, t_s, P);
	}
	return t_s;
}
