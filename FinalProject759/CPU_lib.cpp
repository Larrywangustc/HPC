#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <time.h>
#include <iomanip>
#include "CPU_lib.h"
#define ERROR 0
using namespace std;

int GaussianEliminate(std::vector<std::vector<double>>& array)
{
	int k, j, l;
	for (k = 0;k < array.size() - 1;k++) {
		for (j = k + 1;j < array.size();j++) {
			if (array[k][k] == 0) {
				std::cout << "����";
				return ERROR;
			}
			else {
				array[j][k] = array[j][k] / array[k][k];
			}
		}
		for (l = k + 1;l < array.size();l++) {
			for (j = k + 1;j < array.size();j++) {
				array[j][l] = array[j][l] - array[j][k] * array[k][l];
			}
		}
	}
	return 0;
}

int choleskyElimination(vector<vector<double>>& array) {
	for (int k = 0;k < array.size();k++) {
		array[k][k] = sqrt(array[k][k]);
		for (int i = k + 1;i < array.size();i++) {
			array[i][k] /= array[k][k];
		}
		for (int j = k + 1;j < array.size();j++) {
			for (int i = j;i < array.size();i++) {
				array[i][j] -= array[i][k] * array[j][k];
			}
		}
	}
	for (int j = 0;j < array.size();j++) {
		for (int i = j + 1;i < array.size();i++) {
			array[j][i] = array[i][j];
		}
	}
	return 0;
}

void backsweep(vector<vector<double>>& array) {
	for (int i = array.size() - 1;i >= 1;i--) {
		array[i][array.size()] = array[i][array.size()] / array[i][i];
		for (int j = 0;j <= i - 1;j++) {
			array[j][array.size()] -= array[i][array.size()] * array[j][i];
		}
	}
	array[0][array.size()] = array[0][array.size()] / array[0][0];
}

void forwardsweep(vector<vector<double>>& array) {
	for (int j = 0;j < array.size() - 1;j++) {
		array[j][array.size()] /= array[j][j];
		for (int i = j + 1;i < array.size();i++) {
			array[i][array.size()] -= array[j][array.size()] * array[i][j];
		}
	}
	array[array.size() - 1][array.size()] = array[array.size() - 1][array.size()] / array[array.size() - 1][array.size() - 1];
}

void forwardsweep1(vector<vector<double>>& array) {
	for (int j = 0;j < array.size() - 1;j++) {
		for (int i = j + 1;i < array.size();i++) {
			array[i][array.size()] -= array[j][array.size()] * array[i][j];
		}
	}
}

void LUSOLVING(vector<vector<double>>& array) {
	forwardsweep(array);
	backsweep(array);
}

void LUSOLVING1(vector<vector<double>>& array) {
	forwardsweep1(array);
	backsweep(array);
}

void GaussianEliminatesolving(vector<vector<double>>array, vector<double>& result) {
	GaussianEliminate(array);
	LUSOLVING1(array);
	for (int i = 0;i < array.size();i++) {
		//cout << array[i][array.size()] << "";
		result[i] = array[i][array.size()];
		//cout << "\t";
	}
	//cout << "\n";

}

void choleskyEliminationsolving(vector<vector<double>>array, vector<double>& result) {
	choleskyElimination(array);
	LUSOLVING(array);
	for (int i = 0;i < array.size();i++) {
		//cout << array[i][array.size()] << "";
		result[i] = array[i][array.size()];
		//cout << "\t";
	}
	//cout << "\n";
}

void Householdertransformation(vector<double>x, vector<double>& v, double& b) {
	double n = x.size();
	double y = InfiniteNorm(x);
	for (int i = 0;i < n;i++) {
		x[i] /= y;
	}
	double c = 0;
	for (int i = 1;i < n;i++) {
		c += x[i] * x[i];
	}
	for (int i = 1;i < n;i++) {
		v[i] = x[i];
	}
	double a;
	if (c == 0) {
		b = 0;
	}
	else {
		a = sqrt(x[0] * x[0] + c);
		if (x[0] <= 0) {
			v[0] = x[0] - a;
		}
		else {
			v[0] = -c / (x[0] + a);
		}
		b = 2.0 * v[0] * v[0] / (c + v[0] * v[0]);
		for (int i = 1;i < n;i++) {
			v[i] /= v[0];
		}
		v[0] = 1;
	}
}

void HouseholderQRElimination(vector<vector<double>>& A, vector<double>& d) {
	d = vector<double>(A[0].size());
	vector<double>v;
	vector<double>a;
	double b;
	double temp;
	double m = A.size();
	double n = A[0].size();
	for (int j = 0;j < A[0].size();j++) {
		if (j < A.size() - 1) {
			a = vector<double>(m - j);
			for (int i = 0;i < m - j;i++) {
				a[i] = A[i + j][j];
			}
			v = vector<double>(a.size());
			Householdertransformation(a, v, b);
			for (int k = j;k < n;k++) {
				double temp = A[j][k];
				for (int i = j + 1;i < m;i++) {
					temp += v[i - j] * A[i][k];
				}
				temp *= b;
				A[j][k] -= temp;
				for (int i = j + 1;i < m;i++) {
					A[i][k] -= temp * v[i - j];
				}
			}
			d[j] = b;
			for (int i = 1;i < m - j;i++) {
				A[i + j][j] = v[i];
			}
		}
	}
}

void LU_array(float** a, float** l, float** u, int size){
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			if (j < i)
			{
				l[j][i] = 0;
				continue;
			}
			l[j][i] = a[j][i];
			for (int k = 0; k < i; k++){
				l[j][i] = l[j][i] - l[j][k] * u[k][i];
			}
		}
		for (int j = 0; j < size; j++){
			if (j < i){
				u[i][j] = 0;
				continue;
			}
			if (j == i){
				u[i][j] = 1;
				continue;
			}
			u[i][j] = a[i][j] / l[i][i];
			for (int k = 0; k < i; k++){
				u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
			}
		}
	}
}