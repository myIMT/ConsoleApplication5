/*
C++ code to print histogram

*/
#include<iostream>
#include<conio.h>
#include<iomanip>

using namespace std;

int findmax(int vals[], int n) {
	int max = vals[0];
	for (int i = 0; i<n; i++)
		if (max<vals[i]) max = vals[i];
	return max;
}

int main()
{

	int n = 5;
		int fsize;
	//cout << "Enter number of data points:";
	//cin >> n;
	int vals[5];
	for (int i = 0; i<n; i++) {
		cout << "Value" << i + 1 << ":";
		cin >> vals[i]; //intialize data
	}
	int m = findmax(vals, n); //find max value of data points
	if (m>n) fsize = m + 1;
	else fsize = n;
	int freq[5]; //declare frequency array with an appropriate size
					 //The size of frequency array can be the size of the vals array
					 //or the max value of the vals array items plus 1

	for (int i = 0; i<fsize; i++) //initialize frequency array
		freq[i] = 0;

	//compute frequencies
	for (int i = 0; i<n; i++)
		freq[vals[i]]++;

	//print histogram
	cout << "\n....Histogram....\n\n";
	for (int i = 0; i<fsize; i++) {
		cout << left;
		cout << setw(5) << i;
		cout << setw(5) << freq[i];
		for (int j = 1; j <= freq[i]; j++) cout << "*";
		cout << "\n";
	}


	system("PAUSE");
	return 0;


}