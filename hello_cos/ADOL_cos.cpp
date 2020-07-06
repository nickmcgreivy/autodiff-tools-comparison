#include <cmath>
#include <iostream>
#include <adolc/adouble.h>          // use of active doubles
#include <adolc/interfaces.h>       // use of basic forward/reverse
#include <adolc/taping.h>           // use of taping

const double pi = 3.14159265358979323846;

adouble f(adouble x) {
	return cos(x);
}


int main() {

	adouble x;

	
	// forward pass

	int tag= 1;
	int keep= 1;
	double x0 = pi / 6;
	trace_on(tag, keep);
	x <<= x0;
	adouble y= f(x);
	double dependent;
	y >>= dependent;
	trace_off();


	// reverse pass

	int n = 1;
	double adjoint[1];
	adjoint[0] = 1.0;
	double* grad = new double[1];

    double u[1];
    u[0] = 1.0;
    double* B = new double[1];

    reverse(tag,1,1,0,u,B);  

	//gradient(tag, n, x0, grad);


	std::cout << "The value of cos at x = " << x.value() << " is " << dependent << std::endl;
	std::cout << "The gradient is " << B[0] << std::endl; 
	return 0;
}
