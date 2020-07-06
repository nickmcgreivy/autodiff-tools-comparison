#include <stan/math/rev.hpp>
#include <iostream>

const double pi = 3.14159;

int main() {

	stan::math::var x = pi / 6;

	stan::math::var y = stan::math::cos(x);

	y.grad();


 	std::cout << "The value of cos(x) at x = "
 			<< x << " is " << y
 			<< std::endl;
 	std::cout << "The gradient of cos(x) is "
 			<< x.adj() << std::endl;
}