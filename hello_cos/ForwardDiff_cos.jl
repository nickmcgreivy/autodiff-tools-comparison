using ForwardDiff

f(x::Real) = cos(x);
x0 = pi/6

grad_f = x -> ForwardDiff.derivative(f, x);
g_x0 = grad_f(x0)

println(string("The value of cos(x) at x = ", first(x0), " is ", f(x0)))
println(string("The derivative of cos(x) at x = ", first(x0), " is ", first(g_x0)))

