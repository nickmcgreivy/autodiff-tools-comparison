using ReverseDiff

f(x::AbstractArray) = sum(cos,x);
x0 = Array{Float64,1}(undef, 1)
x0[1] = pi/6

grad_f =  x -> ReverseDiff.gradient(f, x);
g_x0 = grad_f(x0)

println(string("The value of cos(x) at x = ", first(x0), " is ", f(x0)))
println(string("The derivative of cos(x) at x = ", first(x0), " is ", first(g_x0)))
