using Distributions
using JuMP
using CPLEX
using BenchmarkTools

# Demand distribution
mu = [100.0, 200, 150, 170, 180, 170, 170]
sigma = [20.0, 50, 30, 50, 40, 30, 50]

# For each location, we discretize the levels into
# low, medium, high, using quantiles.
q = [quantile(Normal(mu[i], sigma[i]), [1/6, 3/6, 5/6]) for i in 1:7]

const d = vec(collect(Iterators.product(q...)));
# Number of locations
const N = 7
# Number of scenarios
const M = 3^7

# holding cost 
h = 1.0
# shortage cost
p = 4.0
# transshipment cost
c = 0.5

# Decision Variables
model = Model(CPLEX.Optimizer)
@variables(model, begin
        e[1:M,1:N] >= 0
        f[1:M,1:N] >= 0
        q[1:M,1:N] >= 0
        r[1:M,1:N] >= 0
        t[1:M,1:N,1:N] >= 0
        s[1:N] >= 0
    end);

# Objective Function Data
@objective(model, Min, (sum(h*e[m,i] for m in 1:M for i in 1:N)
        +sum(c*t[m,i,j] for m in 1:M for i in 1:N for j in 1:N if i!=j)
    +sum(p*r[m,i] for m in 1:M for i in 1:N))/M)

@constraint(model, c1[m=1:M, i=1:N], f[m,i] 
    + sum(t[m,i,j] for j in 1:N if i!=j) + e[m,i] == s[i]);

@constraint(model, c2[m=1:M, i=1:N], f[m,i] 
    + sum(t[m,j,i] for j in 1:N if i!=j) + r[m,i] == d[m][i]);

@constraint(model, c3[m=1:M], sum(r[m,1:N])
    +sum(q[m,1:N])==sum(d[m]));

@constraint(model, c4[m=1:M,i=1:N], e[m,i]+q[m,i]==s[i]);

@time optimize!(model)
@show objective_value(model)