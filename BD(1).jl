using Distributions
using JuMP
using CPLEX

# Demand distribution
mu = [100.0, 200, 150, 170, 180, 170, 170]
sigma = [20.0, 50, 30, 50, 40, 30, 50]

# For each location, we discretize the levels into
# low, medium, high, using quantiles.
q = [
    quantile(Normal(mu[i], sigma[i]), [1/6, 3/6, 5/6])
    for i in 1:7
]

const d = vec(collect(Iterators.product(q...)))

# Number of locations
const N = 7

# Solve a subproblem given first stage decision s and demand realization d,
# and generate cut
function benders_cut(s, d)
    # holding cost 
    h = 1.0
    # shortage cost
    p = 4.0
    # transshipment cost
    c = 0.5

    # Sanity check
    @assert(length(s) == N)
    @assert(length(d) == N)

    # Build second stage model
    model = Model(CPLEX.Optimizer)

    # Suppress output for solving subproblems
    # to avoid cluttering the output
    set_silent(model)

    ij_set = [(i, j) for i=1:N for j=1:N if i!=j]
    @variables(model, begin
        e[1:N] >= 0
        f[1:N] >= 0
        q[1:N] >= 0
        r[1:N] >= 0
        t[ij_set] >= 0
    end)

    @objective(model, Min, h*sum(e) + c*sum(t) + p*sum(r))

    @constraints(model, begin
        B[i=1:N], f[i] + sum(t[(i,j)] for j=1:N if j!=i) + e[i] == s[i]
        M[i=1:N], f[i] + sum(t[(j,i)] for j=1:N if j!=i) + r[i] == d[i]
        R, sum(r) + sum(q) == sum(d)
        E[i=1:N], e[i] + q[i] == s[i]
    end)

    optimize!(model)

    # Get dual variables corresponding to that constraint
    pi_B = dual.(B)
    pi_M = dual.(M)
    pi_R = dual(R)
    pi_E = dual.(E)

    # Cut coefficients
    alpha = sum(d[i] * (pi_M[i] + pi_R) for i=1:N)
    beta = pi_B + pi_E
    
    obj = objective_value(model)
    dual_obj = alpha + beta'*s
    @assert(isapprox(obj, dual_obj), "obj=$obj, dual_obj=$dual_obj")
    
    return alpha, beta, dual_obj
end

# Given first stage decision s, go through each scenario
# to get an aggregated Benders cut.
function get_aggregate_cuts(s)
    a::Float64 = 0.0
    b::Vector{Float64} = zeros(N)
    d_obj::Float64 = 0.0

    for i in eachindex(d)
        alpha, beta, dual_obj = benders_cut(s, d[i])
        a += alpha / length(d)
        b += beta / length(d)
        d_obj += dual_obj / length(d)
    end

    return a, b, d_obj
end

# Master problem
model = Model(CPLEX.Optimizer)

# Some initial bounds on s
@variable(model, 80 <= s[1:N] <= 300)
@variable(model, eta >= 0)
@objective(model, Min, eta)
# Suppress output
set_silent(model)
optimize!(model)

# TODO: Print out solutions, objective value and number of iterations

# Maximum number of iterations
MAX_ITER = 100

# Stopping criteria
@time begin
    for iter = 1:MAX_ITER
        # ====================
        # Hint:
        # Solve master problem to get lower_bound and new first stage decision.
        # Call get_aggregate_cuts on current first stage decision
        # to get alpha, beta, and upper_bound.
        # Add a new cut to the master problem.
        # TODO: fill in
        optimize!(model)
        lower_bound = objective_value(model)
        alpha, beta, upper_bound = get_aggregate_cuts(value.(s))

        println(iter)
        @show value.(lower_bound)
        @show value.(upper_bound)
        @show value.(s)
        @show objective_value(model)
        # =====================
        # Recommended: Display upper_bound and lower_bound

        # =====================
        # add new cut on master problem
        @constraint(model,eta>=alpha + beta'*s)
        # Stopping criteria
        if upper_bound - lower_bound < 1e-6
            println("Terminating with the optimal solution")
            break
        end
    end
end