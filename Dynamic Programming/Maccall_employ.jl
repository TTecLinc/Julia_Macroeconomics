using LinearAlgebra, Statistics
using Distributions, Expectations, NLsolve, Roots, Random, Plots, Parameters
using StatsPlots
gr(fmt = :png);

n = 50
dist = BetaBinomial(n, 200, 100) # probability distribution
@show support(dist)
w = range(10.0, 60.0, length = n+1) # linearly space wages
plot(w, pdf.(dist, support(dist)), xlabel = "wages", ylabel = "probabilities", legend = false)

#-----------------------------------------------------------------------------------------------------

E = expectation(dist) # expectation operator

# exploring the properties of the operator
wage(i) = w[i+1] # +1 to map from support of 0
E_w = E(wage)
E_w_2 = E(i -> wage(i)^2) - E_w^2 # variance
@show E_w, E_w_2

# use operator with left-multiply
@show E * w # the `w` are values assigned for the discrete states
@show dot(pdf.(dist, support(dist)), w); # identical calculation

#-----------------------------------------------------------------------
# parameters and constant objects

c = 25
β = 0.99
num_plots = 6

# Operator
T(v) = max.(w/(1 - β), c + β * E*v) # (5) broadcasts over the w, fixes the v
# alternatively, T(v) = [max(wval/(1 - β), c + β * E*v) for wval in w]

# fill in  matrix of vs
vs = zeros(n + 1, 6) # data to fill
vs[:, 1] .= w / (1-β) # initial guess of "accept all"

# manually applying operator
for col in 2:num_plots
    v_last = vs[:, col - 1]
    vs[:, col] .= T(v_last)  # apply operator
end
plot(vs)


function compute_reservation_wage_direct(params; v_iv = collect(w ./(1-β)), max_iter = 500,
                                         tol = 1e-6)
    @unpack c, β, w = params

    # create a closure for the T operator
    T(v) = max.(w/(1 - β), c + β * E*v) # (5) fixing the parameter values

    v = copy(v_iv) # start at initial value.  copy to prevent v_iv modification
    v_next = similar(v)
    i = 0
    error = Inf
    while i < max_iter && error > tol
        v_next .= T(v) # (4)
        error = norm(v_next - v)
        i += 1
        v .= v_next  # copy contents into v.  Also could have used v[:] = v_next
    end
    # now compute the reservation wage
    return (1 - β) * (c + β * E*v) # (2)
end

function compute_reservation_wage(params; v_iv = collect(w ./(1-β)), iterations = 500,
                                  ftol = 1e-6, m = 6)
    @unpack c, β, w = params
    T(v) = max.(w/(1 - β), c + β * E*v) # (5) fixing the parameter values

    v_star = fixedpoint(T, v_iv, iterations = iterations, ftol = ftol,
                        m = 0).zero # (5)
    return (1 - β) * (c + β * E*v_star) # (3)
end

mcm = @with_kw (c=25.0, β=0.99, w=w) # named tuples

compute_reservation_wage(mcm()) # call with default parameters

grid_size = 25
R = rand(grid_size, grid_size)

c_vals = range(10.0, 30.0, length = grid_size)
β_vals = range(0.9, 0.99, length = grid_size)

for (i, c) in enumerate(c_vals)
    for (j, β) in enumerate(β_vals)
        R[i, j] = compute_reservation_wage(mcm(c=c, β=β)) # change from defaults
    end
end

contour(c_vals, β_vals, R',
        title = "Reservation Wage",
        xlabel = "c",
        ylabel = "beta",
        fill = true)

        function compute_reservation_wage_ψ(c, β; ψ_iv = E * w ./ (1 - β), max_iter = 500,
                                            tol = 1e-5)
            T_ψ(ψ) = [c + β * E*max.((w ./ (1 - β)), ψ[1])] # (7)
            # using vectors since fixedpoint doesn't support scalar
            ψ_star = fixedpoint(T_ψ, [ψ_iv]).zero[1]
            return (1 - β) * ψ_star # (2)
        end
        compute_reservation_wage_ψ(c, β)

        #-----------------------------------------------------------------------------------------------------------
        function compute_stopping_time(w̄; seed=1234)
            Random.seed!(seed)
            stopping_time = 0
            t = 1
            # make sure the constraint is sometimes binding
            @assert length(w) - 1 ∈ support(dist) && w̄ <= w[end]
            while true
                # Generate a wage draw
                w_val = w[rand(dist)] # the wage dist set up earlier
                if w_val ≥ w̄
                    stopping_time = t
                    break
                else
                    t += 1
                end
            end
            return stopping_time
        end

        compute_mean_stopping_time(w̄, num_reps=10000) = mean(i ->
                                                                 compute_stopping_time(w̄,
                                                                 seed = i), 1:num_reps)
        c_vals = range(10,  40, length = 25)
        stop_times = similar(c_vals)

        beta = 0.99
        for (i, c) in enumerate(c_vals)
            w̄ = compute_reservation_wage_ψ(c, beta)
            stop_times[i] = compute_mean_stopping_time(w̄)
        end

        plot(c_vals, stop_times, label = "mean unemployment duration",
             xlabel = "unemployment compensation", ylabel = "months")
