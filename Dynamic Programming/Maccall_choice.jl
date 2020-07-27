using Distributions, LinearAlgebra, Expectations, Parameters, NLsolve, Plots

function solve_mccall_model(mcm; U_iv = 1.0, V_iv = ones(length(mcm.w)), tol = 1e-5,
                            iter = 2_000)
    # α, β, σ, c, γ, w = mcm.α, mcm.β, mcm.σ, mcm.c, mcm.γ, mcm.w
    @unpack α, β, σ, c, γ, w, dist, u = mcm

    # parameter validation
    @assert c > 0.0
    @assert minimum(w) > 0.0 # perhaps not strictly necessary, but useful here

    # necessary objects
    u_w = u.(w, σ)
    u_c = u(c, σ)
    E = expectation(dist) # expectation operator for wage distribution

    # Bellman operator T. Fixed point is x* s.t. T(x*) = x*
    function T(x)
        V = x[1:end-1]
        U = x[end]
        [u_w + β * ((1 - α) * V .+ α * U); u_c + β * (1 - γ) * U + β * γ * E * max.(U, V)]
    end

    # value function iteration
    x_iv = [V_iv; U_iv] # initial x val
    xstar = fixedpoint(T, x_iv, iterations = iter, xtol = tol, m = 0).zero
    V = xstar[1:end-1]
    U = xstar[end]

    # compute the reservation wage
    wbarindex = searchsortedfirst(V .- U, 0.0)
    if wbarindex >= length(w) # if this is true, you never want to accept
        w̄ = Inf
    else
        w̄ = w[wbarindex] # otherwise, return the number
    end

    # return a NamedTuple, so we can select values by name
    return (V = V, U = U, w̄ = w̄)
end


# a default utility function
u(c, σ) = (c^(1 - σ) - 1) / (1 - σ)

# model constructor
McCallModel = @with_kw (α = 0.2,
    β = 0.98, # discount rate
    γ = 0.7,
    c = 6.0, # unemployment compensation
    σ = 2.0,
    u = u, # utility function
    w = range(10, 20, length = 60), # wage values
    dist = BetaBinomial(59, 600, 400)) # distribution over wage values

# plots setting
gr(fmt=:png);

mcm = McCallModel()
@unpack V, U = solve_mccall_model(mcm)
U_vec = fill(U, length(mcm.w))

plot(mcm.w, [V U_vec], lw = 2, α = 0.7, label = ["V" "U"])

c_vals = range(2,  12, length = 25)

models = [McCallModel(c = cval) for cval in c_vals]
sols = solve_mccall_model.(models)
w̄_vals = [sol.w̄ for sol in sols]

plot(c_vals,
    w̄_vals,
    lw = 2,
    α = 0.7,
    xlabel = "unemployment compensation",
    ylabel = "reservation wage",
    label = "w̄ as a function of c")

w̄_vals = [solve_mccall_model(McCallModel(c = cval)).w̄ for cval in c_vals];
# doesn't allocate new arrays for models and solutions

γ_vals = range(0.05,  0.95, length = 25)

models = [McCallModel(γ = γval) for γval in γ_vals]
sols = solve_mccall_model.(models)
w̄_vals = [sol.w̄ for sol in sols]

plot(γ_vals, w̄_vals, lw = 2, α = 0.7, xlabel = "job offer rate",
     ylabel = "reservation wage", label = "w̄ as a function of gamma")
