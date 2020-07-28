 __precompile__()
using LinearAlgebra, Statistics
using Distributions, Plots, QuantEcon, Interpolations, Parameters

gr(fmt=:png);

w_max = 2
x = range(0,  w_max, length = 200)

G = Beta(3, 1.6)
F = Beta(1, 1)

# use key word argment
function SearchProblem(;β = 0.95, c = 0.6, F_a = 1, F_b = 1,
                       G_a = 3, G_b = 1.2, w_max = 2.0,
                       w_grid_size = 40, π_grid_size = 40)

    F = Beta(F_a, F_b)
    G = Beta(G_a, G_b)

    # scaled pdfs
    f(x) = pdf.(F, x/w_max)/w_max
    g(x) = pdf.(G, x/w_max)/w_max

    π_min = 1e-3  # avoids instability
    π_max = 1 - π_min

    w_grid = range(0,  w_max, length = w_grid_size)
    π_grid = range(π_min,  π_max, length = π_grid_size)

    nodes, weights = qnwlege(21, 0.0, w_max)

    return (β = β, c = c, F = F, G = G, f = f,
            g = g, n_w = w_grid_size, w_max = w_max,
            w_grid = w_grid, n_π = π_grid_size, π_min = π_min,
            π_max = π_max, π_grid = π_grid, quad_nodes = nodes,
            quad_weights = weights)
end

function q(sp, w, π_val)
    new_π = 1.0 / (1 + ((1 - π_val) * sp.g(w)) / (π_val * sp.f(w)))

    # Return new_π when in [π_min, π_max] and else end points
    return clamp(new_π, sp.π_min, sp.π_max)
end

function T!(sp, v, out;
                           ret_policy = false)
    # simplify names
    @unpack f, g, β, c = sp
    nodes, weights = sp.quad_nodes, sp.quad_weights

    vf = extrapolate(interpolate((sp.w_grid, sp.π_grid), v,
                    Gridded(Linear())), Flat())

    # set up quadrature nodes/weights
    # q_nodes, q_weights = qnwlege(21, 0.0, sp.w_max)

    for (w_i, w) in enumerate(sp.w_grid)
        # calculate v1
        v1 = w / (1 - β)

        for (π_j, _π) in enumerate(sp.π_grid)
            # calculate v2
            integrand(m) = [vf(m[i], q.(Ref(sp), m[i], _π)) *
                        (_π * f(m[i]) + (1 - _π) * g(m[i])) for i in 1:length(m)]
            integral = do_quad(integrand, nodes, weights)
            # integral = do_quad(integrand, q_nodes, q_weights)
            v2 = c + β * integral

            # return policy if asked for, otherwise return max of values
            out[w_i, π_j] = ret_policy ? v1 > v2 : max(v1, v2)
        end
    end
    return out
end

function T(sp, v;
                          ret_policy = false)
    out_type = ret_policy ? Bool : Float64
    out = zeros(out_type, sp.n_w, sp.n_π)
    T!(sp, v, out, ret_policy=ret_policy)
end


get_greedy!(sp, v, out) = T!(sp, v, out, ret_policy = true)

get_greedy(sp, v) = T(sp, v, ret_policy = true)

function res_wage_operator!(sp, ϕ, out)
    # simplify name
    @unpack f, g, β, c = sp

    # Construct interpolator over π_grid, given ϕ
    ϕ_f = LinearInterpolation(sp.π_grid, ϕ, extrapolation_bc = Line())

    # set up quadrature nodes/weights
    q_nodes, q_weights = qnwlege(7, 0.0, sp.w_max)

    for (i, _π) in enumerate(sp.π_grid)
        integrand(x) = max.(x, ϕ_f.(q.(Ref(sp), x, _π))) .* (_π * f(x) + (1 - _π) * g(x))
        integral = do_quad(integrand, q_nodes, q_weights)
        out[i] = (1 - β) * c + β * integral
    end
end

function res_wage_operator(sp, ϕ)
    out = similar(ϕ)
    res_wage_operator!(sp, ϕ, out)
    return out
end

# Set up the problem and initial guess, solve by VFI
sp = SearchProblem(;w_grid_size=100, π_grid_size=100)
v_init = fill(sp.c / (1 - sp.β), sp.n_w, sp.n_π)
f(x) = T(sp, x)
v = compute_fixed_point(f, v_init)
policy = get_greedy(sp, v)

# Make functions for the linear interpolants of these
vf = extrapolate(interpolate((sp.w_grid, sp.π_grid), v, Gridded(Linear())),
                Flat())
pf = extrapolate(interpolate((sp.w_grid, sp.π_grid), policy,
                Gridded(Linear())), Flat())

function plot_value_function(;w_plot_grid_size = 100,
                            π_plot_grid_size = 100)
  π_plot_grid = range(0.001,  0.99, length =  π_plot_grid_size)
  w_plot_grid = range(0,  sp.w_max, length = w_plot_grid_size)
  Z = [vf(w_plot_grid[j], π_plot_grid[i])
          for j in 1:w_plot_grid_size, i in 1:π_plot_grid_size]
  p = contour(π_plot_grid, w_plot_grid, Z, levels=15, alpha=0.6,
              fill=true, size=(400, 400), c=:lightrainbow)
  plot!(xlabel="pi", ylabel="w", xguidefont=font(12))
  return p
end

plot_value_function()

function plot_policy_function(;w_plot_grid_size = 100,
                              π_plot_grid_size = 100)
    π_plot_grid = range(0.001,  0.99, length = π_plot_grid_size)
    w_plot_grid = range(0,  sp.w_max, length = w_plot_grid_size)
    Z = [pf(w_plot_grid[j], π_plot_grid[i])
            for j in 1:w_plot_grid_size, i in 1:π_plot_grid_size]
    p = contour(π_plot_grid, w_plot_grid, Z, levels=1, alpha=0.6, fill=true,
                size=(400, 400), c=:coolwarm)
    plot!(xlabel="pi", ylabel="wage", xguidefont=font(12), cbar=false)
    annotate!(0.4, 1.0, "reject")
    annotate!(0.7, 1.8, "accept")
    return p
end

plot_policy_function()

# Determinism and random objects.
using Random
Random.seed!(42)

# Set up model and compute the function w̄
sp = SearchProblem(π_grid_size = 50, F_a = 1, F_b = 1)
ϕ_init = ones(sp.n_π)
g(x) = res_wage_operator(sp, x)
w̄_vals = compute_fixed_point(g, ϕ_init)
w̄ = extrapolate(interpolate((sp.π_grid, ), w̄_vals,
                    Gridded(Linear())), Flat())

# Holds the employment state and beliefs of an individual agent.
mutable struct Agent{TF <: AbstractFloat, TI <: Integer}
    _π::TF
    employed::TI
end

Agent(_π=1e-3) = Agent(_π, 1)

function update!(ag, H)
    if ag.employed == 0
        w = rand(H) * 2   # account for scale in julia
        if w ≥ w̄(ag._π)
            ag.employed = 1
        else
            ag._π = 1.0 ./ (1 .+ ((1 - ag._π) .* sp.g(w)) ./ (ag._π * sp.f(w)))
        end
    end
    nothing
end

num_agents = 5000
separation_rate = 0.025  # Fraction of jobs that end in each period
separation_num = round(Int, num_agents * separation_rate)
agent_indices = collect(1:num_agents)
agents = [Agent() for i=1:num_agents]
sim_length = 600
H = sp.G                 # Start with distribution G
change_date = 200        # Change to F after this many periods
unempl_rate = zeros(sim_length)

for i in 1:sim_length
    if i % 20 == 0
        println("date = $i")
    end

    if i == change_date
        H = sp.F
    end

    # Randomly select separation_num agents and set employment status to 0
    shuffle!(agent_indices)
    separation_list = agent_indices[1:separation_num]

    for agent in agents[separation_list]
        agent.employed = 0
    end

    # update agents
    for agent in agents
        update!(agent, H)
    end
    employed = Int[agent.employed for agent in agents]
    unempl_rate[i] = 1.0 - mean(employed)
end

plot(unempl_rate, linewidth = 2, label = "unemployment rate")
vline!([change_date], color = :red, label = "")
