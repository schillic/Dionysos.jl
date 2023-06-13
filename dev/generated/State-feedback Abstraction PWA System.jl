using StaticArrays, LinearAlgebra, Polyhedra, Random
using MathematicalSystems, HybridSystems
using JuMP, SDPA, Ipopt
using SemialgebraicSets, CDDLib
using Plots, Colors
using Test
Random.seed!(0)

using Dionysos
const DI = Dionysos
const UT = DI.Utils
const DO = DI.Domain
const ST = DI.System
const SY = DI.Symbolic
const CO = DI.Control
const PR = DI.Problem
const OP = DI.Optim
const AB = OP.Abstraction

opt_sdp = optimizer_with_attributes(SDPA.Optimizer, MOI.Silent() => true)
opt_ip = optimizer_with_attributes(Ipopt.Optimizer, MOI.Silent() => true)
lib = CDDLib.Library() # polyhedron lib
include("../../../problems/PWAsys.jl")

Usz = 70 # upper limit on |u|, `Usz = 50` in [1]
Wsz = 3 # `Wsz = 5` in [1]
dt = 0.01; # discretization step

concrete_problem =
    PWAsys.problem(; lib = lib, dt = dt, Usz = Usz, Wsz = Wsz, simple = false)
system = concrete_problem.system

n_step = 3
X_origin = SVector(0.0, 0.0);
X_step = SVector(1.0 / n_step, 1.0 / n_step)
nx = size(system.resetmaps[1].A, 1)
P = (1 / nx) * diagm((X_step ./ 2) .^ (-2))
state_grid = DO.GridEllipsoidalRectangular(X_origin, X_step, P, system.ext[:X]);

optimizer = MOI.instantiate(AB.EllipsoidsAbstraction.Optimizer)
MOI.set(optimizer, MOI.RawOptimizerAttribute("problem"), concrete_problem)
MOI.set(optimizer, MOI.RawOptimizerAttribute("state_grid"), state_grid)
MOI.set(optimizer, MOI.RawOptimizerAttribute("sdp_solver"), opt_sdp)
MOI.set(optimizer, MOI.RawOptimizerAttribute("ip_solver"), opt_ip)

using Suppressor
@suppress begin # this is a workaround to supress the undesired output of SDPA
    MOI.optimize!(optimizer)
end

abstract_system = MOI.get(optimizer, MOI.RawOptimizerAttribute("symmodel"))
abstract_problem = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_problem"))
abstract_controller = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_controller"))
concrete_controller = MOI.get(optimizer, MOI.RawOptimizerAttribute("concrete_controller"))
transitionCont = MOI.get(optimizer, MOI.RawOptimizerAttribute("transitionCont"))
transitionCost = MOI.get(optimizer, MOI.RawOptimizerAttribute("transitionCost"))
lyap_fun = MOI.get(optimizer, MOI.RawOptimizerAttribute("lyap_fun"));

#Return pwa mode for a given x
get_mode(x) = findfirst(m -> (x ∈ m.X), system.resetmaps)
function f_eval1(x, u)
    currState = SY.get_all_states_by_xpos(
        abstract_system,
        DO.crop_to_domain(abstract_system.Xdom, DO.get_all_pos_by_coord(state_grid, x)),
    )
    next_action = nothing
    for action in abstract_controller.data
        if (action[1] ∩ currState) ≠ []
            next_action = action
        end
    end
    c = DO.get_coord_by_pos(
        state_grid,
        SY.get_xpos_by_state(abstract_system, next_action[1]),
    )
    m = get_mode(c)
    W = system.ext[:W]
    w = (2 * (rand(2) .^ (1 / 4)) .- 1) .* W[:, 1]
    return system.resetmaps[m].A * x + system.resetmaps[m].B * u + system.resetmaps[m].c + w
end

function cost_eval(x, u)
    x_aug = vcat(x, u, 1.0)
    Q_aug = CO.get_full_psd_matrix(concrete_problem.transition_cost[1][1])
    return x_aug'Q_aug * x_aug
end

nstep = typeof(concrete_problem.time) == PR.Infinity ? 100 : concrete_problem.time; #max num of steps
function reached(x)
    currState = SY.get_all_states_by_xpos(
        abstract_system,
        DO.crop_to_domain(abstract_system.Xdom, DO.get_all_pos_by_coord(state_grid, x)),
    )
    if !isempty(currState ∩ abstract_problem.target_set)
        return true
    else
        return false
    end
end

x0 = concrete_problem.initial_set
x_traj, u_traj, cost_traj = CO.get_closed_loop_trajectory(
    f_eval1,
    concrete_controller,
    cost_eval,
    x0,
    nstep;
    stopping = reached,
)
cost_bound = AB.EllipsoidsAbstraction.get_guaranteed_cost(optimizer, x0)
cost_true = sum(cost_traj);
println("Goal set reached")
println("Guaranteed cost:\t $(cost_bound)")
println("True cost:\t\t $(cost_true)")

rectX = system.ext[:X];

fig = plot(;
    aspect_ratio = :equal,
    xtickfontsize = 10,
    ytickfontsize = 10,
    guidefontsize = 16,
    titlefontsize = 14,
);
xlims!(rectX.A.lb[1] - 0.2, rectX.A.ub[1] + 0.2);
ylims!(rectX.A.lb[2] - 0.2, rectX.A.ub[2] + 0.2);
xlabel!("\$x_1\$");
ylabel!("\$x_2\$");
title!("Specifictions and domains");
#We display the concrete domain
plot!(rectX; color = :yellow, opacity = 0.5);
#We display the abstract domain
plot!(abstract_system.Xdom; color = :blue, opacity = 0.5);
#We display the abstract specifications
plot!(
    SY.get_domain_from_symbols(abstract_system, abstract_problem.initial_set);
    color = :green,
    opacity = 0.5,
);
plot!(
    SY.get_domain_from_symbols(abstract_system, abstract_problem.target_set);
    color = :red,
    opacity = 0.5,
);
#We display the concrete specifications
plot!(UT.DrawPoint(concrete_problem.initial_set); color = :green, opacity = 1.0);
plot!(UT.DrawPoint(concrete_problem.target_set); color = :red, opacity = 1.0)

fig = plot(;
    aspect_ratio = :equal,
    xtickfontsize = 10,
    ytickfontsize = 10,
    guidefontsize = 16,
    titlefontsize = 14,
);
xlims!(rectX.A.lb[1] - 0.2, rectX.A.ub[1] + 0.2);
ylims!(rectX.A.lb[2] - 0.2, rectX.A.ub[2] + 0.2);
title!("Abstractions");
plot!(abstract_system; arrowsB = true, cost = false)

fig = plot(;
    aspect_ratio = :equal,
    xtickfontsize = 10,
    ytickfontsize = 10,
    guidefontsize = 16,
    titlefontsize = 14,
);
xlims!(rectX.A.lb[1] - 0.2, rectX.A.ub[1] + 0.2);
ylims!(rectX.A.lb[2] - 0.2, rectX.A.ub[2] + 0.2);
xlabel!("\$x_1\$");
ylabel!("\$x_2\$");
title!("Trajectory and Lyapunov-like Fun.");
plot!(abstract_system; arrowsB = false, cost = true, lyap_fun = lyap_fun);
plot!(UT.DrawTrajectory(x_traj); color = :black)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

