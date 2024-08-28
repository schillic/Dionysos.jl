# # Example: Reachability problem solved by [Hierarchical abstraction](https://github.com/dionysos-dev/Dionysos.jl/blob/master/docs/src/manual/manual.md#solvers).
#
#md # [![Binder](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/Hierarchical-abstraction.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/Hierarchical-abstraction.ipynb)
#

using StaticArrays, JuMP, Plots

# At this point, we import Dionysos.
using Dionysos
const DI = Dionysos
const UT = DI.Utils
const DO = DI.Domain
const ST = DI.System
const SY = DI.Symbolic
const PR = DI.Problem
const OP = DI.Optim
const AB = OP.Abstraction

include(joinpath(dirname(dirname(pathof(Dionysos))), "problems", "simple_problem.jl"))

## specific functions
function post_image(abstract_system, concrete_system, xpos, u)
    Xdom = abstract_system.Xdom
    x = DO.get_coord_by_pos(Xdom.grid, xpos)
    Fx = concrete_system.f_eval(x, u)
    r = Xdom.grid.h / 2.0 + concrete_system.measnoise
    Fr = r

    rectI = DO.get_pos_lims_outer(Xdom.grid, UT.HyperRectangle(Fx .- Fr, Fx .+ Fr))
    ypos_iter = Iterators.product(DO._ranges(rectI)...)
    over_approx = []
    allin = true
    for ypos in ypos_iter
        ypos = DO.set_in_period_pos(Xdom, ypos)
        if !(ypos in Xdom)
            allin = false
            break
        end
        target = SY.get_state_by_xpos(abstract_system, ypos)[1]
        push!(over_approx, target)
    end
    return allin ? over_approx : []
end

function pre_image(abstract_system, concrete_system, xpos, u)
    grid = abstract_system.Xdom.grid
    x = DO.get_coord_by_pos(grid, xpos)
    potential = Int[]
    x_prev = concrete_system.f_backward(x, u)
    xpos_cell = DO.get_pos_by_coord(grid, x_prev)
    n = 1
    for i in (-n):n
        for j in (-n):n
            x_n = (xpos_cell[1] + i, xpos_cell[2] + j)
            x_n = DO.set_in_period_pos(abstract_system.Xdom, x_n)
            if x_n in abstract_system.Xdom
                cell = SY.get_state_by_xpos(abstract_system, x_n)[1]
                if !(cell in potential)
                    push!(potential, cell)
                end
            end
        end
    end
    return potential
end

function compute_reachable_set(rect::UT.HyperRectangle, concrete_system, Udom)
    r = (rect.ub - rect.lb) / 2.0 + concrete_system.measnoise
    Fr = r
    x = UT.get_center(rect)
    n = UT.get_dims(rect)
    lb = fill(Inf, n)
    ub = fill(-Inf, n)
    for upos in DO.enum_pos(Udom)
        u = DO.get_coord_by_pos(Udom.grid, upos)
        Fx = concrete_system.f_eval(x, u)
        lb = min.(lb, Fx .- Fr)
        ub = max.(ub, Fx .+ Fr)
    end
    lb = SVector{n}(lb)
    ub = SVector{n}(ub)
    return UT.HyperRectangle(lb, ub)
end

minimum_transition_cost(symmodel, contsys, source, target) = 1.0

#[UT.HyperRectangle(SVector(22.0, 21.0), SVector(25.0, 32.0))],
concrete_problem = SimpleProblem.problem(;
    rectX = UT.HyperRectangle(SVector(0.0, 0.0), SVector(60.0, 60.0)),
    obstacles = [UT.HyperRectangle(SVector(22.0, 21.0), SVector(25.0, 32.0))],
    periodic = Int[],
    periods = [30.0, 30.0],
    T0 = [0.0, 0.0],
    rectU = UT.HyperRectangle(SVector(-2.0, -2.0), SVector(2.0, 2.0)), # UT.HyperRectangle(SVector(-2.0, -2.0), SVector(2.0, 2.0)),
    Uobstacles = [UT.HyperRectangle(SVector(-0.5, -0.5), SVector(0.5, 0.5))],
    _I_ = UT.HyperRectangle(SVector(6.5, 6.5), SVector(7.5, 7.5)),
    _T_ = UT.HyperRectangle(SVector(44.0, 44.0), SVector(49.0, 49.0)),
    state_cost = UT.ZeroFunction(),
    transition_cost = UT.ConstantControlFunction(0.5),
    tstep = 0.8,
    measnoise = SVector(0.0, 0.0),
)

concrete_system = concrete_problem.system;

# Local optimizer parameters
hx_local = [0.5, 0.5]
hx_heuristic = [1.0, 1.0]*1.0
u0 = SVector(0.0, 0.0)
hu = SVector(0.5, 0.5)
Ugrid = DO.GridFree(u0, hu)

local_optimizer = MOI.instantiate(AB.LazyAbstraction.Optimizer)

AB.LazyAbstraction.set_optimizer_parameters!(
    local_optimizer,
    100,
    pre_image,
    post_image,
    compute_reachable_set,
    minimum_transition_cost,
    hx_local,
    hx_heuristic;
    γ = 10.0,
)

# Global optimizer parameters
hx_global = [10.0, 10.0]
u0 = SVector(0.0, 0.0)
hu = SVector(0.5, 0.5)
Ugrid = DO.GridFree(u0, hu)
max_iter = 6
max_time = 1000

optimizer = MOI.instantiate(AB.HierarchicalAbstraction.Optimizer)

AB.HierarchicalAbstraction.set_optimizer!(
    optimizer,
    concrete_problem,
    hx_global,
    Ugrid,
    compute_reachable_set,
    minimum_transition_cost,
    local_optimizer,
    max_iter,
    max_time;
    option = true,
)

# using Suppressor
# @suppress begin
MOI.optimize!(optimizer)
# end

# Get the results
abstract_system = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_system"))

println("Solved : ", optimizer.solved)

# ## Simulation
x0 = UT.get_center(concrete_problem.initial_set)
cost_control_trajectory =
    AB.HierarchicalAbstraction.get_closed_loop_trajectory(optimizer, x0)
cost = sum(cost_control_trajectory.costs.seq);
println(
    "Goal set reached: $(ST.get_state(cost_control_trajectory, ST.length(cost_control_trajectory))∈concrete_problem.target_set)",
)
println("Cost:\t $(cost)")

# # Display the results of the A* algorithm
xlims = (0, 60)
ylims = (0, 60)
 
 # Here we display the coordinate projection on the two first components of the state space along the trajectory.
fig = plot(; aspect_ratio = :equal,xlims=xlims, ylims=ylims);
plot!(framestyle=:box, grid=false, xtick=:auto, ytick=:auto, minorgrid=false, minorgridcolor=:grey, minorgridalpha=0.3,tickfontsize=9)

#We display the concrete domain
plot!(concrete_system.X; color = :yellow, opacity = 0.5);

#We display the abstract domain
plot!(abstract_system.symmodel.Xdom; color = :blue, opacity = 0.5);

#We display the concrete specifications
plot!(concrete_problem.initial_set; color = :green, opacity = 0.8);
plot!(concrete_problem.target_set; dims = [1, 2], color = :red, opacity = 0.8);

#We display the concrete trajectory
plot!(cost_control_trajectory; ms = 0.5)
display(fig)

############################################################
############################################################
############################################################

xlims = (0, 60)
ylims = (0, 60)
 # Here we display the coordinate projection on the two first components of the state space along the trajectory.
fig = plot(; aspect_ratio = :equal,xlims=xlims, ylims=ylims);
plot!(framestyle=:box, grid=false, xtick=:auto, ytick=:auto, minorgrid=false, minorgridcolor=:grey, minorgridalpha=0.3,tickfontsize=9)
dark_grey = RGB(0.30, 0.30, 0.30) # RGB(0.83, 0.83, 0.83)  # Light grey color
light_grey = RGB(0.5, 0.5, 0.5) # RGB(0.83, 0.83, 0.83)  # Light grey color
super_light_grey = RGB(0.75, 0.75, 0.75) # RGB(0.83, 0.83, 0.83)  # Light grey color
# plot!(concrete_system.X; color = light_grey, opacity = 0.48);

# plot!(
#     optimizer.abstract_system_heuristic;
#     arrowsB = false,
#     dims = [1, 2],
#     cost = true,
#     lyap_fun = optimizer.bell_fun,
#     label = false,
# )


plot!(concrete_problem.target_set; color = :red, opacity = 1.0);
plot!(concrete_problem.target_set; dims = [1, 2], color = :red, opacity = 0.6);
plot!(
    optimizer.hierarchical_problem;
    path = optimizer.optimizer_BB.best_sol,
    heuristic = true,
    fine = true,
    color1=dark_grey,color2=light_grey,color3=super_light_grey,opacity1=1.0, opacity2=1.0, opacity3=1.0, opacityI=1.0, opacityT=1.0
)
obstacles = [UT.HyperRectangle(SVector(22.0, 21.0), SVector(25.0, 32.0))]
for obs in obstacles
    plot!(obs; dims = [1, 2], color = :black,opacity=1.0)
end
plot!(concrete_problem.initial_set; color = :green, opacity = 1.0);
plot!(cost_control_trajectory; ms = 2.0, lw=1.5, arrows=false, color=:blue, markerstrokecolor=:blue)
savefig(fig, "hierarchical_simple_system_abstraction_large.pdf")
println(x0)
display(fig)
