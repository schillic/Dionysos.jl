# # Example: Reachability problem solved by [Lazy abstraction](https://github.com/dionysos-dev/Dionysos.jl/blob/master/docs/src/manual/manual.md#solvers).
#
#md # [![Binder](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/Lazy-abstraction-reachability.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/Lazy-abstraction-reachability.ipynb)
#
# This is a **optimal reachability problem** for a **continuous system**.
#
#
# In order to study the concrete system and its symbolic abstraction in a unified framework, we will solve the problem
# for the sampled system with a sampling time $\tau$.
# For the construction of the relations in the abstraction, it is necessary to over-approximate attainable sets of
# a particular cell. In this example, we consider the used of a growth bound function  [1, VIII.2, VIII.5] which is one of the possible methods to over-approximate
# attainable sets of a particular cell based on the state reach by its center. 
#
# For this reachability problem, the abstraction controller is built using a solver that lazily builds the abstraction, constructing the abstraction 
# at the same time as the controller.

# First, let us import [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) and [Plots](https://github.com/JuliaPlots/Plots.jl).
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
    n = 2
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
concrete_system = concrete_problem.system

hx = [0.5, 0.5]
u0 = SVector(0.0, 0.0)
hu = SVector(0.5, 0.5)
Ugrid = DO.GridFree(u0, hu)
hx_heuristic = [1.0, 1.0] * 1.5
maxIter = 100

optimizer = MOI.instantiate(AB.LazyAbstraction.Optimizer)

AB.LazyAbstraction.set_optimizer!(
    optimizer,
    concrete_problem,
    maxIter,
    pre_image,
    post_image,
    compute_reachable_set,
    minimum_transition_cost,
    hx_heuristic,
    hx,
    Ugrid,
)

optimizer.param[:γ] = 0.6
# Build the abstraction and solve the optimal control problem using A* algorithm
using Suppressor
# @suppress begin
MOI.optimize!(optimizer)
# end

# Get the results
abstract_system = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_system"))
abstract_problem = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_problem"))
abstract_controller = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_controller"))
concrete_controller = MOI.get(optimizer, MOI.RawOptimizerAttribute("concrete_controller"))
abstract_lyap_fun = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_lyap_fun"))
concrete_lyap_fun = MOI.get(optimizer, MOI.RawOptimizerAttribute("concrete_lyap_fun"));

println(abstract_system.Udom)
println(DO.get_ncells(abstract_system.Udom))
println(abstract_system.Xdom)
# ## Simulation
# We define the cost and stopping criteria for a simulation
cost_eval(x, u) = UT.function_value(concrete_problem.transition_cost, x, u)
reached(x) = x ∈ concrete_problem.target_set
nstep = typeof(concrete_problem.time) == PR.Infinity ? 100 : concrete_problem.time; # max num of steps
# We simulate the closed loop trajectory
x0 = UT.get_center(concrete_problem.initial_set)
cost_control_trajectory = ST.get_closed_loop_trajectory(
    concrete_system.f_eval,
    concrete_controller,
    cost_eval,
    x0,
    nstep;
    stopping = reached,
    noise = false,
)
println(x0)

cost_bound = concrete_lyap_fun(x0)
cost_true = sum(cost_control_trajectory.costs.seq);
println("Goal set reached")
println("Guaranteed cost:\t $(cost_bound)")
println("True cost:\t\t $(cost_true)")




# # Display the results of the A* algorithm
xlims = (0, 60)
ylims = (0, 60)
 
 # Here we display the coordinate projection on the two first components of the state space along the trajectory.
fig = plot(; aspect_ratio = :equal,xlims=xlims, ylims=ylims);
plot!(framestyle=:box, grid=false, xtick=:auto, ytick=:auto, minorgrid=false, minorgridcolor=:grey, minorgridalpha=0.3,tickfontsize=9)
# We display the concrete domain
dark_grey = RGB(0.30, 0.30, 0.30) # RGB(0.83, 0.83, 0.83)  # Light grey color
light_grey = RGB(0.5, 0.5, 0.5) # RGB(0.83, 0.83, 0.83)  # Light grey color
super_light_grey = RGB(0.75, 0.75, 0.75) # RGB(0.83, 0.83, 0.83)  # Light grey color
# plot!(concrete_system.X; color = light_grey, opacity = 0.48);

plot!(
    optimizer.abstract_system_heuristic;
    arrowsB = false,
    dims = [1, 2],
    cost = true,
    lyap_fun = optimizer.bell_fun,
    label = false,
    colorMapStylde="Oranges",
)


plot!(concrete_problem.target_set; color = :red, opacity = 1.0);
plot!(concrete_problem.target_set; dims = [1, 2], color = :red, opacity = 0.6);
plot!(optimizer.lazy_search_problem;color1=dark_grey,color2=light_grey,color3=super_light_grey,opacity1=1.0, opacity2=1.0, opacity3=1.0, opacityI=1.0, opacityT=1.0)
plot!(concrete_problem.initial_set; color = :green, opacity = 1.0);


obstacles = [UT.HyperRectangle(SVector(22.0, 21.0), SVector(25.0, 32.0))]
for obs in obstacles
    plot!(obs; dims = [1, 2], color = :black,opacity=1.0)
end
plot!(cost_control_trajectory; ms = 2.0, lw=1.5, arrows=false, color=:blue, markerstrokecolor=:blue)
savefig(fig, "lazy_simple_system_abstraction_large.pdf")
display(fig)
