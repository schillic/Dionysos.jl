using Test     #src
# # Example: Path planning problem solved by [Uniform grid abstraction](https://github.com/dionysos-dev/Dionysos.jl/blob/master/docs/src/manual/manual.md#solvers).
#
#md # [![Binder](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/Path planning.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/Path planning.ipynb)
#
# This example was borrowed from [1, IX. Examples, A] whose dynamics comes from the model given in [2, Ch. 2.4].
# This is a **reachability problem** for a **continuous system**.
#
# Let us consider the 3-dimensional state space control system of the form
# ```math
# \dot{x} = f(x, u)
# ```
# with $f: \mathbb{R}^3 × U ↦ \mathbb{R}^3$ given by
# ```math
# f(x,(u_1,u_2)) = \begin{bmatrix} u_1 \cos(α+x_3)\cos(α^{-1}) \\ u_1 \sin(α+x_3)\cos(α^{-1}) \\ u_1 \tan(u_2)  \end{bmatrix}
# ```
# and with $U = [−1, 1] \times [−1, 1]$ and $α = \arctan(\tan(u_2)/2)$. Here, $(x_1, x_2)$ is the position and $x_3$ is the
# orientation of the vehicle in the 2-dimensional plane. The control inputs $u_1$ and $u_2$ are the rear
# wheel velocity and the steering angle.
# The control objective is to drive the vehicle which is situated in a maze made of obstacles from an initial position to a target position.
#
#
# In order to study the concrete system and its symbolic abstraction in a unified framework, we will solve the problem
# for the sampled system with a sampling time $\tau$.
# For the construction of the relations in the abstraction, it is necessary to over-approximate attainable sets of
# a particular cell. In this example, we consider the used of a growth bound function  [1, VIII.2, VIII.5] which is one of the possible methods to over-approximate
# attainable sets of a particular cell based on the state reach by its center.
#
# For this reachability problem, the abstraction controller is built by solving a fixed-point equation which consists in computing the pre-image
# of the target set.

# First, let us import [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) and [Plots](https://github.com/JuliaPlots/Plots.jl).
using StaticArrays, Plots

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

# And the file defining the hybrid system for this problem
include(joinpath(dirname(dirname(pathof(Dionysos))), "problems", "path_planning.jl"))

# ### Definition of the problem

# Now we instantiate the problem using the function provided by [PathPlanning.jl](@__REPO_ROOT_URL__/problems/PathPlanning.jl) 
concrete_problem = PathPlanning.problem(; simple = false, approx_mode = PathPlanning.GROWTH);
concrete_system = concrete_problem.system;

# ### Definition of the abstraction

# Definition of the grid of the state-space on which the abstraction is based (origin `x0` and state-space discretization `h`):
x0 = SVector(0.0, 0.0, 0.0);
h = SVector(0.2, 0.2, 0.2);
state_grid = DO.GridFree(x0, h);

# Definition of the grid of the input-space on which the abstraction is based (origin `u0` and input-space discretization `h`):
u0 = SVector(0.0, 0.0);
h = SVector(0.3, 0.3);
input_grid = DO.GridFree(u0, h);

# We now solve the optimal control problem with the `Abstraction.UniformGridAbstraction.Optimizer`.

using JuMP
optimizer = MOI.instantiate(AB.UniformGridAbstraction.Optimizer)
MOI.set(optimizer, MOI.RawOptimizerAttribute("concrete_problem"), concrete_problem)
MOI.set(optimizer, MOI.RawOptimizerAttribute("state_grid"), state_grid)
MOI.set(optimizer, MOI.RawOptimizerAttribute("input_grid"), input_grid)
MOI.optimize!(optimizer)

# Get the results
abstract_system = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_system"))
abstract_problem = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_problem"))
abstract_controller = MOI.get(optimizer, MOI.RawOptimizerAttribute("abstract_controller"))
concrete_controller = MOI.get(optimizer, MOI.RawOptimizerAttribute("concrete_controller"))

# @test length(abstract_controller.data) == 19400 #src

# ### Trajectory display
# We choose a stopping criterion `reached` and the maximal number of steps `nsteps` for the sampled system, i.e. the total elapsed time: `nstep`*`tstep`
# as well as the true initial state `x0` which is contained in the initial state-space `_I_` defined previously.
nstep = 1000
function reached(x)
    if x ∈ concrete_problem.target_set
        return true
    else
        return false
    end
end
x0 = SVector(0.4, 0.4, 0.0)
control_trajectory = ST.get_closed_loop_trajectory(
    concrete_system.f,
    concrete_controller,
    x0,
    nstep;
    stopping = reached,
)

function get_obstacles(
    _X_;
    X1_lb = [1.0, 2.2, 2.2, 3.4, 4.6, 5.8, 5.8, 7.0, 8.2, 8.2, 9.3, 8.4, 9.3, 8.4, 9.3],
    X1_ub = [1.2, 2.4, 2.4, 3.6, 4.8, 6.0, 6.0, 7.2, 8.4, 9.3, 10.0, 9.3, 10.0, 9.3, 10.0],
    X2_lb = [0.0, 0.0, 6.0, 0.0, 1.0, 0.0, 7.0, 1.0, 0.0, 8.2, 7.0, 5.8, 4.6, 3.4, 2.2],
    X2_ub = [9.0, 5.0, 10.0, 9.0, 10.0, 6.0, 10.0, 10.0, 8.5, 8.6, 7.4, 6.2, 5.0, 3.8, 2.6],
)
    return [
        UT.HyperRectangle(SVector(x1lb, x2lb, _X_.lb[3]), SVector(x1ub, x2ub, _X_.ub[3]))
        for (x1lb, x2lb, x1ub, x2ub) in zip(X1_lb, X2_lb, X1_ub, X2_ub)
    ]
end

# Define plot limits
xlims = (0, 10)
ylims = (0, 10)


# Here we display the coordinate projection on the two first components of the state space along the trajectory.
fig = plot(; aspect_ratio = :equal,xlims=xlims, ylims=ylims);
plot!(framestyle=:box, grid=false, xtick=:auto, ytick=:auto, minorgrid=false, minorgridcolor=:grey, minorgridalpha=0.3,tickfontsize=9)
# We display the concrete domain
light_grey = RGB(0.6, 0.6, 0.6) # RGB(0.83, 0.83, 0.83)  # Light grey color
plot!(concrete_system.X; color = light_grey, opacity = 0.5);

# We display the abstract domain
plot!(abstract_system.Xdom; color = light_grey, opacity = 0.5);



# We display the abstract specifications
plot!(
    SY.get_domain_from_symbols(abstract_system, abstract_problem.initial_set);
    color = :green,opacity=0.2,
);
# We display the concrete specifications
plot!(concrete_problem.initial_set; color = :green, opacity = 1.0);

# We display the concrete trajectory
plot!(control_trajectory; ms = 2.0, lw=1.5, arrows=false, color=:blue, markerstrokecolor=:blue)

plot!(
    SY.get_domain_from_symbols(abstract_system, abstract_problem.initial_set);
    color = :green,opacity=1.0
);

plot!(concrete_problem.target_set; dims = [1, 2], color = :red, opacity = 0.6);
plot!(
    SY.get_domain_from_symbols(abstract_system, abstract_problem.target_set);
    color = :red,opacity=1.0
);
_X_ = UT.HyperRectangle(SVector(0.0, 0.0, -pi - 0.4), SVector(4.0, 10.0, pi + 0.4))
obstacles = get_obstacles(_X_)
for obs in get_obstacles(_X_)
    plot!(obs; dims = [1, 2], color = :black,opacity=1.0)
end
savefig(fig, "path_planning_problem.pdf")
display(fig)
# ### References
# 1. G. Reissig, A. Weber and M. Rungger, "Feedback Refinement Relations for the Synthesis of Symbolic Controllers," in IEEE Transactions on Automatic Control, vol. 62, no. 4, pp. 1781-1796.
# 2. K. J. Aström and R. M. Murray, Feedback systems. Princeton University Press, Princeton, NJ, 2008.
