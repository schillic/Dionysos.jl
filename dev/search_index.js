var documenterSearchIndex = {"docs":
[{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"EditURL = \"https://github.com/dionysos-dev/Dionysos.jl/blob/master/docs/src/examples/Gol, Lazar & Belta (2013).jl\"","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/#Example:-Gol,-Lazar-and-Belta-(2013)","page":"Gol, Lazar & Belta (2013)","title":"Example: Gol, Lazar and Belta (2013)","text":"","category":"section"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"(Image: Binder) (Image: nbviewer)","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"This example was borrowed from [1, Example VIII.A] and tackles an optimal control for the hybrid system with state evolution governed by","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"x(k+1) = beginbmatrix 1  1  0  1 endbmatrixx(k) + beginbmatrix 05  10 endbmatrix u(k)","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"The goal is to take the state vector toward a target set XT by visiting one of the squares A or B and avoiding the obstacles O1 and O2","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"First, let us import CDDLib, GLPK, OSQP, JuMP, Pavito and Ipopt","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"import CDDLib\nimport GLPK\nimport OSQP\nusing JuMP\nimport Pavito\nimport Cbc\nimport Ipopt","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"At this point we import Dionysos","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"using Dionysos","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"And the file defining the hybrid system for this problem","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"include(joinpath(dirname(dirname(pathof(Dionysos))), \"examples\", \"gol_lazar_belta.jl\"))","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"Now we instantiate our system using the function provided by gol_lazar_belta.jl","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"system = gol_lazar_belta(CDDLib.Library(), Float64);\nnothing #hide","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"Then, we define initial conditions (continuous and discrete states) to this system and set N as the search depth, i.e., the number of allowed time steps.","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"x0 = [1.0, -6.0];\nq0 = 3;\n\nN = 11;\nnothing #hide","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"We instantiate our Optimal Control Problem by defining the state and transition costs. Notice that Comment that state_cost is defined to be zero for each mode/discrete state of the system and the transition_cost is defined to be u_1^2 which is defined by the quadratic form u' * Q * u with Q = ones(1, 1).","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"state_cost = Fill(ZeroFunction(), nmodes(system))\ntransition_cost = QuadraticControlFunction(ones(1, 1))\n\nproblem = OptimalControlProblem(\n    system,\n    q0, x0,\n    Fill(state_cost, N),\n    Fill(Fill(transition_cost, ntransitions(system)), N),\n    system.ext[:q_T],\n    N\n);\nnothing #hide","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"Notice that we used Fill for all N time steps as we consider time-invariant costs. Finally, we select the method presented in [2] as our optimizer","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"qp_solver = optimizer_with_attributes(\n    OSQP.Optimizer,\n    \"eps_abs\" => 1e-8,\n    \"eps_rel\" => 1e-8,\n    \"max_iter\" => 100000,\n    MOI.Silent() => true\n);\n\nmip_solver = optimizer_with_attributes(\n    Cbc.Optimizer,\n    MOI.Silent() => true\n);\n\ncont_solver = optimizer_with_attributes(\n    Ipopt.Optimizer,\n    MOI.Silent() => true\n);\n\nmiqp_solver = optimizer_with_attributes(\n    Pavito.Optimizer,\n    \"mip_solver\" => mip_solver,\n    \"cont_solver\" => cont_solver,\n    MOI.Silent() => true\n);\n\n\nalgo = optimizer_with_attributes(BemporadMorari.Optimizer{Float64},\n    \"continuous_solver\" => qp_solver,\n    \"mixed_integer_solver\" => miqp_solver,\n    \"indicator\" => false,\n    \"log_level\" => 0\n);\nnothing #hide","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"and use it to solve the given problem, with the help of the abstraction layer MathOptInterface provided by JuMP","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"optimizer = MOI.instantiate(algo)\nMOI.set(optimizer, MOI.RawParameter(\"problem\"), problem)\nMOI.optimize!(optimizer)","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"We check the solver time","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"MOI.get(optimizer, MOI.SolveTime())","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"the termination status","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"termination = MOI.get(optimizer, MOI.TerminationStatus())","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"the objective value","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"objective_value = MOI.get(optimizer, MOI.ObjectiveValue())","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"and recover the corresponding continuous trajectory","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"xu = MOI.get(optimizer, ContinuousTrajectoryAttribute());\nnothing #hide","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"A little bit of data visualization now:","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"using Plots\nusing Colors\n\n##Auxiliary function for annotating\nfunction text_in_set_plot!(pl, po, t; kws...)\n    ##solve finding center (other solvers? https://jump.dev/JuMP.jl/dev/installation/#Supported-solvers)\n    solver = optimizer_with_attributes(GLPK.Optimizer, \"presolve\" => GLPK.ON)\n    plot!(pl, po; kws...)\n    if t !== nothing\n        c, r = hchebyshevcenter(hrep(po), solver, verbose=0)\n        annotate!(pl, [(c..., text(t, 12))])\n    end\nend\n\n##Initialize our canvas\np = Plots.plot(fmt = :png, fillcolor = :white)\n\n##Show the discrete modes\nfor mode in states(system)\n    t = (system.ext[:q_T] in [mode, mode + 11]) ? \"XT\" : (mode == system.ext[:q_A] ? \"A\" : (mode == system.ext[:q_B] ? \"B\" :\n            mode <= 11 ? string(mode) : string(mode - 11)))\n    text_in_set_plot!(p, stateset(system, mode), t, fillcolor = :white, linecolor = :black)\nend\n\n##Plot obstacles\nfor i in eachindex(system.ext[:obstacles])\n    text_in_set_plot!(p, system.ext[:obstacles][i], \"O$i\", fillcolor = :black, fillalpha = 0.1)\nend\n\n\n##Initial state\nscatter!(p, [x0[1]], [x0[2]])\nannotate!(p, [(x0[1], x0[2] - 0.5, text(\"x0\", 10))])\n\n##Split the vector into x1 and x2\nx1 = [xu.x[j][1] for j in eachindex(xu.x)]\nx2 = [xu.x[j][2] for j in eachindex(xu.x)]\n\n##Plot the trajectory\nscatter!(p, x1, x2)","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/#References","page":"Gol, Lazar & Belta (2013)","title":"References","text":"","category":"section"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"Gol, E. A., Lazar, M., & Belta, C. (2013). Language-guided controller synthesis for linear systems. IEEE Transactions on Automatic Control, 59(5), 1163-1176.\nBemporad, A., & Morari, M. (1999). Control of systems integrating logic, dynamics, and constraints. Automatica, 35(3), 407-427.","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"","category":"page"},{"location":"generated/Gol, Lazar & Belta (2013)/","page":"Gol, Lazar & Belta (2013)","title":"Gol, Lazar & Belta (2013)","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"EditURL = \"https://github.com/dionysos-dev/Dionysos.jl/blob/master/docs/src/examples/DC-DC converter.jl\"","category":"page"},{"location":"generated/DC-DC converter/#Example:-DC-DC-converter","page":"DC-DC converter","title":"Example: DC-DC converter","text":"","category":"section"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"(Image: Binder) (Image: nbviewer)","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"We consider a boost DC-DC converter which has been widely studied from the point of view of hybrid control, see for example in  [1, V.A],[2],[3]. This is a safety problem for a switching system.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"(Image: Boost DC-DC converter.)","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"The state of the system is given by x(t) = beginbmatrix i_l(t)  v_c(t) endbmatrix^top. The switching system has two modes consisting in two-dimensional affine dynamics:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"dotx = f_p(x) = A_p x + b_pquad p=12","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"with","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"A_1 = beginbmatrix -fracr_lx_l 0  0  -frac1x_cfrac1r_0+r_c  endbmatrix A_2= beginbmatrix -frac1x_lleft(r_l+fracr_0r_cr_0+r_cright)  -frac1x_lfracr_0r_0+r_c   frac1x_cfracr_0r_0+r_c    -frac1x_cfrac1r_0+r_c  endbmatrix b = beginbmatrix fracv_sx_l0endbmatrix","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"The goal is to design a controller to keep the state of the system in a safety region around the reference desired value, using as input only the switching signal.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"In order to study the concrete system and its symbolic abstraction in a unified framework, we will solve the problem for the sampled system with a sampling time tau.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"The abstraction is based on a feedback refinment relation [4,V.2 Definition]. Basically, this is equivalent to an alternating simulation relationship with the additional constraint that the input of the concrete and symbolic system preserving the relation must be identical. This allows to easily determine the controller of the concrete system from the abstraction controller by simply adding a quantization step.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"For the construction of the relations in the abstraction, it is necessary to over-approximate attainable sets of a particular cell. In this example, we consider the used of a growth bound function  [4, VIII.2, VIII.5] which is one of the possible methods to over-approximate attainable sets of a particular cell based on the state reach by its center. Therefore, it is used to compute the relations in the abstraction based on the feedback refinement relation.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"First, let us import StaticArrays.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"using StaticArrays","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"At this point, we import the useful Dionysos sub-module for this problem: Abstraction.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"using Dionysos\nusing Dionysos.Abstraction\nAB = Dionysos.Abstraction;\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/#Definition-of-the-system","page":"DC-DC converter","title":"Definition of the system","text":"","category":"section"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the parameters of the system:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"vs = 1.0; rL = 0.05; xL = 3.0; rC = 0.005; xC = 70.0; r0 = 1.0;\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the dynamics functions f_p of the system:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"b = SVector(vs/xL, 0.0);\nA1 = SMatrix{2,2}(-rL/xL, 0.0, 0.0, -1.0/xC/(r0+rC));\nA2 = SMatrix{2,2}(-(rL+r0*rC/(r0+rC))/xL, 5.0*r0/(r0+rC)/xC,\n    -r0/(r0+rC)/xL/5.0, -1.0/xC/(r0+rC));\nF_sys = let b = b, A1 = A1, A2 = A2\n    (x, u) -> u[1] == 1 ? A1*x + b : A2*x + b\nend;\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the growth bound functions of f_p:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"ngrowthbound = 5;\nA2_abs = SMatrix{2,2}(-(rL+r0*rC/(r0+rC))/xL, 5.0*r0/(r0+rC)/xC,\n                      r0/(r0+rC)/xL/5.0, -1.0/xC/(r0+rC));\nL_growthbound = let A1 = A1, A2_abs = A2_abs\n    u -> u[1] == 1 ? A1 : A2_abs\nend;\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Here it is considered that there is no system and measurement noise:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"sysnoise = SVector(0.0, 0.0);\nmeasnoise = SVector(0.0, 0.0);\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the discretization time step parameters: tstep and nsys:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"tstep = 0.5;\nnsys = 5;\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Finally, we build the control system:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"contsys = AB.NewControlSystemGrowthRK4(tstep, F_sys, L_growthbound, sysnoise,\n                                       measnoise, nsys, ngrowthbound);\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/#Definition-of-the-control-problem","page":"DC-DC converter","title":"Definition of the control problem","text":"","category":"section"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the state-space (limited to be rectangle):","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"_X_ = AB.HyperRectangle(SVector(1.15, 5.45), SVector(1.55, 5.85));\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the input-space, the later discretization of the input ensures that it can only take the values 1 or 2:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"_U_ = AB.HyperRectangle(SVector(1), SVector(2));\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/#Definition-of-the-abstraction","page":"DC-DC converter","title":"Definition of the abstraction","text":"","category":"section"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the grid of the state-space on which the abstraction is based (origin x0 and state-space discretization h):","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"x0 = SVector(0.0, 0.0);\nh = SVector(2.0/4.0e3, 2.0/4.0e3);\nXgrid = AB.GridFree(x0, h);\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Construction of the struct DomainList containing the feasible cells of the state-space. Note, we used AB.INNER to make sure to add cells entirely contained in the domain because we are working with a safety problem.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Xfull = AB.DomainList(Xgrid);\nAB.add_set!(Xfull, _X_, AB.INNER)","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Definition of the grid of the input-space on which the abstraction is based (origin u0 and input-space discretization h):","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"u0 = SVector(1);\nh = SVector(1);\nUgrid = AB.GridFree(u0, h);\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Construction of the struct DomainList containing the quantized inputs:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Ufull = AB.DomainList(Ugrid);\nAB.add_set!(Ufull, _U_, AB.OUTER);\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Construction of the abstraction:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"symmodel = AB.NewSymbolicModelListList(Xfull, Ufull);\n@time AB.compute_symmodel_from_controlsystem!(symmodel, contsys)","category":"page"},{"location":"generated/DC-DC converter/#Construction-of-the-controller","page":"DC-DC converter","title":"Construction of the controller","text":"","category":"section"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"In this problem, we consider both: the initial state-space and the safety state-space are equal to the entire state-space.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Computation of the initial symbolic states:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Xinit = AB.DomainList(Xgrid);\nunion!(Xinit, Xfull)\ninitlist = [AB.get_state_by_xpos(symmodel, pos) for pos in AB.enum_pos(Xinit)];\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Computation of the safety symbolic states:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Xsafe = AB.DomainList(Xgrid)\nunion!(Xsafe, Xfull)\nsafelist = [AB.get_state_by_xpos(symmodel, pos) for pos in AB.enum_pos(Xsafe)];\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"Construction of the controller:","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"contr = AB.NewControllerList();\n@time AB.compute_controller_safe!(contr, symmodel.autom, initlist, safelist)","category":"page"},{"location":"generated/DC-DC converter/#Trajectory-display","page":"DC-DC converter","title":"Trajectory display","text":"","category":"section"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"We choose the number of steps nsteps for the sampled system, i.e. the total elapsed time: nstep*tstep as well as the true initial state x0 which is contained in the initial state-space defined previously.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"nstep = 300;\nx0 = SVector(1.2, 5.6);\nnothing #hide","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"To complete","category":"page"},{"location":"generated/DC-DC converter/#References","page":"DC-DC converter","title":"References","text":"","category":"section"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"A. Girard, G. Pola and P. Tabuada, \"Approximately Bisimilar Symbolic Models for Incrementally Stable Switched Systems,\" in IEEE Transactions on Automatic Control, vol. 55, no. 1, pp. 116-126, Jan. 2010.\nS. Mouelhi, A. Girard, and G. Gössler. “CoSyMA: a tool for controller synthesis using multi-scale abstractions”. In: HSCC. ACM. 2013, pp. 83–88.\nA. Girard. “Controller synthesis for safety and reachability via approximate bisimulation”. In: Automatica 48.5 (2012), pp. 947–953.\nG. Reissig, A. Weber and M. Rungger, \"Feedback Refinement Relations for the Synthesis of Symbolic Controllers,\" in IEEE Transactions on Automatic Control, vol. 62, no. 4, pp. 1781-1796.","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"","category":"page"},{"location":"generated/DC-DC converter/","page":"DC-DC converter","title":"DC-DC converter","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#Dionysos","page":"Index","title":"Dionysos","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"Dionysos implements a solver for the optimal control of cyber-physical systems.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"EditURL = \"https://github.com/dionysos-dev/Dionysos.jl/blob/master/docs/src/examples/Path planning.jl\"","category":"page"},{"location":"generated/Path planning/#Example:-Path-planning-problem","page":"Path planning","title":"Example: Path planning problem","text":"","category":"section"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"(Image: Binder) (Image: nbviewer)","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"This example was borrowed from [1, IX. Examples, A] whose dynamics comes from the model given in [2, Ch. 2.4]. This is a reachability problem for a continuous system.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Let us consider the 3-dimensional state space control system of the form","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"dotx = f(x u)","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"with f mathbbR^3  U  mathbbR^3 given by","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"f(x(u_1u_2)) = beginbmatrix u_1 cos(α+x_3)cos(α^-1)  u_1 sin(α+x_3)cos(α^-1)  u_1 tan(u_2)  endbmatrix","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"and with U = 1 1 times 1 1 and α = arctan(tan(u_2)2). Here, (x_1 x_2) is the position and x_3 is the orientation of the vehicle in the 2-dimensional plane. The control inputs u_1 and u_2 are the rear wheel velocity and the steering angle. The control objective is to drive the vehicle which is situated in a maze made of obstacles from an initial position to a target position.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"In order to study the concrete system and its symbolic abstraction in a unified framework, we will solve the problem for the sampled system with a sampling time tau.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"The abstraction is based on a feedback refinment relation [1,V.2 Definition]. Basically, this is equivalent to an alternating simulation relationship with the additional constraint that the input of the concrete and symbolic system preserving the relation must be identical. This allows to easily determine the controller of the concrete system from the abstraction controller by simply adding a quantization step.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"For the construction of the relations in the abstraction, it is necessary to over-approximate attainable sets of a particular cell. In this example, we consider the used of a growth bound function  [1, VIII.2, VIII.5] which is one of the possible methods to over-approximate attainable sets of a particular cell based on the state reach by its center. Therefore, it is used to compute the relations in the abstraction based on the feedback refinement relation.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"For this reachability problem, the abstraction controller is built by solving a fixed-point equation which consists in computing the the pre-image of the target set.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"First, let us import StaticArrays.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"using StaticArrays","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"At this point, we import the useful Dionysos sub-module for this problem: Abstraction.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"using Dionysos\nusing Dionysos.Abstraction\nAB = Dionysos.Abstraction;\nnothing #hide","category":"page"},{"location":"generated/Path planning/#Definition-of-the-system","page":"Path planning","title":"Definition of the system","text":"","category":"section"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the dynamics function f of the system:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"function F_sys(x, u)\n    α = atan(tan(u[2])/2)\n    return SVector{3}(\n        u[1]*cos(α + x[3])/cos(α),\n        u[1]*sin(α + x[3])/cos(α),\n        u[1]*tan(u[2]))\nend;\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the growth bound function of f:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"ngrowthbound = 5;\nfunction L_growthbound(u)\n    β = abs(u[1]/cos(atan(tan(u[2])/2)))\n    return SMatrix{3,3}(\n        0.0, 0.0, 0.0,\n        0.0, 0.0, 0.0,\n        β, β, 0.0)\nend;\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Here it is considered that there is no system and measurement noise:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"sysnoise = SVector(0.0, 0.0, 0.0);\nmeasnoise = SVector(0.0, 0.0, 0.0);\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the discretization time step parameters: tstep and nsys:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"tstep = 0.3;\nnsys = 5;\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Finally, we build the control system:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"contsys = AB.NewControlSystemGrowthRK4(tstep, F_sys, L_growthbound, sysnoise,\n                                       measnoise, nsys, ngrowthbound);\nnothing #hide","category":"page"},{"location":"generated/Path planning/#Definition-of-the-control-problem","page":"Path planning","title":"Definition of the control problem","text":"","category":"section"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the state-space (limited to be rectangle):","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"_X_ = AB.HyperRectangle(SVector(0.0, 0.0, -pi - 0.4), SVector(4.0, 10.0, pi + 0.4));\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the obstacles (limited to be rectangle):","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"X1_lb = [1.0, 2.2, 2.2, 3.4, 4.6, 5.8, 5.8, 7.0, 8.2, 8.4, 9.3, 8.4, 9.3, 8.4, 9.3];\nX1_ub = [1.2, 2.4, 2.4, 3.6, 4.8, 6.0, 6.0, 7.2, 8.4, 9.3, 10.0, 9.3, 10.0, 9.3, 10.0];\nX2_lb = [0.0, 0.0, 6.0, 0.0, 1.0, 0.0, 7.0, 1.0, 0.0, 8.2, 7.0, 5.8, 4.6, 3.4, 2.2];\nX2_ub = [9.0, 5.0, 10.0, 9.0, 10.0, 6.0, 10.0, 10.0, 8.5, 8.6, 7.4, 6.2, 5.0, 3.8, 2.6];\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the input-space (limited to be rectangle):","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"_U_ = AB.HyperRectangle(SVector(-1.0, -1.0), SVector(1.0, 1.0));\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the initial state-space (here it consists in a single point):","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"_I_ = AB.HyperRectangle(SVector(0.4, 0.4, 0.0), SVector(0.4, 0.4, 0.0));\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the target state-space (limited to be hyper-rectangle):","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"_T_ = AB.HyperRectangle(SVector(3.0, 0.5, -100.0), SVector(3.6, 0.8, 100.0));\nnothing #hide","category":"page"},{"location":"generated/Path planning/#Definition-of-the-abstraction","page":"Path planning","title":"Definition of the abstraction","text":"","category":"section"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the grid of the state-space on which the abstraction is based (origin x0 and state-space discretization h):","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"x0 = SVector(0.0, 0.0, 0.0);\nh = SVector(0.2, 0.2, 0.2);\nXgrid = AB.GridFree(x0, h);\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Construction of the struct DomainList containing the feasible cells of the state-space:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Xfull = AB.DomainList(Xgrid);\nAB.add_set!(Xfull, _X_, AB.OUTER)\nfor (x1lb, x2lb, x1ub, x2ub) in zip(X1_lb, X2_lb, X1_ub, X2_ub)\n    box = AB.HyperRectangle(SVector(x1lb, x2lb, _X_.lb[3]), SVector(x1ub, x2ub, _X_.ub[3]))\n    if box ⊆ _X_ && isempty(box ∩ _I_) && isempty(box ∩ _T_)\n        AB.remove_set!(Xfull, box, AB.OUTER)\n    end\nend","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Definition of the grid of the input-space on which the abstraction is based (origin u0 and input-space discretization h):","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"u0 = SVector(0.0, 0.0);\nh = SVector(0.3, 0.3);\nUgrid = AB.GridFree(u0, h);\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Construction of the struct DomainList containing the quantized inputs:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Ufull = AB.DomainList(Ugrid);\nAB.add_set!(Ufull, _U_, AB.OUTER)","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Construction of the abstraction:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"symmodel = AB.NewSymbolicModelListList(Xfull, Ufull);\n@time AB.compute_symmodel_from_controlsystem!(symmodel, contsys)","category":"page"},{"location":"generated/Path planning/#Construction-of-the-controller","page":"Path planning","title":"Construction of the controller","text":"","category":"section"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Computation of the initial symbolic states:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Xinit = AB.DomainList(Xgrid);\nAB.add_subset!(Xinit, Xfull, _I_, AB.OUTER)\ninitlist = [AB.get_state_by_xpos(symmodel, pos) for pos in AB.enum_pos(Xinit)];\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Computation of the target symbolic states:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Xtarget = AB.DomainList(Xgrid)\nAB.add_subset!(Xtarget, Xfull, _T_, AB.OUTER)\ntargetlist = [AB.get_state_by_xpos(symmodel, pos) for pos in AB.enum_pos(Xtarget)];\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Construction of the controller:","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"contr = AB.NewControllerList();\n@time AB.compute_controller_reach!(contr, symmodel.autom, initlist, targetlist)","category":"page"},{"location":"generated/Path planning/#Trajectory-display","page":"Path planning","title":"Trajectory display","text":"","category":"section"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"We choose the number of steps nsteps for the sampled system, i.e. the total elapsed time: nstep*tstep as well as the true initial state x0 which is contained in the initial state-space _I_ defined previously.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"nstep = 100;\nx0 = SVector(0.4, 0.4, 0.0);\nnothing #hide","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"Here we display the coordinate projection on the two first components of the state space along the trajectory.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"To complete","category":"page"},{"location":"generated/Path planning/#References","page":"Path planning","title":"References","text":"","category":"section"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"G. Reissig, A. Weber and M. Rungger, \"Feedback Refinement Relations for the Synthesis of Symbolic Controllers,\" in IEEE Transactions on Automatic Control, vol. 62, no. 4, pp. 1781-1796.\nK. J. Aström and R. M. Murray, Feedback systems. Princeton University Press, Princeton, NJ, 2008.","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"","category":"page"},{"location":"generated/Path planning/","page":"Path planning","title":"Path planning","text":"This page was generated using Literate.jl.","category":"page"}]
}
