#abstract type SymbolicModel{N,M} end




mutable struct MultiSymbolicModel{B,A,N,M} #<: AB.SymbolicModel{N,M}
    Xdom::MultiDomain
    Udom::B
    autom::A
    xpos2int::Dict{Tuple{Int,NTuple{N,Int}},Int}
    xint2pos::Vector{Tuple{Int,NTuple{N,Int}}}
    #upos2int::Dict{NTuple{M,Int},Int}
    uint2input::Vector{M}
    active::Vector{Bool}

    mc#::MarkovChain
end

function get_ncells(symmodel::MultiSymbolicModel)
    return length(symmodel.xint2pos)
end

function get_cells(symmodel::MultiSymbolicModel)
    return findall(symmodel.active)
end

function update_MC!(symmodel::MultiSymbolicModel)
    #symmodel.mc = build_Markov_Chain(get_cells(symmodel),symmodel.autom.transitions)
    symmodel.mc = build_Markov_Chain(symmodel)
end

function plot_shannon_entropy(sys,symmodel::MultiSymbolicModel;dims=[1,2])
    fig = plot(aspect_ratio = 1,legend = false,title="outgoing shannon-entropy")
    plot_shannon_entropy!(symmodel;dims=dims)
    x0 = SVector(2.0,2.0)
    plot_trajectory!(sys,x0,200)
    display(fig)
end

function plot_shannon_entropy!(symmodel::MultiSymbolicModel;dims=[1,2])
    mc = symmodel.mc
    max_val = max(mc.shannon_entropy...)
    for s in get_cells(symmodel) #symmodel.symmodel_from_mc
        (l,pos) = get_xpos_by_state(symmodel, s)
        dom = symmodel.Xdom.domains[l]
        val = get_shannon_entropy(mc, s)
        opacity = val/max_val
        if opacity > 0.01
            AB.plot_elem!(dom.grid, pos, dims=dims,opacity=opacity,color=:yellow)
        end
    end
end

function plot_steady_state(sys,symmodel::MultiSymbolicModel;dims=[1,2],fact=1.0,tol=0.0)
    fig = plot(aspect_ratio = 1,legend = false,title="steady-state probability")
    plot_steady_state!(symmodel;dims=dims,fact=fact,tol=tol)
    x0 = SVector(2.0,2.0)
    plot_trajectory!(sys,x0,200)
    display(fig)
end

function plot_steady_state!(symmodel::MultiSymbolicModel;dims=[1,2],fact=1.0,tol=0.0,color=:yellow)
    mc = symmodel.mc
    max_val = max(mc.steady_state...)
    for s in get_cells(symmodel)
        (l,pos) = get_xpos_by_state(symmodel, s)
        dom = symmodel.Xdom.domains[l]
        val = get_steady_state(mc, s)
        opacity = val/max_val
        if val>fact
            opacity = 1.0
        end
        if opacity > tol
            AB.plot_elem!(dom.grid, pos, dims=dims,opacity=opacity,color=color)
        end
    end
end

function plot_entropy(sys,symmodel::MultiSymbolicModel;dims=[1,2])
    fig = plot(aspect_ratio = 1,legend = false,title="entropy")
    plot_entropy!(symmodel;dims=dims)
    x0 = SVector(2.0,2.0)
    plot_trajectory!(sys,x0,200)
    display(fig)
end

function plot_entropy!(symmodel::MultiSymbolicModel;dims=[1,2])
    mc = symmodel.mc
    max_val = max(mc.entropy...)
    for s in  get_cells(symmodel)
        (l,pos) = get_xpos_by_state(symmodel, s)
        dom = symmodel.Xdom.domains[l]
        val = get_entropy(mc, s)
        opacity = val/max_val
        if opacity > 0.01
            AB.plot_elem!(dom.grid, pos, dims=dims,opacity=opacity,color=:yellow)
        end
    end
end

function plot_SCC(sys,symmodel::MultiSymbolicModel;dims=[1,2])
    mc = symmodel.mc
    fig = plot(aspect_ratio = 1,legend = false,title="SCC recurrent")
    for (i,class) in enumerate(mc.SCC[1])
        if mc.SCC[2][i]
            for (j,idx) in enumerate(class)
                s = mc.symmodel_from_mc[idx]
                (l,pos) = get_xpos_by_state(symmodel, s)
                dom = symmodel.Xdom.domains[l]
                AB.plot_elem!(dom.grid, pos, dims=dims,color=i)
            end
        end
    end
    x0 = SVector(2.0,2.0)
    plot_trajectory!(sys,x0,200)
    display(fig)
    fig = plot(aspect_ratio = 1,legend = false,title="SCC non recurrent ")
    for (i,class) in enumerate(mc.SCC[1])
        if !mc.SCC[2][i]
            for (j,idx) in enumerate(class)
                s = mc.symmodel_from_mc[idx]
                (l,pos) = get_xpos_by_state(symmodel, s)
                dom = symmodel.Xdom.domains[l]
                AB.plot_elem!(dom.grid, pos, dims=dims,color=i)
            end
        end
    end
    x0 = SVector(2.0,2.0)
    plot_trajectory!(sys,x0,200)
    display(fig)
end


function NewMultiSymbolicModel(dom::D.GeneralDomainList, Udom) #{N,D.RectanglularObstacles{NTuple{N,T}}}
    N = D.get_dim(dom)
    Xdom = MultiDomain(dom)
    nu = AB.get_ncells(Udom)
    uint2pos = [input for input in AB.enum_pos(Udom)]
    #upos2int = Dict((pos, i) for (i, pos) in enumerate(AB.enum_pos(Udom)))
    symmodel = MultiSymbolicModel(
        Xdom,
        Udom,
        NewProbaAutomaton(0, nu),
        Dict{Tuple{Int,NTuple{N,Int}},Int}(),
        Tuple{Int,NTuple{N,Int}}[],
        uint2pos,
        Bool[],
        nothing
    )
end


function with_automaton(symmodel::MultiSymbolicModel, autom)
    return MultiSymbolicModel(
        symmodel.Xdom,
        symmodel.Udom,
        autom,
        symmodel.xpos2int,
        symmodel.xint2pos,
        symmodel.upos2int,
        symmodel.uint2pos,
    )
end

function get_state_by_xpos(
    symmodel::MultiSymbolicModel,pos,l
) where {N,M,T}
    dom = symmodel.Xdom.domains[l]
    pos = D.set_in_period_pos(dom,pos)
    id = get(symmodel.xpos2int, (l,pos), nothing)
    created = false
    if id === nothing
        if pos in dom
            push!(symmodel.active,true)
            created = true
            push!(symmodel.xint2pos, (l,pos))
            id = length(symmodel.xint2pos)
            symmodel.xpos2int[(l,pos)] = id
            i = AB.HybridSystems.add_state!(symmodel.autom)
            @assert i == id
        else
            error("$pos is not in state domain $(symmodel.Xdom)")
        end
    end
    return id::Int
end

function delete_state!(symmodel::MultiSymbolicModel,s::Int)
    #should we delete in probaautomaton the transition related to node s, no should be done before deleting the node.
    symmodel.active[s] = false
end

function delete_transitions_post!(symmodel, source, symbol)
    delete_transition_post!(symmodel.autom,source,symbol)
end

function Base.in(symmodel::MultiSymbolicModel, s::Int)
    return symmodel.active[s]
end

function get_transitions_pre(symmodel::MultiSymbolicModel, s::Int)
    active = symmodel.active
    translist = pre(symmodel.autom, s)
    list = []
    for e in translist
        if active[e[1]]
            push!(list,e)
        end
    end
    return list
end

function get_transitions_post(symmodel::MultiSymbolicModel, s::Int)
    active = symmodel.active
    translist = post(symmodel.autom, s)
    list = []
    for e in translist
        if e[1] == 0 || active[e[1]]
            push!(list,e)
        end
    end
    return list
end

function get_transitions(symmodel::MultiSymbolicModel)
    active = symmodel.active
    return autom
end



function get_xpos_by_state(symmodel::MultiSymbolicModel, state::Int)
    return symmodel.xint2pos[state]
end

function get_state_by_coord(symmodel::MultiSymbolicModel, coord)
    l = 1
    Xdom = symmodel.Xdom
    pos = AB.get_pos_by_coord(Xdom,l,coord)
    s = get_state_by_xpos(symmodel, pos, l)
    return s
end

function get_upos_by_symbol(symmodel::MultiSymbolicModel, symbol::Int)
    return symmodel.uint2pos[symbol]
end

function get_symbol_by_upos(symmodel::MultiSymbolicModel, upos)
    return symmodel.upos2int[upos]
end

function Plots.plot(symmodel::MultiSymbolicModel;dims=[1,2],annotate=false)
    fig = plot(aspect_ratio = 1,legend = false)
    plot!(symmodel;dims=dims,annotate=annotate)
    display(fig)
end
function Plots.plot!(symmodel::MultiSymbolicModel;dims=[1,2],annotate=false)
    colors = :yellow
    for value in symmodel.xint2pos
        (l,pos) = value
        dom = symmodel.Xdom.domains[l]
        grid = dom.grid
        if annotate
           s = get_state_by_xpos(symmodel, pos, l)
           center = AB.get_coord_by_pos(dom.grid, pos)
           annotate!([(center[dims[1]], center[dims[2]], text(s, :red))])
        end
        AB.plot_elem!(grid,pos, opacity=.9,color=colors)
    end
end

function plot_vector_field(symmodel::MultiSymbolicModel,f;dims=[1,2])
    fig = plot(aspect_ratio = 1,legend = false)
    plot!(symmodel,dims=dims)
    for (i,value) in enumerate(symmodel.xint2pos)
        if i%1 == 0
            (l,pos) = value
            dom = symmodel.Xdom.domains[l]
            grid = dom.grid
            center = AB.get_coord_by_pos(grid, pos)
            d = f(center,0.0)
            d =d/norm(d,2)
            quiver!([center[1]],[center[2]],quiver=([d[1]],[d[2]]),color=:black,linewidth=1)
        end
    end
    display(fig)
end

function plot_map(symmodel::MultiSymbolicModel,f;dims=[1,2])
    fig = plot(aspect_ratio = 1,legend = false)
    tab = []
    val_max = -Inf
    for (i,value) in enumerate(symmodel.xint2pos)
        (l,pos) = value
        dom = symmodel.Xdom.domains[l]
        grid = dom.grid
        center = AB.get_coord_by_pos(grid, pos)
        val = f(center,0.0)
        val_max = max(val_max,val)
        push!(tab,(pos,val))
    end
    grid = symmodel.Xdom.domains[1].grid
    for (pos,val) in tab
        opacity = val/val_max
        # println()
        # println(val)
        # println(opacity)
        AB.plot_elem!(grid, pos, opacity=opacity,color=:yellow)
    end
    # x0 = SVector(2.0,2.0)
    # plot_trajectory!(sys,x0,200)
    display(fig)
end


function plot_Jacobian(sys,symmodel::MultiSymbolicModel,Jacobian;dims=[1,2])
    fig = plot(aspect_ratio = 1,legend = false,title="Stability")
    for (i,value) in enumerate(symmodel.xint2pos)
        (l,pos) = value
        dom = symmodel.Xdom.domains[l]
        grid = dom.grid
        center = AB.get_coord_by_pos(grid, pos)
        J = Jacobian(center,0.0)
        eigenvalues = eigvals(Array(J))
        r1 = real(eigenvalues[1])
        r2 = real(eigenvalues[2])
        if r1<0 && r2<0
            AB.plot_elem!(grid, pos, opacity=1.0,color=:green)
        elseif r1>0 && r2>0
            AB.plot_elem!(grid, pos, opacity=1.0,color=:red)
        else
            AB.plot_elem!(grid, pos, opacity=1.0,color=:yellow)
        end
    end
    x0 = SVector(2.0,2.0)
    plot_trajectory!(sys,x0,200)
    display(fig)
end
