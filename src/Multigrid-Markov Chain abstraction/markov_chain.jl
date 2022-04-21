
using DiscreteMarkovChains

mutable struct MarkovChain
    #conversion of symbol
    symmodel_from_mc #list
    mc_from_symmodel #dictionnary

    transition_matrix
    chain
    shannon_entropy
    SCC
    steady_state
    entropy
end


function new_MarkovChain(symmodel_from_mc, mc_from_symmodel,transition_matrix,shannon_entropy)
    chain = DiscreteMarkovChain(transition_matrix)
    communications = communication_classes(chain)
    n = length(shannon_entropy)
    steady_state = zeros(n)
    entropy = zeros(n)
    for (i,class) in enumerate(communications[1])
        if communications[2][i]
            transition_matrix_scc = transition_matrix[class,class]
            chain_scc = DiscreteMarkovChain(transition_matrix_scc)
            p = stationary_distribution(chain_scc)
            for (j,s) in enumerate(class)
                steady_state[s] = p[j]
                entropy[s] = steady_state[s]*shannon_entropy[s]
            end
            # else let value to zero: transcient class (non recurent)
        end
    end
    return MarkovChain(symmodel_from_mc, mc_from_symmodel,transition_matrix, chain, shannon_entropy, communications, steady_state, entropy)
end


function conversion(symbols)
    symmodel_from_mc = symbols #get_cells(symmodel)
    mc_from_symmodel = Dict{Int, Int}()
    for (i,s) in enumerate(symmodel_from_mc)
        mc_from_symmodel[s] = i
    end
    return symmodel_from_mc, mc_from_symmodel
end

#inputs: list of symbols, pas necessairement 1,2,3,4,5...
# list de transitions
function build_Markov_Chain(symmodel)
    symbols = get_cells(symmodel)
    symmodel_from_mc, mc_from_symmodel = conversion(symbols)

    n = length(symbols)
    transition_matrix = zeros(n,n)
    shannon_entropy = zeros(n)
    entropy = zeros(n)
    for s in symmodel_from_mc
        idx_s = mc_from_symmodel[s]
        h = 0.0
        if s == 0
            transition_matrix[idx_s,idx_s] = 1.0
        else
            for (t,symbol,proba) in get_transitions_post(symmodel, s)
                idx_t = mc_from_symmodel[t]
                transition_matrix[idx_s,idx_t] = proba
                h = h-proba*log2(proba)
            end
        end
        shannon_entropy[idx_s] = h
    end
    return new_MarkovChain(symmodel_from_mc, mc_from_symmodel,transition_matrix,shannon_entropy)
end


function get_entropy(mc::MarkovChain, s::Int)
    i = mc.mc_from_symmodel[s]
    return mc.entropy[i]
end

function get_shannon_entropy(mc::MarkovChain, s::Int)
    i = mc.mc_from_symmodel[s]
    return mc.shannon_entropy[i]
end

function get_steady_state(mc::MarkovChain, s::Int)
    i = mc.mc_from_symmodel[s]
    return mc.steady_state[i]
end

function get_entropy_chain(mc::MarkovChain)
    h = 0.0
    for p in mc.steady_state
        if p > 0
            h = h - p*log2(p)
        end
    end
    return h
end

function get_highest_entropy(mc::MarkovChain)
    i = findmax(mc.entropy)
    return (mc.symmodel_from_mc[i], mc.entropy[i])
end
