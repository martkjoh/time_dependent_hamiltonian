using SparseArrays
using Arpack

function make_H0(N, V)

    I = Array{Int}(undef, 3*(N-1) - 2)
    J = Array{Int}(undef, 3*(N-1) - 2)
    H = Array{Float64}(undef, 3*(N-1) - 2)
    x = LinRange(0, 1, N-1)

    I[1:3:end] = J[1:3:end] = 1:N-1     # Diagonal
    I[2:3:end] = J[3:3:end] = 1:N-2     # To the right (ii+1)
    I[3:3:end] = J[2:3:end] = 2:N-1     # To the left  (ii-1)
    H[1:3:end] = V(x) + 2*N^2*ones(N-1)
    H[2:3:end] = H[3:3:end] = -N^2*ones(N-2)

    return sparse(I, J, H)
end
