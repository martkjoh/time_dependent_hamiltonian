using SparseArrays
using Arpack
using Plots


function make_H0(N, V)

    I = Array{Int}(undef, 3*N - 2)
    J = Array{Int}(undef, 3*N - 2)
    H = Array{Float64}(undef, 3*N - 2)
    x = LinRange(0, 1, N)

    I[1:3:end] = J[1:3:end] = 1:N       # Diagonal
    I[2:3:end] = J[3:3:end] = 1:N-1     # To the right (ii+1)
    I[3:3:end] = J[2:3:end] = 2:N       # To the left  (ii-1)
    H[1:3:end] = V(x) + 2*(N-1)^2*ones(N)
    H[2:3:end] = H[3:3:end] = -(N-1)^2*ones(N-1)

    return sparse(I, J, H)
end

V(x) = zeros(size(x))

H = make_H0(10, V)

heatmap(Matrix(H))
