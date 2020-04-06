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

function inner(u, v)
    return dot(conj(u)', v)
end


function check_ortho(v, n)
    inners = Array{Float64}(undef, (n, n))
    for i in 1:n
        for j in 1:n
            inners[i, j] = inner(v[:, i], v[:, j])
        end
    end
    print(inners)
end


function f(l, V0)
    k = sqrt(l)
    kappa = sqrt(V0 - l)
    core_p = (kappa*sin(k/3) + k*cos(k/3))^2
    core_m = (kappa*sin(k/3) - k*cos(k/3))^2
    return exp(kappa/3)*core_p - exp(-kappa/3)*core_m
end


function secant(x1, x2, f, V0, tol=1e-10)
    n = 1
    while true
        if x2>=V0 return V0 end
        if x2<=0 return 0 end
        f1 = f(x1, V0); f2 = f(x2, V0)
        if abs(f2)<tol return Float64(x2) end
        xnew = x2 - (f2 * (x2 - x1)) / (f2 - f1)
        x1 = x2; x2 = xnew
    end
end


function roots(f, dx, V0)
    x = [0, dx, dx+1]
    roots = []
    while true
        # We want to walk downward before starting secant
        n = 1
        while true
            fs = f.(x, V0)
            if fs[2]<fs[1] && fs[2]<fs[3] break end
            x = dx .+ x
            if x[3]>=V0 return roots end
        end
        # Know there are two roots near bottom
        append!(roots, secant(x[1], x[2], f, V0))
        # "heuristic", also known as a hack. For low V0
        while f(x[3], V0)<0 && x[3]+dx<V0 x.+=dx end
        append!(roots, secant(x[2], x[3], f, V0))
        x = x .+ 1.
        if x[3]>=V0 return roots end
    end
end


function euler(v, H, T, dt)
    steps = ceil(Int, T/dt)
    I = sparse(diagm(ones(size(v)[1])))
    H1 = I .- 1im*dt*H
    for i in 1:steps
        v = H1*v
    end
    return v
end


function pade(v, H, T, dt)
    steps = ceil(Int, T/dt)
    N = size(v)[1]
    I = sparse(diagm(ones(N)))
    H1 = I-1im/2*H*dt
    H2 = inv(Matrix(I+1im/2*H*dt))
    for i in 1:steps
        v = H2*(H1*v)
    end
    return v
end
