
module LogToolKit

# export some structs and functions as global
export
    Symb, ISymb, HSymb, Li,
    monomial, lincomb, polynomial, tensor, Phi, Phi_inv, fundamental_column, complementary_entry, VariationMatrix,
    ordered_partitions, findfirst_subsequence, latex_repr

using LaTeXStrings
using Latexify
using Combinatorics

"""
    ordered_partitions(n, k, positive=true)

Return the all (positive) ordered k-partitions of an integer of n, implemented on top of partitions

```julia
julia> ordered_partitions(5, 2)
6-element Vector{Vector{Int64}}:
 [6, 1]
 [1, 6]
 [5, 2]
 [2, 5]
 [4, 3]
 [3, 4]

julia> ordered_partitions(5, 2, positive=false)
6-element Vector{Vector{Int64}}:
 [5, 0]
 [0, 5]
 [4, 1]
 [1, 4]
 [3, 2]
 [2, 3]
```
"""
function ordered_partitions(n::Int, k::Int; positive=true)::Vector{Vector{Int}}
    if positive # only positive partitions
        return union((unique(permutations(unordered_partition)) for unordered_partition in partitions(n + k, k))...)
    else # all non-negative partitions
        return [partition .- 1 for partition in ordered_partitions(n, k)] # implemented on top of the positive case
    end
end

"""
    findfirst_subsequence(A, B)

return the indices of the first occurence of a subsequence B in A, or nothing if none is found

```julia
julia> findfirst_subsequence((1,1,2,3,4,3),[1,3])
2-element Vector{Int64}:
 1
 4

julia> findfirst_subsequence((1,1,2,3,4,3),(1,5))
```
"""
function findfirst_subsequence(A::Union{Tuple{Vararg{T}}, Vector{T}}, B::Union{Tuple{Vararg{T}}, Vector{T}}) where T
    A, B = collect(A), collect(B) # If A is a range, generator, or other iterable, collect(A) produces a concrete Array containing all elements
    indices = Int[0] # the matching indices, the last element is supposed to be the last matching index, which initalizes to 0 as there is no matching index yet
    for b in B
        ind = findfirst(a -> a == b, A[indices[end]+1:end]) # find the next matching index
        if ind == nothing # if nothing found, then it fails, and return nothing
            indices = nothing
            break
        end
        push!(indices, indices[end] + ind) # push the newly matched index
    end

    return indices == nothing ? nothing : indices[2:end] # doesn't include 0 of course
end

"""
    latex_repr

return the latex representation of Number, Matrix, or any type that has a `latex_repr` field

```julia
julia> latex_repr([[1,2],[3,4]])
"\\begin{bmatrix}\\begin{bmatrix}1 \\\\ 2\\end{bmatrix} \\\\ \\begin{bmatrix}3 \\\\ 4\\end{bmatrix}\\end{bmatrix}"

julia> latex_repr(ISymb(0,0,1))
"I(a_{0},a_{1})"

julia> latex_repr([ISymb(0,0,1), ISymb(0,1,2)])
"\\begin{bmatrix}I(a_{0},a_{1}) \\\\ I(a_{0},a_{2})\\end{bmatrix}"
```
"""
function latex_repr(expr)
    if expr isa Rational
        if denominator(expr)==1
            return string(expr)
        else
            return "\\frac{" * latex_repr(numerator(expr)) * "}{" * latex_repr(denominator(expr)) * "}"
        end
    elseif expr isa Number
        return string(expr)
    elseif expr isa Vector || expr isa Tuple
        return "\\begin{bmatrix}" * join(map(latex_repr, expr), " \\\\ ") * "\\end{bmatrix}"
    elseif expr isa Matrix
        m, n = size(expr)
        return "\\begin{bmatrix}" * join([join(map(latex_repr, row), " & ") for row in eachrow(expr)], " \\\\ ") * "\\end{bmatrix}"
    elseif :latex_repr in fieldnames(typeof(expr))
        return expr.latex_repr # simply return its latex_repr attribute
    else
        throw(ArgumentError("No LaTeX representations!"))
    end
end



"""
    monomial{T}

Monomials with variables in type T, where T must be a comparable type that has the `latex_repr` attribute

```julia
julia> monomial{ISymb}((ISymb(0,0,1),2), (ISymb(0,1,2),-1)).latex_repr
"I(a_{0},a_{1})^{2}I(a_{0},a_{2})^{-1}"

julia> monomial{ISymb}(ISymb(0,1,2)).latex_repr
"I(a_{0},a_{2})"
```
"""
struct monomial{T} # T must be comparable type that has the latex_repr attribute
    args::Tuple{Vararg{Pair{T, Int}}} # variables and their exponents
    deg::Int
    latex_repr::String
end

function monomial{T}(args::Vararg{Union{Tuple{T, Int}, Pair{T, Int}, T}}) where T
    # multiply the variables
    counter = Dict{T, Int}()
    for var in args
        (key, value) = typeof(var) <: T ? (var, 1) : var
        counter[key] = get(counter, key, 0) + value
    end

    # sort the variables
    args = Tuple(sort(filter(x->x.second!=0, collect(counter))))

    # return 1 when the args is empty
    if length(args) == 0
        return 1
    end

    # compute the degree of the monomial
    deg = sum([exponent for (_, exponent) in args])

    # generate latex_repr
    latex_repr = ""
    for (var, exponent) in args
        if exponent == 1
            latex_repr *= "$(var.latex_repr)"
        else
            latex_repr *= "$(var.latex_repr)^{$(exponent)}"
        end
    end

    return monomial{T}(args, deg, latex_repr)
end

"""
    lincomb{T}

Linear/affine combination with terms in type T, where T must be a comparable type that has the `latex_repr` attribute

```julia
julia> lincomb{ISymb}(ISymb(0,1,2)).latex_repr
"I(a_{0},a_{2})"

julia> lincomb{ISymb}((ISymb(0,0,1),2), (ISymb(0,1,2),-1)).latex_repr
"(2)I(a_{0},a_{1})+(-1)I(a_{0},a_{2})"
```
"""
struct lincomb{T}
    args::Tuple{Vararg{Pair{T, Number}}} # variable and coefficient, coef is in the second argument to ease the sorting
    intercept::Number # constant term
    latex_repr::String
end

"""
    polynomial{T}

A polynomial in T is a linear/affine combination of monomials in T
"""
polynomial{T} = lincomb{Union{T, monomial{T}}}

function lincomb{T}(args::Vararg{Union{T, Tuple{T, Number}, Pair{T, Number}}}; intercept=0) where T
    # combine like terms
    counter = Dict{T, Number}()
    for term in args
        (key, value) = typeof(term) <: T ? (term, 1) : term
        counter[key] = get(counter, key, 0) + value
    end

    # sort the terms
    args = Tuple(sort(filter(x->x.second!=0, collect(counter))))

    # return the intercept when the args is empty
    if length(args) == 0
        return intercept
    end

    # generate latex_repr
    argstr = String[]
    if intercept != 0
        push!(argstr, "($intercept)")
    end
    for term in args
        if term.second == 1
            push!(argstr, term.first.latex_repr)
        else
            push!(argstr, "($(term.second))" * term.first.latex_repr)
        end
    end
    latex_repr = join(argstr, '+')

    return lincomb{T}(args, intercept, latex_repr)
end

"""
    tensor{T}

single tensor in type T, where T must be a comparable type that has the `latex_repr` attribute
"""
struct tensor{T} # T must be a type that has the latex_repr attribute
    args::Tuple{Vararg{T}}
    latex_repr::String
end
"""
    tensors{T}

A linear combination of single tensors
"""
tensors{T} = lincomb{Union{T, tensor{T}}}

# constructor for tensor{T}
function tensor{T}(args::Vararg{T}) where T
    if length(args) == 0
        return 1
    elseif length(args) == 1
        return args[1]
    else
        return tensor{T}(args, join(map(x -> x.latex_repr, args), "\\otimes "))
    end
end


"""
    Symb

An abstract type with instances called "symbols" that has no relations among themselves
"""
abstract type Symb end

## Overloading the Base operators ==, <, *, +, -, inv, ^


Base.isequal(a::Number, b::Union{T, monomial{T}, polynomial{T}, tensor{T}}) where {T <: Symb} = false
Base.isequal(a::T, b::monomial{T}) where {T <: Symb} = monomial{T}(a) == b
Base.isequal(a::Union{T, monomial{T}}, b::polynomial{T}) where {T <: Symb} = polynomial{T}(a) == b


Base.isless(a::Number, b::Union{T, monomial{T}, polynomial{T}, tensor{T}}) where {T <: Symb} = true
Base.isless(a::Union{T, monomial{T}, polynomial{T}, tensor{T}}, b::Number) where {T <: Symb} = false
Base.isless(a::monomial{T}, b::monomial{T}) where {T <: Symb} = (a.deg, reverse(a.args)) < (b.deg, reverse(b.args))
Base.isless(a::polynomial{T}, b::polynomial{T}) where {T <: Symb} = (reverse(a.args), a.intercept) < (reverse(b.args), b.intercept)
Base.isless(a::T, b::monomial{T}) where {T <: Symb} = monomial{T}(a) < b
Base.isless(a::monomial{T}, b::T) where {T <: Symb} = a < monomial{T}(b)
Base.isequal(a::Union{T, monomial{T}}, b::polynomial{T}) where {T <: Symb} = polynomial{T}(a) < b
Base.isequal(a::polynomial{T}, b::Union{T, monomial{T}}) where {T <: Symb} = a < polynomial{T}(b)
Base.isless(a::tensor{T}, b::tensor{T}) where {T <: Symb} = (length(a), a.args) < (length(b), b.args)


Base.:+(a::Number, b::Union{T, monomial{T}}) where {T <: Symb} = polynomial{T}(b; intercept=a)
Base.:+(a::Number, b::polynomial{T}) where {T <: Symb} = polynomial{T}(b.args...; intercept = a + b.intercept)
Base.:+(a::Union{T, monomial{T}, polynomial{T}}, b::Number) where {T <: Symb} = b + a
Base.:+(a::Union{T, monomial{T}}, b::Union{T, monomial{T}}) where {T <: Symb} = polynomial{T}(a, b)
Base.:+(a::polynomial{T}, b::polynomial{T}) where {T <: Symb} = polynomial{T}(a.args..., b.args...; intercept = a.intercept + b.intercept)
Base.:+(a::Union{T, monomial{T}}, b::polynomial{T}) where {T <: Symb} = polynomial{T}(a, b.args...; intercept = b.intercept)
Base.:+(a::polynomial{T}, b::Union{T, monomial{T}}) where {T <: Symb} = b + a


Base.:-(a::Union{T, monomial{T}}) where {T <: Symb} = polynomial{T}((a, -1))
Base.:-(a::polynomial{T}) where {T <: Symb} = polynomial{T}(map(x->(x.first, -x.second), a.args)...; intercept=-a.intercept)


Base.:-(a::Number, b::Union{T, monomial{T}, polynomial{T}}) where {T <: Symb} = a + (-b)
Base.:-(a::Union{T, monomial{T}, polynomial{T}}, b::Number) where {T <: Symb} = a + (-b)
Base.:-(a::Union{T, monomial{T}, polynomial{T}}, b::Union{T, monomial{T}, polynomial{T}}) where {T <: Symb} = a + (-b)


Base.:*(a::Number, b::Union{T, monomial{T}}) where {T <: Symb} = polynomial{T}((b, a))
Base.:*(a::Number, b::polynomial{T}) where {T <: Symb} = polynomial{T}(map(x->(x.first, a*x.second), b.args)...; intercept=a*b.intercept)
Base.:*(a::Union{T, monomial{T}, polynomial{T}}, b::Number) where {T <: Symb} = b * a
Base.:*(a::Union{T, monomial{T}}, b::Union{T, monomial{T}}) where {T <: Symb} = monomial{T}((a isa T ? [a] : a.args)..., (b isa T ? [b] : b.args)...)
Base.:*(a::Union{T, monomial{T}}, b::polynomial{T}) where {T <: Symb} = polynomial{T}((a, b.intercept), map(x->(a*x.first, x.second), b.args)...)
Base.:*(a::polynomial{T}, b::Union{T, monomial{T}}) where {T <: Symb} = b * a
Base.:*(a::polynomial{T}, b::polynomial{T}) where {T <: Symb} =
    polynomial{T}(
        vec([(t1.first * t2.first, t1.second * t2.second) for t1 in a.args, t2 in b.args])...,
        [(t.first, a.intercept * t.second) for t in b.args]...,
        [(t.first, b.intercept * t.second) for t in a.args]...,
        ; intercept = a.intercept * b.intercept
    )
Base.:*(a::tensor{T}, b::tensor{T}) where {T <: Symb} = tensor(x * y for (x,y) in zip(a.args, b.args))
Base.:*(a::tensor{T}, b::lincomb{tensor{T}}) where {T <: Symb} = lincomb{tensor{T}}((a, b.intercept), map(x->(a*x.first, x.second), b.args)...)
Base.:*(a::lincomb{tensor{T}}, b::tensor{T}) where {T <: Symb} = b * a


Base.inv(a::T) where {T <: Symb} = monomial{T}((a, -1))
Base.:inv(a::monomial{T}) where {T <: Symb} = monomial{T}(map(x -> (x.first, -x.second), a.args)...)


Base.:^(a::T, n::Int) where {T <: Symb} = monomial{T}((a, n))
Base.:^(a::monomial{T}, n::Int) where {T <: Symb} = monomial{T}(map(x -> (x.first, n*x.second), a.args)...)
Base.:^(a::polynomial{T}, n::Int) where {T <: Symb} =
    if n < 0
        throw(ArgumentError("the exponent must be at least 0"))
    elseif n <= 1
        n == 1 ? a : 1
    else
        temp = a^div(n, 2)
        temp * temp * (n % 2 == 0 ? 1 : a)
    end

"""
    ISymb

Iterated integral symbol
`I(a_{i_0}; 0^{n_0-1}, ...; a_{i_{m+1}}) <==> (i_0, n_0, ..., i_{m+1})` is of weight `n_0 + ... + n_m - 1`
"""
struct ISymb <: Symb
    args::Tuple{Vararg{Int}}
    weight::Int
    m::Int
    i::Function
    n::Function
    latex_repr::String

    function ISymb(args::Vararg{Int})
        # Validate the number of arguments
        if length(args) % 2 == 0 || length(args) < 3
            throw(ArgumentError("The number of arguments should be odd and at least 3"))
        end

        m = div(length(args), 2) - 1
        i(r) = args[1 + 2 * r]
        n(r) = args[2 + 2 * r]
        argstrs = String[]

        # Check the validity of arguments
        if m < 0 || any(n(r) < 1 for r in 1:m)
            throw(ArgumentError("The arguments are not valid"))
        end

        # Form the latex string
        for r in 0:m
            push!(argstrs, "a_{$(i(r))}")
            if n(r) == 2
                push!(argstrs, "0")
            elseif n(r) > 2
                push!(argstrs, "0^{$(n(r)-1)}")
            end
        end
        push!(argstrs, "a_{$(i(m+1))}")

        # Return the final object
        return new(args, sum(n(r) for r in 0:m) - 1, m, i, n, "I($(join(argstrs, ',')))")
    end
end

Base.isless(a::ISymb, b::ISymb) = begin
    if a.weight != b.weight
        return a.weight < b.weight
    end

    for r in 0: a.m-1
        if a.i(a.m-r) != b.i(b.m-r)
            return a.i(a.m-r) < b.i(b.m-r)
        elseif a.n(a.m-r) != b.n(b.m-r)
            return a.n(a.m-r) > b.n(b.m-r)
        end
    end

    return a.i(0) < b.i(0)
end

"""
    HSymb

Multiple polylogarithm symbols

`[x_{i_1->i_2}, ..., x_{i_d->i_{d+1}}]_{n_1,...,n_d} <==> (i_1, n_1, i_2 - i_1, ..., n_d, i_{d+1} - i_d)`

Or `(m_1, n_1, ..., m_{d+1})` so that `i_r = m_1 + ... + m_r`
"""
struct HSymb <: Symb
    args::Tuple{Vararg{Int}}
    weight::Int
    d::Int
    i::Tuple{Vararg{Int}}
    m::Function
    n::Function
    latex_repr::String

    function HSymb(args::Vararg{Int})
        # Validate the number of arguments
        if length(args) % 2 == 0 || length(args) < 3
            throw(ArgumentError("The number of arguments should be odd and at least 3"))
        end

        d = div(length(args), 2)
        i = cumsum(args[1:2:end])
        m(r) = r == 0 ? 0 : args[2 * r - 1]
        n(r) = r == 0 ? 1 : args[2 * r]
        weight = n(1)==0 ? 1 : sum(n(r) for r in 1:d)
        argstrs = String[]
        indices = Int64[]

        # Check the validity of arguments
        if d <= 0 || any(m(r) < 1 for r in 1:d+1) || n(1)<0 || (d > 1 && any(n(r) < 1 for r in 1:d))
            throw(ArgumentError("The arguments are not valid"))
        end

        # Form the latex string
        for r in 1:d
            push!(argstrs, "x_{$(i[r])\\to $(i[r+1])}")
            push!(indices, n(r))
        end

        # Return the final object
        return new(args, weight, d, i, m, n, "[$(join(argstrs, ','))]_{$(join(indices, ','))}")
    end
end

"""
    Li(n_args, x_args)

Multiple polylogarithm symbols, `x_args` can be Int or UnitRange
```julia
julia> Li((1,1,2),(2:3,3:5,5)).latex_repr
"[x_{2\\to 3},x_{3\\to 5},x_{5\\to 6}]_{1,1,2}"
```
"""
function Li(n_args::Tuple{Vararg{Int}}, x_args::Tuple{Vararg{Union{Int, UnitRange{Int}}}})::HSymb
    # turn Int into UnitRange
    x_args = Tuple(map(x -> x isa Int ? UnitRange{Int}(x,x+1) : x, x_args))
    # examine the arguments
    if length(n_args) == 0
        throw(ArgumentError("The depth must be positive"))
    elseif length(n_args) != length(x_args)
        throw(ArgumentError("The depth is not in accordance with the x variables"))
    elseif any(n <= 0 for n in n_args)
        throw(ArgumentError("The weights must be positive"))
    elseif first(x_args[1]) <= 0 || any(first(x_args[r]) != last(x_args[r-1]) for r in 2:length(x_args))
        throw(ArgumentError("The x variables is not valid"))
    end

    return HSymb(
        first(x_args[1]),
        vcat([[n_args[r], last(x_args[r]) - first(x_args[r])] for r in 1:length(x_args)]...)...
    )
end

Base.isless(a::HSymb, b::HSymb) = begin
    if a.weight != b.weight
        return a.weight < b.weight
    elseif a.i[end] != b.i[end]
        return a.i[end] < b.i[end]
    end

    for r in 0: a.d-1
        if a.m(a.d-r) != b.m(b.d-r)
            return a.m(a.d-r) > b.m(b.d-r)
        elseif a.n(a.d-r) != b.n(b.d-r)
            return a.n(a.d-r) > b.n(b.d-r)
        end
    end

    return false
end

function partial_differential(H::HSymb, r::Int)
    if r > H.d || r < 1
        throw(ArgumentError("The partial differential is invalid"))
    end
end

function differential(H::HSymb)

end

function Phi(I::Union{Number, ISymb, monomial{ISymb}, polynomial{ISymb}}, d::Int)
    if I isa Number
        return I
    elseif I isa monomial{ISymb}
        return prod(map(x -> Phi(x.first, d)^(x.second), I.args))
    elseif I isa polynomial{ISymb}
        return I.intercept + sum(map(x -> x.second * Phi(x.first, d), I.args))
    end

    # check i_{m+1} <= d + 1
    if d + 1 < I.i(I.m+1)
        throw(ArgumentError("The depth is invalid"))
    end

    if I.m == 0 && I.n(1) == 1
        return 1

    elseif I.i(0) == 0 && I.i(I.m+1) == 0
        return 0

    elseif I.i(0) > 0 && I.i(I.m+1) == 0
        return (-1)^(I.weight) * Phi(ISymb(reverse(I.args)), d)

    elseif I.i(0) > 0 && I.i(I.m+1) > 0
        return sum(
            sum(
                Phi(ISymb(I.args[1:2*k+1]..., p, 0), d) * Phi(ISymb(0, I.n(k)-p, I.args[2*k+3:end]...), d)
                for p in 0:I.n(k)
            )
            for k in 0:I.m
        )

    elseif I.i(0) == 0 && I.i(I.m+1) == d + 1
        return sum(
            (-1)^(I.n(0)+I.m-1) * prod(binomial(I.n(r)+p[r]-1, p[r]) for r in 1:I.m)
            * HSymb(I.i(1), vcat(([I.n(r)+p[r], I.i(r+1)-I.i(r)] for r in 1:I.m)...)...)
            for p in ordered_partitions(I.n(0)-1, I.m; positive = false)
        )

    elseif I.i(0) == 0 && I.i(I.m+1) > 0
        return sum(
            sum(
                (-1)^(I.n(0)+p0+I.m-1) * HSymb(I.i(I.m+1),1,d+1)^p0 * prod(binomial(I.n(r)+p[r]-1, p[r]) for r in 1:I.m)
                * HSymb(I.i(1), vcat(([I.n(r)+p[r], I.i(r+1)-I.i(r)] for r in 1:I.m)...)...)
                for p in ordered_partitions(I.n(0)-1-p0, I.m; positive = false)
            )
            for p0 in 0:I.n(0)-1
        )
    end
end

function Phi_inv(H::Union{Number, HSymb, monomial{HSymb}, polynomial{HSymb}})
    if H isa Number
        return H
    elseif H isa monomial{HSymb}
        return prod(map(x -> Phi_inv(x.first, d)^(x.second), H.args))
    elseif H isa polynomial{HSymb}
        return H.intercept + sum(map(x -> x.second * Phi_inv(x.first, d), H.args))
    else
        return (-1)^H.d * ISymb(0, 1, vcat(([H.i[r], H.n(r)] for r in 1:H.d)...)..., H.i[end])
    end
end



function fundamental_column(h::HSymb)
    visited = Set{HSymb}()
    result = []

    function dfs(H::HSymb)
        if H in visited
            return
        end

        push!(result, H)
        push!(visited, H)

        if H.d == 1 && H.n(1) == 1
            return
        end

        for r in 1:H.d
            if H.n(r) > 1
                dfs(HSymb(H.args[1:2*r-1]..., H.n(r)-1, H.args[2*r+1:end]...))
            elseif r == H.d
                dfs(HSymb(H.args[1:end-3]..., H.m(H.d)+H.m(H.d+1)))
            else
                dfs(HSymb(H.args[1:2*r-2]..., H.m(r)+H.m(r+1), H.args[2*r+2:end]...))
                dfs(HSymb(H.args[1:2*r-1]..., H.n(r+1), H.m(r+1)+H.m(r+2), H.args[2*r+4:end]...))
            end
        end
    end

    dfs(h)
    return [1, sort(result)...]
end

function complementary_entry(a::Union{HSymb, Int}, b::Union{HSymb, Int})
    # initiate the parameters
    k, l = b.d, a.d
    i(r) = r == 0 ? 0 : b.i[r]
    j(r) = r == 0 ? 0 : a.i[r]
    p, m = a.n, b.n

    # q_r sequence
    qr = findfirst_subsequence(map(j, 0:l+1), map(i, 0:k+1))
    if qr == nothing
        return 0
    end

    # shift to 0-indexing
    q(r) = qr[r+1] - 1

    # I^{sigma_0^{m_r}}(...)
    function Isigma(r::Int)
        if m(r) == 1
            # I(a_{j_{q_r}}, ..., a_{j_{q_{r+1}}})
            return ISymb([_ for t in q(r):q(r+1)-1 for _ in (j(t), p(t))]..., j(q(r+1)))
        end

        result = 0

        for s in q(r):q(r+1)-1
            if m(r) > p(s)
                continue
            end

            result += sum(
                ISymb([_ for t in q(r):s-1 for _ in (j(t), p(t))]..., j(s), u+1, 0) *
                ISymb(0, p(s)-m(r)-u+1, [_ for t in s+1:q(r+1)-1 for _ in (j(t), p(t))]..., j(q(r+1)))
                for u in 0:p(s)-m(r)
            )
        end

        return result
    end

    return (-1)^(l-k) * prod(Isigma(r) for r in 0:k)
end

function VariationMatrix(H::HSymb)
    fun_col = fundamental_column(H)
    return [
        if j == 1
            Phi_inv(fun_col[i])
        elseif i < j
            0
        elseif i == j
            1
        else
            complementary_entry(fun_col[i], fun_col[j])
        end
        for i in 1:length(fun_col), j in 1:length(fun_col)
    ]
end

# This is M_{w,v}^{(i_0)}
function MonodromyMatrixEntry(w0::Union{HSymb,Int}, v0::Union{HSymb,Int}, i0::Int)
    # turn HSymb into ISymb
    if w0 isa Int
        return 0
    end
    w = ISymb(0, 1, vcat(([w0.i[r], w0.n(r)] for r in 1:w0.d)...)..., w0.i[end])
    j, p, l = w.i, w.n, w.m
    v = v0 isa Int ? ISymb(0,0,w0.i[end]) : ISymb(0, 1, vcat(([v0.i[r], v0.n(r)] for r in 1:v0.d)...)..., v0.i[end])
    i, m, k = v.i, v.n, v.m
    r = findfirst(x -> i0<i(x), 1:k+1) - 1
    qr = p(l-k+r) - m(r)
    if any(j(s)!=i(s) for s in 1:r) || any(m(s)!=p(s) for s in 1:r-1) ||
        any(j(l-s)!=i(k-s) for s in 0:k-r-1) || any(p(l-s)!=m(k-s) for s in 0:k-r-1) ||
        qr<0 || sum((p(s)-1 for s in r:l-k+r-1); init=0)!=0
        # so that extra sigmas are between sigma_{i_r} and sigma_{i_{r+1}} and q_r>=0
        return 0
    end
    theta = sum((j(s) for s in r+1:l-k+r); init=0)
    return qr>1 ? (-1)^(k+theta) // factorial(qr) : (-1)^(k+theta)
end

# This is M_{w,v}^{(i_0,j_0)}
function MonodromyMatrixEntry(w0::Union{HSymb,Int}, v0::Union{HSymb,Int}, i0::Int, j0::Int)
    if i0 > j0
        return 0
    end
    # turn HSymb into ISymb
    if w0 isa Int
        return 0
    end
    w = ISymb(0, 1, vcat(([w0.i[r], w0.n(r)] for r in 1:w0.d)...)..., w0.i[end])
    j, p, l = w.i, w.n, w.m
    v = v0 isa Int ? ISymb(0,0,w0.i[end]) : ISymb(0, 1, vcat(([v0.i[r], v0.n(r)] for r in 1:v0.d)...)..., v0.i[end])
    i, m, k = v.i, v.n, v.m
    r = findfirst(x -> i0<=i(x), 1:k+1)
    if any(j(s)!=i(s) for s in 1:r) || any(m(s)!=p(s) for s in 1:r-1) ||
        any(j(l-s)!=i(k-s) for s in 0:k-r-1) || any(p(l-s)!=m(k-s) for s in 0:k-r-1) ||
        sum((p(s)-1 for s in r:l-k+r-1); init=0)!=0
        # so that extra sigmas are between sigma_{i_r} and sigma_{i_{r+1}}
        return 0
    end
    theta = sum((j(s) for s in r+1:l-k+r); init=0) # this is suppose to be sum(delta*theta)+delta
    if i0 == i(r) && j0+1 < i(r+1)
        return (-1)^(k+theta-j(l-k+r))
    elseif j0+1 == i(r+1) && i(r) < i0
        return (-1)^(k+theta)
    else
        return 0
    end
end

function MonodromyMatrix(H::HSymb, i0::Int)
    L = fundamental_column(H)
    return [(i > j ? MonodromyMatrixEntry(L[i], L[j], i0) : Int(i==j)) for i in 1:length(L), j in 1:length(L)]
end

function MonodromyMatrix(H::HSymb, i0::Int, j0::Int)
    L = fundamental_column(H)
    return [(i > j ? MonodromyMatrixEntry(L[i], L[j], i0, j0) : Int(i==j)) for i in 1:length(L), j in 1:length(L)]
end

end
