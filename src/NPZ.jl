__precompile__()
module NPZ

# NPZ file format is described in
# https://github.com/numpy/numpy/blob/v1.7.0/numpy/lib/format.py

using ZipFile, Compat

@static if VERSION >=  v"0.7.0-DEV.2575"
    import Base.CodeUnits
else
    # CodeUnits not yet supported by Compat but not needed in julia 0.6...
    # codeunits function in Compat returns uintX instead of codeunits
    # therefore this 'stump' type should work
    abstract type CodeUnits{U, S} end
end

export npzread, npzwrite

const NPYMagic = UInt8[0x93, 'N', 'U', 'M', 'P', 'Y']
const ZIPMagic = UInt8['P', 'K', 3, 4]
const Version = UInt8[1, 0]

const MaxMagicLen = maximum(length.([NPYMagic, ZIPMagic]))

const TypeMaps = [
    ("b1", Bool),
    ("i1", Int8),
    ("i2", Int16),
    ("i4", Int32),
    ("i8", Int64),
    ("u1", UInt8),
    ("u2", UInt16),
    ("u4", UInt32),
    ("u8", UInt64),
    ("f2", Float16),
    ("f4", Float32),
    ("f8", Float64),
    ("c8", Complex{Float32}),
    ("c16", Complex{Float64}),
]
const Numpy2Julia = Dict{String, DataType}()
for (s,t) in TypeMaps
    Numpy2Julia[s] = t
end

const Julia2Numpy = Dict{DataType, String}()

@static if VERSION >= v"0.4.0"
    function __init__()
        for (s,t) in TypeMaps
            Julia2Numpy[t] = s
        end
    end
else
    for (s,t) in TypeMaps
        Julia2Numpy[t] = s
    end
end

abstract type ArrayLayout{A} end
struct F_CONTIGUOUS{A} <: ArrayLayout{A}
    arr :: A
end
struct C_CONTIGUOUS{A} <: ArrayLayout{A}
    arr :: A
end

Base.ndims(::ArrayLayout{A}) where {A} = ndims(A)
Base.parent(A::ArrayLayout) = A.arr

# Julia2Numpy is a dictionary that uses Types as keys.
# This is problematic for precompilation because the
# hash of a Type changes everytime Julia is run.
# The hash of the keys when NPZ is precompiled will
# not be the same as when it is later run. This can
# be fixed by rehashing the Dict when the module is
# loaded.

readle(ios::IO, ::Type{T}) where T = ltoh(read(ios, T)) # ltoh is inverse of htol

function writecheck(io::IO, x::Any)
    n = write(io, x) # returns size in bytes
    n == sizeof(x) || error("short write") # sizeof is size in bytes
end

# Endianness only pertains to multi-byte things
writele(ios::IO, x::AbstractVector{UInt8}) = writecheck(ios, x)
writele(ios::IO, x::AbstractVector{CodeUnits{UInt8, <:Any}}) = writecheck(ios, x)
# codeunits returns vector of CodeUnits in 7+, uint in 6
writele(ios::IO, x::AbstractString) = writele(ios, codeunits(x))

writele(ios::IO, x::UInt16) = writecheck(ios, htol(x))

function parsechar(s::AbstractString, c::Char)
    firstchar = s[firstindex(s)]
    if  firstchar != c
        error("parsing header failed: expected character '$c', found '$firstchar'")
    end
    SubString(s, nextind(s, 1))
end

function parsestring(s::AbstractString)
    s = parsechar(s, '\'')
    parts = split(s, '\'', limit = 2)
    length(parts) != 2 && error("parsing header failed: malformed string")
    parts[1], parts[2]
end

function parsebool(s::AbstractString)
    if SubString(s, firstindex(s), thisind(s, 4)) == "True"
        return true, SubString(s, nextind(s, 4))
    elseif SubString(s, firstindex(s), thisind(s, 5)) == "False"
        return false, SubString(s, nextind(s, 5))
    end
    error("parsing header failed: excepted True or False")
end

function parseinteger(s::AbstractString)
    isdigit(s[firstindex(s)]) || error("parsing header failed: no digits")
    tail_idx = findfirst(c -> !isdigit(c), s)
    if tail_idx == nothing
        intstr = SubString(s, firstindex(s))
    else
        intstr = SubString(s, firstindex(s), prevind(s, tail_idx))
        if s[tail_idx] == 'L' # output of firstindex should be a valid code point
            tail_idx = nextind(s, tail_idx)
        end
    end
    n = parse(Int, intstr)
    return n, SubString(s, tail_idx)
end

function parsetuple(s::AbstractString)
    s = parsechar(s, '(')
    tup = Int[]
    while true
        s = strip(s)
        if s[firstindex(s)] == ')'
            break
        end
        n, s = parseinteger(s)
        push!(tup, n)
        s = strip(s)
        if s[firstindex(s)] == ')'
            break
        end
        s = parsechar(s, ',')
    end
    s = parsechar(s, ')')
    Tuple(tup), s
end

function parsedtype(s::AbstractString)
    dtype, s = parsestring(s)
    c = dtype[firstindex(s)]
    t = SubString(dtype, nextind(s, 1))
    if c == '<'
        toh = ltoh
    elseif c == '>'
        toh = ntoh
    elseif c == '|'
        toh = identity
    else
        error("parsing header failed: unsupported endian character $c")
    end
    if !haskey(Numpy2Julia, t)
        error("parsing header failed: unsupported type $t")
    end
    (toh, Numpy2Julia[t]), s
end

struct Header{T,N,F<:Function}
    descr::F
    fortran_order::Bool
    shape::NTuple{N,Int}
end

Header{T}(descr::F, fortran_order, shape::NTuple{N,Int}) where {T,N,F} = Header{T,N,F}(descr, fortran_order, shape)
Base.size(hdr::Header) = hdr.shape
Base.eltype(hdr::Header{T}) where T = T
Base.ndims(hdr::Header{T,N}) where {T,N} = N

function parseheader(s::AbstractString)
    s = parsechar(s, '{')

    dict = Dict{String,Any}()
    T = Any
    for _ in 1:3
        s = strip(s)
        key, s = parsestring(s)
        s = strip(s)
        s = parsechar(s, ':')
        s = strip(s)
        if key == "descr"
            (descr, T), s = parsedtype(s)
            dict[key] = descr
        elseif key == "fortran_order"
            dict[key], s = parsebool(s)
        elseif key == "shape"
            dict[key], s = parsetuple(s)
        else
            error("parsing header failed: bad dictionary key")
        end
        s = strip(s)
        if s[firstindex(s)] == '}'
            break
        end
        s = parsechar(s, ',')
    end
    s = strip(s)
    s = parsechar(s, '}')
    s = strip(s)
    if s != ""
        error("malformed header")
    end
    Header{T}(dict["descr"], dict["fortran_order"], dict["shape"])
end

function readheader(f::IO)
    @compat b = read!(f, Vector{UInt8}(undef, length(NPYMagic)))
    if b != NPYMagic
        error("not a numpy array file")
    end
    @compat b = read!(f, Vector{UInt8}(undef, length(Version)))

    # support for version 2 files
    if b[1] == 1
        hdrlen = UInt32(readle(f, UInt16))
    elseif b[1] == 2 
        hdrlen = UInt32(readle(f, UInt32))
    else
        error("unsupported NPZ version")
    end

    @compat hdr = ascii(String(read!(f, Vector{UInt8}(undef, hdrlen))))
    parseheader(strip(hdr))
end

function _npzreadarray(f, hdr::Header{T}, f_contiguous::Bool = true) where {T}
    toh = hdr.descr
    if hdr.fortran_order
        @compat x = map(toh, read!(f, Array{T}(undef, hdr.shape)))
    else
        @compat x = map(toh, read!(f, Array{T}(undef, reverse(hdr.shape))))
        if f_contiguous
            if ndims(x) > 1
                x = permutedims(x, collect(ndims(x):-1:1))
            end
        end
    end
    ndims(x) == 0 ? x[1] : x
end

function npzreadarray(f::IO, f_contiguous::Bool = true)
    hdr = readheader(f)
    _npzreadarray(f, hdr, f_contiguous)
end

function samestart(a::AbstractVector, b::AbstractVector)
    nb = length(b)
    length(a) >= nb && view(a, 1:nb) == b
end

function _maybetrimext(name::AbstractString)
    fname, ext = splitext(name)
    if ext == ".npy"
        name = fname
    end
    name
end

"""
    npzread(filename::AbstractString, [vars::Vector]; [f_contiguous = true])

Read a variable or a collection of variables from `filename`. 
The input needs to be either an `npy` or an `npz` file.
The optional argument `vars` is used only for `npz` files.
If it is specified, only the matching variables are read in from the file.

Arrays in Python usually follow a row-major layout in memory, 
while those in Julia follow a column-major layout.
The argument `f_contiguous` decides if an array written out 
in python is permuted while being read in.
If `f_contiguous` is `true`, an array written out in python will 
be read back in identically in julia.
If `f_contiguous` is set of `false`, the array will be transposed.
The latter preserves the contiguity of array elements across languages. 
The flag makes no difference while reading in column-major (`F_CONTIGUOUS` in numpy) arrays.

!!! note "Zero-dimensional arrays"
    Zero-dimensional arrays are stripped while being read in, and the values that they
    contain are returned. This is a notable difference from numpy, where 
    numerical values are written out and read back in as zero-dimensional arrays.

# Examples

```julia
julia> npzwrite("temp.npz", x = ones(3), y = 3)

julia> npzread("temp.npz") # Reads all variables
Dict{String,Any} with 2 entries:
  "x" => [1.0, 1.0, 1.0]
  "y" => 3

julia> npzread("temp.npz", ["x"]) # Reads only "x"
Dict{String,Array{Float64,1}} with 1 entry:
  "x" => [1.0, 1.0, 1.0]
```

# Examples of `f_contiguous`

Write out a row-major (`C_CONTIGUOUS`) array in python

```python
>>> import numpy as np

>>> a = np.reshape([i for i in range(1,13)], (4,3))

>>> a
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]])

>>> np.save("temp.npy", a)
```

Read it back in Julia by preserving the indices of elements:

```julia
julia> npzread("temp.npy", f_contiguous = true)
4×3 Array{Int64,2}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
```

Read it back in Julia by preserving the contiguity of elements (resulting in a transpose):

```julia
julia> npzread("temp.npy", f_contiguous = false)
3×4 Array{Int64,2}:
 1  4  7  10
 2  5  8  11
 3  6  9  12
```
"""
function npzread(filename::AbstractString, vars...; f_contiguous::Bool = true)
    # Detect if the file is a numpy npy array file or a npz/zip file.
    f = open(filename)
    @compat b = read!(f, Vector{UInt8}(undef, MaxMagicLen))

    if samestart(b, ZIPMagic)
        fz = ZipFile.Reader(filename)
        data = npzread(fz, vars...; f_contiguous = f_contiguous)
        close(fz)
    elseif samestart(b, NPYMagic)
        seekstart(f)
        data = npzreadarray(f, f_contiguous)
    else
        close(f)
        error("not a NPY or NPZ/Zip file: $filename")
    end
    close(f)
    return data
end

function npzread(dir::ZipFile.Reader, 
    vars = map(f -> _maybetrimext(f.name), dir.files); f_contiguous::Bool = true)

    Dict(_maybetrimext(f.name) => npzreadarray(f, f_contiguous)
        for f in dir.files 
            if f.name in vars || _maybetrimext(f.name) in vars)
end

"""
    readheader(filename, [vars...])

Return a header or a collection of headers corresponding to each variable contained in `filename`. 
The header contains information about the `eltype` and `size` of the array that may be extracted using 
the corresponding accessor functions.
"""
function readheader(filename::AbstractString, vars...)
    # Detect if the file is a numpy npy array file or a npz/zip file.
    f = open(filename)
    @compat b = read!(f, Vector{UInt8}(undef, MaxMagicLen))

    if samestart(b, ZIPMagic)
        fz = ZipFile.Reader(filename)
        data = readheader(fz, vars...)
        close(fz)
    elseif samestart(b, NPYMagic)
        seekstart(f)
        data = readheader(f)
    else
        close(f)
        error("not a NPY or NPZ/Zip file: $filename")
    end

    close(f)
    return data
end
function readheader(dir::ZipFile.Reader, 
    vars = map(f -> _maybetrimext(f.name), dir.files))

    Dict(_maybetrimext(f.name) => readheader(f)
        for f in dir.files 
            if f.name in vars || _maybetrimext(f.name) in vars)
end

_metahdr(x::F_CONTIGUOUS, shape) = "'fortran_order': True, 'shape': $(Tuple(shape))"
_metahdr(x::C_CONTIGUOUS, shape) = "'fortran_order': False, 'shape': $(reverse(Tuple(shape)))"
function metahdr(x::ArrayLayout, descr, shape)
    "{'descr': '$descr', " * _metahdr(x, shape) * ", }"
end

function npzwritearray(f::IO, x::ArrayLayout, T::DataType, shape)

    if !haskey(Julia2Numpy, T)
        error("unsupported type $T")
    end
    writele(f, NPYMagic)
    writele(f, Version)

    descr =  (ENDIAN_BOM == 0x01020304 ? ">" : "<") * Julia2Numpy[T]
    dict = metahdr(x, descr, shape)

    # The dictionary is padded with enough whitespace so that
    # the array data is 16-byte aligned
    n = length(NPYMagic)+length(Version)+2+length(dict)
    pad = (div(n+16-1, 16)*16) - n
    if pad > 0
        dict *= " "^(pad-1) * "\n"
    end

    writele(f, UInt16(length(dict)))
    writele(f, dict)
    y = parent(x)
    N = write(f, y)
    if N != length(y)
        error("short write")
    end
end

_to1D(x::Number) = [x]
_to1D(a::AbstractArray) = vec(a)

for DT in [:F_CONTIGUOUS, :C_CONTIGUOUS]
    @eval function npzwritearray(f, x::$DT)
        y = parent(x)
        npzwritearray(f, $DT(reinterpret(UInt8, _to1D(y))), eltype(y), size(y))
    end
end
# by default arrays are written out in the column-major convention
npzwritearray(f::IO, x::Union{AbstractArray, Number}) = npzwritearray(f, F_CONTIGUOUS(x))

"""
    npzwrite(filename::AbstractString, x)

Write the variable `x` to the `npy` file `filename`. 
Unlike `numpy`, the extension `.npy` is not appened to `filename`.

The variable `x` may be wrapped in the tags `NPZ.F_CONTIGUOUS` or `NPZ.C_CONTIGUOUS` to 
explicitly specify the layout in memory that is to be assumed while reading it back in. 
The former indicates that it is to be interpreted as a column-major array, 
whereas the latter states that it is to be interpreted as a row-major one.
The default choice implicit in the function is `NPZ.F_CONTIGUOUS`.

!!! warn "Warning"
    Any existing file with the same name will be overwritten.

# Examples

```julia
julia> npzwrite("abc.npy", zeros(3))

julia> npzread("abc.npy")
3-element Array{Float64,1}:
 0.0
 0.0
 0.0
```

# Examples of the usage of `F_CONTIGUOUS` and `C_CONTIGUOUS`

### `F_CONTIGUOUS`

```julia
julia> a = reshape(1:12, 3, 4)
3×4 reshape(::UnitRange{Int64}, 3, 4) with eltype Int64:
 1  4  7  10
 2  5  8  11
 3  6  9  12
```

Write out an array with the `F_CONTIGUOUS` tag:
```julia
julia> npzwrite("temp.npy", NPZ.F_CONTIGUOUS(a))

julia> npzread("temp.npy") # always read in identically in julia
3×4 Array{Int64,2}:
 1  4  7  10
 2  5  8  11
 3  6  9  12
```

The array is read in identically in python, except the memory layout is now column-major.

```python
>>> import numpy as np

>>> np.load("temp.npy")
array([[ 1,  4,  7, 10],
       [ 2,  5,  8, 11],
       [ 3,  6,  9, 12]])

>>> np.load("temp.npy").strides
(8, 24)
```

### `C_CONTIGUOUS`

Write out an array with the `C_CONTIGUOUS` tag:

```julia
julia> npzwrite("temp.npy", NPZ.C_CONTIGUOUS(a))
```

`C_CONTIGUOUS` arrays are transposed by default while being read back using [`npzread`](@ref):

```julia
julia> npzread("temp.npy")
4×3 Array{Int64,2}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
```

To avoid the transpose and read in the original array that was written out, 
specify the tag `f_contiguous = false`:

```julia
julia> npzread("temp.npy", f_contiguous = false)
3×4 Array{Int64,2}:
 1  4  7  10
 2  5  8  11
 3  6  9  12
```

The array is read in as a row-major one in python. As a consequence the matrix is transposed.

```python
>>> import numpy as np

>>> np.load("temp.npy")
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]])

>>> np.load(f).strides
(24, 8)
```
"""
function npzwrite(filename::AbstractString, x)
    open(filename, "w") do f
        npzwritearray(f, x)
    end
end

"""
    npzwrite(filename::AbstractString, vars::Dict{<:AbstractString})
    npzwrite(filename::AbstractString, args...; kwargs...)

In the first form, write the variables in `vars` to an `npz` file named `filename`.

In the second form, collect the variables in `args` and `kwargs` and write them all
to `filename`. The variables in `args` are saved with names `arr_0`, `arr_1` 
and so on, whereas the ones in `kwargs` are saved with the specified names.

Each variable to be written out may be wrapped in `NPZ.F_CONTIGUOUS` or `NPZ.C_CONTIGUOUS` to 
explicitly specify the layout in memory that is to be assumed while reading it back in. 
The former indicates that the variable is to be interpreted 
as a column-major array, whereas the latter states that it is to be interpreted as a 
row-major one. The default choice implicit in the function is `NPZ.F_CONTIGUOUS`.

Unlike `numpy`, the extension `.npz` is not appened to `filename`.

!!! warn "Warning"
    Any existing file with the same name will be overwritten.

# Examples

```julia
julia> npzwrite("temp.npz", Dict("x" => ones(3), "y" => 3))

julia> npzread("temp.npz")
Dict{String,Any} with 2 entries:
  "x" => [1.0, 1.0, 1.0]
  "y" => 3

julia> npzwrite("temp.npz", ones(2,2), x = ones(3), y = 3)

julia> npzread("temp.npz")
Dict{String,Any} with 3 entries:
  "arr_0" => [1.0 1.0; 1.0 1.0]
  "x"     => [1.0, 1.0, 1.0]
  "y"     => 3
```

# Examples of the usage of `F_CONTIGUOUS` and `C_CONTIGUOUS`

Write out arrays in julia
```julia
julia> a = reshape(1:12, 3, 4)
3×4 reshape(::UnitRange{Int64}, 3, 4) with eltype Int64:
 1  4  7  10
 2  5  8  11
 3  6  9  12

julia> npzwrite("temp.npz", ac = NPZ.C_CONTIGUOUS(a), af = NPZ.F_CONTIGUOUS(a))

julia> npzread("temp.npz")["af"] # F_CONTIGUOUS arrays are read back in identically
3×4 Array{Int64,2}:
 1  4  7  10
 2  5  8  11
 3  6  9  12

julia> npzread("temp.npz")["ac"] # C_CONTIGUOUS arrays are transposed by default
4×3 Array{Int64,2}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12

julia> npzread("temp.npz", f_contiguous = false)["ac"] # read the original array back without a transposition
3×4 Array{Int64,2}:
 1  4  7  10
 2  5  8  11
 3  6  9  12
```
"""
function npzwrite(filename::AbstractString, vars::Dict{<:AbstractString}) 
    dir = ZipFile.Writer(filename)

    if length(vars) == 0
        @warn "no data to be written to $filename. It might not be possible to read the file correctly."
    end

    for (k, v) in vars
        f = ZipFile.addfile(dir, k * ".npy")
        npzwritearray(f, v)
        close(f)
    end

    close(dir)
end

function npzwrite(filename::AbstractString, args...; kwargs...)
    dkwargs = Dict(string(k) => v for (k,v) in kwargs)
    dargs = Dict("arr_"*string(i-1) => v for (i,v) in enumerate(args))

    d = convert(Dict{String,Any}, merge(dargs, dkwargs))

    npzwrite(filename, d)
end

end # module
