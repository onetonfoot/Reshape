module Reshape
using MLStyle
using Base.Iterators: flatten

export _permute, @permute

# https://discuss.pytorch.org/t/difference-between-2-reshaping-operations-reshape-vs-permute/30749/4

# After the reference need to walk the expression
# and keep track of the index of the underscores
# Need to walk expression tree and keep a counter
# Could also filter out single variables
rmlines = @λ begin
    e :: Expr           -> Expr(e.head, filter(x -> x !== nothing, map(rmlines, e.args))...)
      :: LineNumberNode -> nothing
    a                   -> a
end

function hasreshape(expr)
    lsym, lexpr, rsym, rexpr = splitexpr(expr)
    f(x) = @match x begin
        Expr(:tuple, args...) => true
        :_                    => true
        _                     => false
    end
    any(map(f,lexpr))
end

function haspermute(expr)
    _, lexpr, _ , rexpr = splitexpr(expr)
    f(x) = @match x begin
        Expr(:tuple,args...) => map(f,args)
        :_ => []
        x::Symbol => x
    end
    filter_expr(lexpr) != filter_expr(rexpr)
end

#TODO think of better name
#filters non dim symbols fron exprsion
function filter_expr(expr)
    f(x) = @match x begin
        Expr(:tuple,args...) => map(f,args)
        :_ => []
        x::Symbol => x
    end
    vcat(map(f, expr)...)
end


#TODO throw usefull error if expr not in correct form
function splitexpr(expr)
    lexpr, rexpr = @match expr begin
        Expr(:(=), lexpr, rexpr) => (lexpr, rexpr)
        _ => "Should be in from B[x₁...xₙ] = A[x₁...x₂]"
    end

    lsym,ldims = @match lexpr begin
        Expr(:ref,lvar, ldims...) => (lvar,ldims)
    end

    rsym,rdims = @match rexpr begin
        Expr(:ref,rvar, rdims...) => (rvar,rdims)
    end
    lsym, ldims, rsym, rdims
end

#Possible since we can calculate from just the exprsion
function getpermutedims(expr)
    lvar, lexpr, lvar, rexpr = splitexpr(expr)
    ldims = filter_expr(lexpr)
    rdims = rexpr
    tuple(indexin(ldims,rdims)...)
end

#this requies some form of evaluation
function getreshapedims(lexpr, dim2size)
    f(x) = @match x begin
        :_                    => 1
        Expr(:tuple, args...) => reduce(*,map(f, args))
        x::Symbol             => dim2size[x]
    end
    tuple(map(f,lexpr)...)
end

function _permute(expr)
    #B[i,j,k] = A[i,j,k]
    lsym, lexpr, rsym, rexpr = splitexpr(expr)

    #TODO handle concats

    #handle permutation
    a = if haspermute(expr)
        dims = getpermutedims(expr)
        lsym, lexpr, rsym, rexpr = splitexpr(expr)
        quote
            $lsym = permutedims($rsym,$dims)
            $lsym
        end
    else
        :($lsym = $rsym)
    end

    #handle reshaping
    b = if hasreshape(expr)
        quote
            dim2size = Dict(k=>v for (k,v) in zip($rexpr,size($rsym)))
            dims = $getreshapedims($lexpr, dim2size)
            $lsym = reshape($lsym,dims...)
        end
    else
        :($lsym)
    end

    quote
        $a
        $b
    end
end

macro permute(expr)
    #esc ensures symbols in expr are evaluated outside this modules scope
    _permute(expr) |> esc |> rmlines
end

end # module
