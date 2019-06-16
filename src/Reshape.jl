module Reshape
using MLStyle
using Base.Iterators: flatten

export _permute, @permute

# https://discuss.pytorch.org/t/difference-between-2-reshaping-operations-reshape-vs-permute/30749/4

rmlines = @λ begin
    e :: Expr           -> Expr(e.head, filter(x -> x !== nothing, map(rmlines, e.args))...)
      :: LineNumberNode -> nothing
    a                   -> a
end

function hasreshape(expr)
    (lsym, lexpr) , (rsym, rexpr) = split_expr.(split_expr(expr))
    f(x) = @match x begin
        Expr(:tuple, args...) => true
        :_                    => true
        _                     => false
    end
    any(map(f,lexpr))
end

function haspermute(expr)
    (_, lexpr) , (_, rexpr) = split_expr.(split_expr(expr))
    f(x) = @match x begin
        Expr(:tuple,args...) => map(f,args)
        :_ => []
        x::Symbol => x
    end
    filter_expr(lexpr) != filter_expr(rexpr)
end

function hascat(expr)
    lexpr, rexpr = split_expr(expr)
    #should check the expression more throughly
    @match rexpr.args begin
        [expr::Expr, tail...] => true
        _ => false
    end
end


#Should rename to split_expr
function split_expr(expr)
    lexpr, rexpr = @match expr begin
        Expr(:(=), lexpr, rexpr) => (lexpr, rexpr)
        Expr(:ref, var::Symbol, dims...) => (var, dims)
        Expr(:ref, expr::Expr, dims...) => (expr, dims)
        _ => ErrorException("Couldn't split expr")
    end
    lexpr, rexpr
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


#Possible since we can calculate from just the exprsion
function getpermutedims(expr)

    # haspeexpr

    (lvar, lexpr) , (lvar, rexpr) = split_expr.(split_expr(expr))
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
    (lsym, lexpr) , (rsym, rexpr) = split_expr.(split_expr(expr))

    a = if hascat(expr)
        #lexpr and bexpr
        (lvar, lexpr) , (rexpr, bexpr) = split_expr.(split_expr(expr))
        rvar, rdims = split_expr(rexpr)
        bdims  = bexpr #is an array? [:batch]
        code = quote
            #should check for empty list somewhere to prevent error
            n = ndims($rvar[1]) + 1
            $lvar = cat($rvar...; dims=n)
        end

        #update expression
        expr = Expr(Symbol("="),
            #left expression is assumed to be simple here
            Expr(:ref, lvar, lexpr...),
            Expr(:ref, lvar, rdims..., bdims...)
        )
        code
    else
        :(:nocat)
    end

    b = if haspermute(expr)
        (lvar, ldims) , (rvar, rdims) = split_expr.(split_expr(expr))
        dims = getpermutedims(expr)
        quote
            $lvar = permutedims($rvar,$dims)
            $lvar
        end
    else
        (lvar, ldims) , (rvar, rdims) = split_expr.(split_expr(expr))
        quote
            :nopermute
            $lvar = $rvar
        end
    end

    c = if hasreshape(expr)
        (lsym, lexpr) , (rsym, rexpr) = split_expr.(split_expr(expr))
        quote
            dim2size = Dict(k=>v for (k,v) in zip($rexpr,size($rsym)))
            dims = $getreshapedims($lexpr, dim2size)
            $lsym = reshape($lsym,dims...)
        end
    else
        (lvar, ldims) , (rvar, rdims) = split_expr.(split_expr(expr))
        :($lvar)
    end

    quote
        $a
        $b
        $c
    end
end

macro permute(expr)
    #esc ensures symbols in expr are evaluated outside this modules scope
    _permute(expr) |> esc |> rmlines
end

end # module
