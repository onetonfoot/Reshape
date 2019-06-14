using Test, Reshape, MLStyle, Base.Meta
import Reshape: hasreshape, haspermute, getreshapedims, getpermutedims

#Unit Tests

@testset "hasreshape" begin
    @testset "true" begin
        @test hasreshape(:(B[(i,j), k] = A[i,j,k]))
        @test hasreshape(:(B[i,_] = A[i]))
        @test hasreshape(:(B[(i,j), _, k] = A[i,j,k]))
    end

    @testset "false" begin
        @test !hasreshape(:(B[i,j,k] = A[i,j,k]))
        @test !hasreshape(:(B[j,i] = A[i, j]))
    end
end

@testset "haspermute" begin

    @testset "true" begin
        @test haspermute(:(B[k,j,i] = A[i,j,k]))
    end

    @testset "false" begin
        @test !haspermute(:(B[i,j,_] = A[i,j]))
        @test !haspermute(:(B[(i,j), k] = A[i,j,k]))
    end
end

#TODO add test for getreshapedims, getpermutedims

#Intergration test

@testset "permutation" begin
    @testset "Two dims" begin
        A = rand(5,10)
        B = permutedims(A,(2,1))
        @permute C[j,i] = A[i,j]
        @test C == B
    end

    @testset "Three dims" begin
        A = rand(1,2,3)
        B = permutedims(A,(3,2,1))
        @permute C[k,j,i] = A[i,j,k]
        @test C == B
    end
end

@testset "reshape" begin
    @testset "add dim" begin
        A = rand(5,10)
        B = reshape(A, 1, 5, 10)
        @permute C[_, i,j] = A[i,j]
        @test C == B
    end

    @testset "group dims" begin
        A = rand(5,10)
        B = reshape(A, 50)
        @permute C[(i,j)] = A[i,j]
        @test C == B
    end
end
