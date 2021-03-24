
using Test

function example_test()
    # Test stuff here.
    ret = (1==1)
    return ret
end


function tests()

    @testset "Tests" begin
        @testset "Example Test" begin
            @test example_test() == true
        end
    end
end

tests()
