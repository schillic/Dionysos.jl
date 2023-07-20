module Test_ZT

include(joinpath(@__DIR__,"..", "..", "src", "ZMPBipedRobot.jl"))
import .ZMPBipedRobot as ZMProbot
using Test 

sleep(0.1) # used for good printing
println("Started test")


@testset "ZMP Trajectory Generator" begin 
    br = ZMProbot.BipedRobot(;
        readFile = true,
        paramFileName = "param_test.jl",
    )

    fp = ZMProbot.FootPlanner(br = br)
    zt = ZMProbot.ZMPTrajectory(br = br, fp = fp, check = false)
    tref = reduce(vcat, zt.timeVec)
    
    @test length(zt.ZMP) == length(fp.center)
    @test [tref[i] ≈  br.Ts * (i-1) for i = 1: length(tref)] == ones( length(tref))
end 

sleep(0.1) # used for good printing
println("End test")

end # End Main Module 
