module Test_SF

include(joinpath(@__DIR__, "..", "..", "src", "ZMPBipedRobot.jl"))
import .ZMPBipedRobot as ZMProbot
using Test

sleep(0.1) # used for good printing
println("Started test")

@testset "Swing Foot Trajectory Generator" begin
    URDFfileName = "ZMP_2DBipedRobot.urdf"

    br = ZMProbot.BipedRobot(;
        readFile = true,
        URDFfileName = URDFfileName,
        paramFileName = "param_test.jl",
    )
    fp = ZMProbot.FootPlanner(; br = br)
    zt = ZMProbot.ZMPTrajectory(; br = br, fp = fp, check = false)
    sf = ZMProbot.SwingFootTrajectory(; br = br, fp = fp, zt = zt)

    tref = reduce(vcat, zt.timeVec)
    stepL_plot = reduce(hcat, sf.stepL)
    stepR_plot = reduce(hcat, sf.stepR)

    @test length(sf.stepL) == length(fp.center)
    @test length(sf.stepR) == length(fp.center)
    @test length(stepR_plot[1, :]) == length(tref)
    @test length(stepL_plot[1, :]) == length(tref)

    @test sf.stepL == Array{Float64}[
        [
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        ],
        [
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0003082666266872036 0.0012311659404862342 0.0027630079602323555 0.00489434837048467 0.00761204674887136 0.010899347581163267 0.014735983564590839 0.019098300562505322 0.023959403439996987 0.029289321881345344 0.03505519516698174 0.0412214747707528 0.04775014352840509 0.05460095002604531 0.061731656763491016 0.06909830056250525 0.07665546361440947 0.08435655349597691 0.09215409042721551 0.10000000000000003 0.1078459095727845 0.11564344650402313 0.12334453638559058 0.1309016994374948 0.13826834323650902 0.14539904997395478 0.15224985647159497 0.1587785252292474 0.16494480483301846 0.1707106781186548 0.17604059656000315 0.18090169943749482 0.1852640164354093 0.18910065241883686 0.19238795325112873 0.19510565162951543 0.19723699203976772 0.19876883405951382 0.1996917333733128 0.2
            1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.155829702431226e-5 0.00024471741852423456 0.0005449673790581645 0.0009549150281252684 0.001464466094067269 0.0020610737385376426 0.002730047501302276 0.003454915028125273 0.004217827674798858 0.005000000000000013 0.005782172325201167 0.0065450849718747504 0.007269952498697732 0.007938926261462363 0.008535533905932738 0.009045084971874737 0.00945503262094184 0.009755282581475769 0.009938441702975689 0.01 0.009938441702975689 0.009755282581475766 0.009455032620941838 0.009045084971874734 0.008535533905932733 0.00793892626146236 0.007269952498697727 0.00654508497187473 0.005782172325201142 0.004999999999999992 0.004217827674798838 0.003454915028125251 0.002730047501302255 0.0020610737385376244 0.0014644660940672538 0.0009549150281252533 0.0005449673790581533 0.0002447174185242268 6.155829702431226e-5 0.0
        ],
        [
            0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2
            1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206 1.206
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        ],
    ]

    @test sf.stepR == Array{Float64}[
        [
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0001541333133436018 0.0006155829702431115 0.0013815039801161723 0.0024471741852423235 0.0038060233744356634 0.005449673790581611 0.007367991782295369 0.00954915028125261 0.011979701719998431 0.014644660940672611 0.017527597583490807 0.020610737385376332 0.023875071764202546 0.027300475013022654 0.030865828381745508 0.034549150281252626 0.03832773180720474 0.04217827674798846 0.046077045213607754 0.04999999999999996 0.053922954786392224 0.057821723252011514 0.061672268192795254 0.06545084971874736 0.06913417161825446 0.07269952498697733 0.07612492823579745 0.07938926261462365 0.08247240241650917 0.08535533905932738 0.08802029828000156 0.09045084971874738 0.09263200821770462 0.0945503262094184 0.09619397662556435 0.09755282581475769 0.09861849601988384 0.0993844170297569 0.0998458666866564 0.1
            1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.155829702431115e-5 0.0002447174185242318 0.0005449673790581606 0.0009549150281252627 0.0014644660940672626 0.0020610737385376356 0.0027300475013022595 0.003454915028125257 0.00421782767479884 0.004999999999999995 0.00578217232520115 0.006545084971874735 0.007269952498697732 0.007938926261462363 0.008535533905932738 0.009045084971874737 0.00945503262094184 0.009755282581475769 0.009938441702975689 0.01 0.009938441702975689 0.009755282581475769 0.009455032620941842 0.00904508497187474 0.008535533905932742 0.007938926261462366 0.007269952498697735 0.006545084971874737 0.0057821723252011546 0.005000000000000001 0.004217827674798842 0.003454915028125264 0.002730047501302267 0.0020610737385376317 0.00146446609406726 0.0009549150281252611 0.0005449673790581594 0.0002447174185242296 6.155829702431282e-5 0.0
        ],
        [
            0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
            1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        ],
        [
            0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.10015413331334361 0.10061558297024312 0.10138150398011618 0.10244717418524234 0.10380602337443569 0.10544967379058164 0.10736799178229542 0.10954915028125267 0.11197970171999849 0.11464466094067267 0.11752759758349088 0.12061073738537641 0.12387507176420255 0.12730047501302266 0.1308658283817455 0.13454915028125264 0.13832773180720476 0.14217827674798847 0.14607704521360776 0.15000000000000002 0.15392295478639226 0.15782172325201158 0.1616722681927953 0.1654508497187474 0.1691341716182545 0.17269952498697738 0.1761249282357975 0.1793892626146237 0.18247240241650925 0.1853553390593274 0.18802029828000155 0.19045084971874743 0.1926320082177046 0.19455032620941842 0.19619397662556434 0.19755282581475772 0.19861849601988385 0.1993844170297569 0.1998458666866564 0.2
            1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154 1.154
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.155829702431226e-5 0.00024471741852423456 0.0005449673790581645 0.0009549150281252684 0.001464466094067269 0.0020610737385376426 0.002730047501302276 0.003454915028125273 0.004217827674798858 0.005000000000000013 0.005782172325201167 0.0065450849718747504 0.007269952498697732 0.007938926261462363 0.008535533905932738 0.009045084971874737 0.00945503262094184 0.009755282581475769 0.009938441702975689 0.01 0.009938441702975689 0.009755282581475766 0.009455032620941838 0.009045084971874734 0.008535533905932733 0.00793892626146236 0.007269952498697727 0.00654508497187473 0.005782172325201142 0.004999999999999992 0.004217827674798851 0.003454915028125251 0.0027300475013022707 0.0020610737385376244 0.0014644660940672633 0.0009549150281252533 0.0005449673790581611 0.0002447174185242268 6.155829702431226e-5 0.0
        ],
    ]
end

sleep(0.1) # used for good printing
println("End test")

end # End Main Module 
