module Test_CT

include(joinpath(@__DIR__, "..", "..", "src", "ZMPBipedRobot.jl"))
import .ZMPBipedRobot
const ZMProbot = ZMPBipedRobot
using Test, Plots

sleep(0.1) # used for good printing
println("Started test")

# If the maximal  error is bounded within -2cm and + 2cm with the ZMP generated by Preview Controller and the reference ZMP 
@testset "CoM Trajectory Generator" begin
    br = ZMProbot.BipedRobot(; readFile = true, paramFileName = "param_test.jl")
    pc = ZMProbot.PreviewController(; br = br)
    fp = ZMProbot.FootPlanner(; br = br)
    zt = ZMProbot.ZMPTrajectory(; br = br, fp = fp)
    ct = ZMProbot.CoMTrajectory(; br = br, pc = pc, zt = zt, check = false)

    tref = reduce(vcat, zt.timeVec)
    p = reduce(hcat, ct.p)
    zmp = reduce(hcat, zt.ZMP)

    @test [isapprox(p[1, i], zmp[1, i]; atol = 0.02) for i in 1:length(p[1, :])] == ones(length(p[1, :]))
    @test [isapprox(p[2, i], zmp[2, i]; atol = 0.02) for i in 1:length(p[1, :])] == ones(length(p[1, :]))

    p_compare = [
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 9.723912904543997e-23 -1.5378475029592325e-6 -7.243235737782787e-6 -1.8480398366660927e-5 -3.4828182838114304e-5 -5.4436983746927565e-5 -7.464376168164115e-5 -9.25247346229958e-5 -0.0001052701761958389 -0.00011037991492095239 -0.00010750962994660394 -9.899983695902508e-5 -8.76526829640458e-5 -7.569906101328443e-5 -6.452087347834994e-5 -5.4756531810964876e-5 -4.653819096254919e-5 -3.972290637410014e-5 -3.4062026622820295e-5 -2.9301366822693236e-5 -2.5226670660315706e-5 -2.1674395279788843e-5 -1.852520021461094e-5 -1.5692183685278805e-5 -1.3110804851140577e-5 -1.0733605290534534e-5 -8.530216449782371e-6 -6.491097891393006e-6 -4.631276408401205e-6 -2.9875718544440525e-6 -1.5995209225213386e-6 -4.6158418967407267e-7 5.651379427726346e-7 1.8851541632270396e-6 4.32641595383864e-6 9.264076966930826e-6 1.8587571534068566e-5 3.4311630578595136e-5 5.756855922911781e-5 8.672678900680833e-5 0.00011455449957758498 0.0001248146466180371 8.956727213396944e-5 -3.018196346747968e-5 -0.0002761759045027464 -0.0006712339972737381 -0.0011786198384059398 -0.0016405350364256481 -0.00170384753794968 -0.0007603332904405609 0.0020417824393874273 0.00762025409394465 0.01649471302548186 0.028437652993965747 0.04252715296391325 0.05738320151812726 0.07146883471968578 0.08340511516951694 0.09227251008575596 0.09784737714506515 0.10065232104713479 0.10160429270776532 0.10155226364552782 0.10110222572117093 0.1006059275295359 0.1002205272054974 0.09998262682610684 0.09986954734840543 0.0998397916176095 0.09985461603731886 0.09988629091543615 0.09991873813057776 0.09994483985802499 0.09996304524760671 0.09997454398714381 0.09998138997086327 0.09998549337902692 0.09998822773920459 0.09999039303343461 0.09999234200421998 0.09999415687291584 0.09999582926200716 0.09999743673805397 0.09999932289615182 0.10000227424806804 0.10000764636009887 0.10001732587042697 0.1000333347125489 0.10005681666801862 0.10008615214429535 0.10011411983718618 0.10012449109121711 0.10008933228345049 0.09996965374813516 0.09972371610451557 0.09932870254183451 0.09882135154587367 0.09835946321317011 0.09829617102769383 0.09923970025536998 0.10204182665855538 0.10762184339396452 0.11650201222997504 0.12845619175678188 0.14256204027389005 0.1574376971538096 0.1715435357381595 0.18349769510013805 0.192377832933682 0.19795780690369805 0.2007598775142532 0.20170333629407522 0.2016399569833063 0.20117796239197633 0.2006704830587502 0.2002753155595508 0.2000291941465268 0.19990929684483255 0.19987387785508368 0.19988393939829213 0.19991153752490157 0.19994043056695876 0.19996338135369846 0.1999787521262868 0.19998766802396306 0.1999921354742835 0.19999403428047147 0.1999947284211467 0.19999503289675788 0.19999533726152452 0.19999576326568672 0.19999629698488255 0.19999687707907055 0.19999744262809477 0.19999795218612385 0.19999838616012913 0.19999874174872392 0.1999990261496614 0.1999992508349482 0.19999942777648186 0.19999956748671952 0.19999967835115984 0.19999976670181777 0.1999998372097802 0.19999989333611692 0.19999993771412491 0.19999997242418524 0.19999999916953737 0.20000001937913034 0.2000000342648809
        1.18 1.1834408861150767 1.1830153116797164 1.1812208230064012 1.1794248090184156 1.1781810412149543 1.1775668999604605 1.1774438776262612 1.1776202833132554 1.1779321397533717 1.1782696233334458 1.178574197607104 1.178823910237501 1.179018089443476 1.179165991359911 1.1792794471441488 1.179369049919319 1.1794427922255442 1.179506079036001 1.1795623197178855 1.1796136198697786 1.179660883600466 1.1797040490256727 1.179742862056207 1.1797772210543502 1.1798072592694138 1.1798333003629906 1.1798557750095628 1.1798751446681568 1.1798918495855713 1.1799062817399129 1.1799187764808219 1.1799296153220344 1.1799390337028397 1.1799472296433242 1.1799543711014377 1.17996060117909 1.1799660411747213 1.1799707920603595 1.1799749355132827 1.1799785363187598 1.179981648772545 1.1799843303636048 1.179986665830513 1.179988802515016 1.1799909922484468 1.1799936242717557 1.179997217322665 1.1800023190615472 1.1800092445505914 1.1800175874991876 1.1800254831708639 1.1800287234069096 1.1800200555011895 1.179989351056118 1.1799257657166986 1.179823374043961 1.1796917341073987 1.179571879215429 1.179555628667213 1.1798011250215101 1.180529833441597 1.1819815728398002 1.1842935013342688 1.1874075338374916 1.191083644490314 1.1949615894605425 1.1986396825390606 1.2017571208264515 1.20407263540429 1.2055261301676417 1.2062532146635794 1.2064941225086867 1.2064717726453664 1.2063454636373216 1.2062077285087327 1.2060999267739398 1.2060316784338039 1.205996975769768 1.2059848358589869 1.2059849873496518 1.2059900536164068 1.2059957269701924 1.2060000638521817 1.2060025933976262 1.2060035747045763 1.2060035048728988 1.2060028569529437 1.2060019806991655 1.2060010958961125 1.2060003230355336 1.2059997143849146 1.2059992634898358 1.2059988822144552 1.2059983444101203 1.2059972087966906 1.205994755226634 1.2059900004888444 1.2059818988147502 1.2059698644058796 1.2059547488057336 1.2059403153073445 1.2059350093654833 1.2059533614146274 1.2060156497345411 1.2061435815117076 1.2063490235494299 1.206612873464574 1.206853076541386 1.2068860044879521 1.206395381115808 1.2049382838341336 1.2020358811993737 1.1974152302175456 1.1911932154874687 1.1838496740746596 1.1761041356098345 1.168758590985087 1.1625331284484228 1.1579088272153317 1.1550045815146226 1.1535489950294004 1.1530628191730194 1.153101674542846 1.1533481252575941 1.1536178248806461 1.1538283872419897 1.1539606418259463 1.1540265304007413 1.1540478900863391 1.1540451319059382 1.1540328981258625 1.1540197194529809 1.1540094212689007 1.1540029003918844 1.15399960530482 1.1539985173374048 1.1539986747770605 1.1539993733093528 1.154000181313629 1.1540008770690702 1.154001376700419 1.1540016903230943 1.1540019220562694 1.15400231391785 1.154003317989692 1.1540056595356682 1.1540103225142775 1.154018350839022 1.1540303273396941 1.154045397372155 1.1540597949204912 1.1540650723585906 1.1540466976132806 1.1539843911968655 1.1538564450286082 1.1536509916253694 1.1533871328335268 1.1531469229295426 1.1531139898368508 1.1536046094314263 1.1550617040391218 1.1579633051996503 1.1625809882804803 1.168797159127844 1.176132199550218 1.1838675416102025 1.191202579114594 1.1974187441168382 1.2020364184073815 1.2049380078057426 1.2063950876436882 1.2068856894149302 1.2068527353867096 1.2066125013616833 1.206348615176781 1.2061431310206414 1.2060151506833805 1.2059528067248482 1.2059343913222322 1.205939625628029 1.2059539787896287 1.2059690051435286 1.2059809413667106 1.2059889358317142 1.205993573618977 1.2059958980793903 1.2059968867194386 1.2059972488168382 1.2059974081805334 1.20599756738717 1.2059977897549836 1.2059980680510183 1.205998370385731 1.205998665087456 1.2059989306107155 1.205999156773033 1.2059993421231283 1.205999490408842 1.2059996076002104 1.2059996999268177 1.205999772858972 1.2059998307606155 1.205999876927556 1.205999913791622 1.2059999431551576 1.2059999663898167 1.205999984579504 1.205999998611796 1.2060000092314513 1.206000017070194
    ]

    @test isapprox(p, p_compare; atol = 1e-6)
end

sleep(0.1) # used for good printing
println("End test")

end # End Main Module 
