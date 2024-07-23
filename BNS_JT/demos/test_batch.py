import n3e3.batch as n3e3
import pf_map.batch as pf_map
import max_flow.batch as max_flow
import toy.batch as toy
import bridge.batch as bridge
import routine.batch as routine
import rbd.batch as rbd
import SF.batch as sf
import EMA.batch as ema
import road.batch as road
import power_house.batch as power_house

def test_batch():

    n3e3.main()
    ema.main('od2', max_sf=10)
    rbd.main()
    pf_map.debug()
    max_flow.debug()
    toy.main()
    bridge.main()
    routine.main()
    sf.main(max_sf=10)
    ema.main(max_sf=10)
    road.main()
    #power_house.batch()
