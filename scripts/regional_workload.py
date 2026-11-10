"""Deterministic controlled workload for the 16x16 region pilot."""
import math
import random


def controlled(config, seed, size, profile, diagnostic=False):
    hot_fraction, distribution, update_fraction = config["profiles"][profile]
    rng = random.Random(seed)
    motion = random.Random(seed ^ 0x160B)
    query_rng = random.Random(seed ^ 0xA72B)
    x0,y0,x1,y1 = config["world_bounds_m"]
    side = config["region_side"]
    width,height = (x1-x0)/side,(y1-y0)/side
    anchors = config["cluster_regions_xy"]
    cluster_mass = config["cluster_mass"][distribution]
    sigma = config["cluster_sigma_m"][distribution]

    def clamp(x,y):
        return min(x1,max(x0,x)),min(y1,max(y0,y))

    def cell_center(x,y):
        return x0+(x+0.5)*width,y0+(y+0.5)*height

    def initial_position():
        if rng.random() >= cluster_mass:
            return rng.uniform(x0,x1),rng.uniform(y0,y1)
        x,y = cell_center(*rng.choice(anchors))
        return clamp(rng.gauss(x,sigma),rng.gauss(y,sigma))

    def hot_cell(phase, offset):
        if phase == "fixed":
            return tuple(anchors[0])
        if phase == "migrate":
            start,end = anchors[0],anchors[1]
            fraction = (offset+1)/config["steps_per_phase"]
            return tuple(round(a + fraction*(b-a)) for a,b in zip(start,end))
        return tuple(anchors[2+offset%2])

    current = {i: initial_position() for i in range(1,size+1)}
    initial = {i:(0,point) for i,point in current.items()}
    events = []
    phases = config["phases"]
    duration = config["steps_per_phase"]
    schedule = []
    area = (x1-x0)*(y1-y0)
    scales = config["query_area_fractions"]
    per_step = config["queries_per_step"]
    for phase_num,phase in enumerate(phases):
        for offset in range(duration):
            step = phase_num*duration+offset
            stamp = (step+1)*1000
            cell = hot_cell(phase,offset)
            hot_count = round(per_step*hot_fraction)
            schedule.append({"step":step,"phase":phase,"center_region_xy":cell,
                             "forced_hot_queries":hot_count})
            if diagnostic:
                # Exercise construction, replacement and teardown in occupied anchor regions.
                for x,y in anchors:
                    events.append((step,stamp,"S",side*y+x,((step+1)%3,)))
            for ident in motion.sample(range(1,size+1),round(size*update_fraction)):
                x,y = current[ident]
                sd = config["movement_sigma_m"]
                current[ident] = clamp(motion.gauss(x,sd),motion.gauss(y,sd))
                events.append((step,stamp,"U",ident,current[ident]))
            hot_positions = set(query_rng.sample(range(per_step),hot_count))
            for i in range(per_step):
                if i in hot_positions:
                    cx,cy = cell_center(*cell)
                    x = cx + query_rng.uniform(-0.49,0.49)*width
                    y = cy + query_rng.uniform(-0.49,0.49)*height
                else:
                    x,y = query_rng.uniform(x0,x1),query_rng.uniform(y0,y1)
                a = area*scales[(i//2)%len(scales)]
                if i % 2:
                    events.append((step,stamp,"C",None,(x,y,math.sqrt(a/math.pi))))
                else:
                    half = math.sqrt(a)/2
                    events.append((step,stamp,"R",None,(x-half,y-half,x+half,y+half)))
        events.append((step,stamp,"P",phase_num,()))
    return initial,events,len(phases)*duration,{
        "trajectory_records":0,
        "hot_fraction":hot_fraction,"distribution":distribution,"update_fraction":update_fraction,
        "phase_names":phases,"phase_window_seconds":duration,
        "hotspot_schedule":schedule,
        "query_source":"seeded RECT/CIRCLE, forced hot-cell centers plus uniform background",
    }
