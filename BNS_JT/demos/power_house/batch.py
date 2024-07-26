from pathlib import Path
import numpy as np
import random
import copy
import pdb
import time
import pandas as pd
import matplotlib.pyplot as plt
import typer
import json

from BNS_JT import cpm, variable, operation

HOME = Path(__file__).parent

app = typer.Typer()

@app.command()
def get_lins(pole1, paths):
    """
    Get link sets of a pole's end (to be connected to substations) given paths information
    [INPUT]
    - pole1: a string
    - paths: a dictionary of substation: LIST of paths
    [OUTPUT]
    - lins: a list of link-sets

    [EXAMPLE]
    lins = get_lins( 'p1',  {'s0': [['p0', 'p1'], ['p2', 'p3', 'p5']], 's1':[['p4','p5'], ['p6', 'p7']]})
    """

    lins = []
    for s, ps in paths.items():
        for p in ps:
            if pole1 in p:
                idx = p.index(pole1)
                lin1 = [s] + p[:(idx+1)]
                lins += [lin1]
    return lins


def read_model_json(model_json):

    with open(model_json, 'r') as f:
        model = json.load(f)

    haz = {}
    for k,v in model['hazard'].items():
        haz[k] = v

    pfs = {}
    for k,v in model['fail_probs'].items():
        pfs[k] = v

    rep_pri = model["repair_priority"]

    paths = {}
    for k,v in model["paths"].items():
        paths[k] = v

    houses = {}
    for k,v in model["houses"].items():
        houses[k] = v

    if "locations" in model:
        locs = {}
        for k,v in model["locations"].items():
            locs[k] = (v["pos_x"], v["pos_y"])
    else:
        locs = None

    return haz, pfs, rep_pri, paths, houses, locs


#TODO
def write_output_jason(version):
    if version is None:
        filename = "result_" + version +".json"
    else:
        filename = "result.json"

    #with open(filename, 'w') as w:



def quant_cpms(haz, pfs, rep_pri, lins):

    """
    Quantify CPMs from user inputs
    """

    cpms = {}
    varis = {}

    ## Hazard
    vals_h = list(haz.keys())
    p_h = list(haz.values())

    varis['haz'] = variable.Variable(name='haz', values=vals_h) # values=(mild, medium, intense)
    cpms['haz'] = cpm.Cpm(variables=[varis['haz']], no_child=1, C=np.array(range(len(p_h))), p=p_h)

    ## Structures
    C_x = []
    for hs in range(len(vals_h)):
        C_x += [[0, hs], [1, hs]]
    C_x = np.array(C_x)

    for s, pf in pfs.items():
        name = f'x{s}'
        varis[name] = variable.Variable(name=name, values=['fail','surv']) # values=(failure, survival)

        p_x = []
        for p in pf:
            p_x += [p, 1-p]

        cpms[name] = cpm.Cpm(variables = [varis[name], varis['haz']], no_child = 1,
                        C=C_x, p=p_x )

    ## Number of damaged structures so far (following repair priority)
    for i, s in enumerate(rep_pri):
        name = f'n{s}'
        varis[name] = variable.Variable(name=name, values=list(range(i+2)))

        if i < 1: # first element
            cpms[name] = cpm.Cpm(variables = [varis[name], varis[f'x{s}']], no_child = 1, C=np.array([[1,0], [0, 1]]), p=np.array([1,1]))

        else:
            t_old_vars = varis[n_old].values

            Cx = np.empty(shape=(0,3), dtype=int)
            for v in t_old_vars:
                Cx_new = [[v, 1, v], [v + 1, 0, v]]
                Cx = np.vstack([Cx, Cx_new])

            cpms[name] = cpm.Cpm(variables = [varis[name], varis[f'x{s}'], varis[n_old]], no_child = 1, C=Cx, p=np.ones(shape=(2*len(t_old_vars)), dtype=np.float32))

        n_old = copy.deepcopy(name)

    ## Closure time
    for s in rep_pri:
        name = f'c{s}'
        name_n = f'n{s}'
        vals = varis[name_n].values
        varis[name] = variable.Variable(name=name, values=vals)

        try:
            cst_all = varis[name_n].B.index(set(vals)) # composite state representing all states
        except AttributeError:
            print(name_n)
        else:
            Cx = np.array([[0, 1, cst_all]])
            for v in vals:
                if v > 0:
                    Cx = np.vstack((Cx, [v, 0, v]))

            cpms[name] = cpm.Cpm(variables = [varis[name], varis[f'x{s}'], varis[name_n]], no_child = 1, C=Cx, p=np.ones(shape=(len(Cx), 1), dtype=np.float32))

    ## Power-cut days of houses
    for h, sets in lins.items():
        if len(sets) == 1:
            vars_h = [varis[f'c{x}'] for x in sets[0]]

            cpms[h], varis[h] = operation.sys_max_val(h, vars_h)

        else:
            names_hs = [h+str(i) for i in range(len(sets))]
            for h_i, s_i in zip(names_hs, sets):
                vars_h_i = [varis['c'+x] for x in s_i]

                cpms[h_i], varis[h_i] = operation.sys_max_val( h_i, vars_h_i )

            vars_hs = [varis[n] for n in names_hs]
            cpms[h], varis[h] = operation.sys_min_val( h, vars_hs )

    return cpms, varis


def get_pole_conn(paths,houses):

    poles = []
    for _, ps in paths.items():
        for p1 in ps:
            poles += p1
    poles = set(poles)

    conn = {}
    for pl in poles:
        for sub, path in paths.items():
            for path1 in path:
                if pl in path1:
                    pl_idx = path1.index(pl)
                    if pl_idx < 1:
                        pl_tail = sub
                    else:
                        pl_tail_pl = path1[pl_idx-1]

                        for h, pls_h in houses.items():
                            if pl_tail_pl in pls_h:
                                pl_tail = h
                                break # a pole is connected to only one house
                    break # a pole appears only in only one path
            
        for h, pls_h in houses.items():
            if pl in pls_h:
                pl_head = h
                break # a pole's head is connected to exactly one house
        conn[pl] = (pl_tail, pl_head)

    return conn


def add_pole_loc( locs_sub_hou, conn_pol ):

    locs = copy.deepcopy(locs_sub_hou)

    for pl, pair in conn_pol.items():
        locs[pl] = tuple( [0.5*sum(tup) for tup in zip(locs_sub_hou[pair[0]], locs_sub_hou[pair[1]])] )

    return locs


def plot_result(locs, conn, avg_cut_days, pfs_mar):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # draw edges
    for _, pair in conn.items():
        loc0, loc1 = locs[pair[0]], locs[pair[1]]
        plt.plot([loc0[0], loc1[0]], [loc0[1], loc1[1]], color='grey')

    # draw houses
    plt.scatter([locs[h][0] for h in avg_cut_days.keys()], [locs[h][0] for h in avg_cut_days.keys()], c = [avg_cut_days[h] for h in avg_cut_days.keys()], cmap='Reds', s=200, marker="X")
    cb1 = plt.colorbar()
    cb1.ax.set_xlabel('Avg. \n cut days', size = 15)

    # draw structures    
    plt.scatter( [locs[s][0] for s in pfs_mar.keys()], [locs[s][1] for s in pfs_mar.keys()], c = [pfs_mar[s] for s in pfs_mar.keys()], cmap='Reds' )
    cb2 = plt.colorbar()
    cb2.ax.set_xlabel('Fail. \n prob.', size = 15)

    # texts
    tx, ty = 0.10, 0.01
    for x in list(avg_cut_days.keys()) + list(pfs_mar.keys()):
        ax.text(locs[x][0]+tx, locs[x][1]+ty, x)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig('result.png', dpi=200)
    print('result.png is created.')


@app.command()
def main(h_list, model_json, plot: bool=False, version=None):

    """
    Compute the marginal distributions of power cut days for the houses in h_list

    INPUT:
    h_list: a list of strings
    model_json: a sting as json file name that contains problem information

    OUTPUT:
    Mhs: a dictionary {house: CPM} (representing marginal distributions)
    avg_cut_days: a dictionary {house: # of days} 
    """

    """
    ######### USER INPUT (in model.json) ###################
    # network topology
    ## Topology is assumed radial--Define each substation and connected paths (in the order of connectivity)
    paths = {'s0': [['p0', 'p1'], ['p2', 'p3', 'p5']], 's1':[['p4','p5'], ['p6', 'p7']]}
    ## House--connected poles (NB the edges are uni-directional in the direction away from substation; all poles must be connected to a house at its head)
    houses = {'h0':['p0'], 'h1':['p1'], 'h2':['p2'], 'h3':['p3', 'p4'], 'h4':['p5'], 'h5':['p6'], 'h6':['p7']}

    # Random variables
    haz ={'mild': 0.5, 'medi': 0.2, 'inte': 0.3 } # {scenario: prob}
    pfs = {'s0': [0.001, 0.01, 0.1], 's1':[0.005, 0.05, 0.2], # {structure: failure probability for each hazard scenario}
          'p0': [0.001, 0.01, 0.1], 'p1':[0.005, 0.05, 0.2],
          'p2': [0.001, 0.01, 0.1], 'p3':[0.005, 0.05, 0.2],
          'p4': [0.001, 0.01, 0.1], 'p5':[0.005, 0.05, 0.2],
          'p6': [0.001, 0.01, 0.1], 'p7':[0.005, 0.05, 0.2]}

    # Repair priority
    rep_pri = ['s0', 's1', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    ###############################################

    ############### if plot==True ##############
    # Locations for mapping
    locs = {"h0": (-0.8, -1.0), "h1": (-2*0.8, -2*1.0), "h2": (0.8, -1.0), "h3": (2*0.8, -2*1.0), "h4": (0.8, -3*1.0), "h5": (5*0.8,-2*1.0), "h6": (6*0.8, -3*1.0)}
    locs["s0"], locs["s1"] = (0.0, 0.0), (3.5*0.8, 0.0)
    #########################################
    """

    haz, pfs, rep_pri, paths, houses, locs = read_model_json(HOME.joinpath(model_json))

    ## Compute link-sets from paths and houses
    lins = {}
    for h, pls in houses.items():
        lins_h = []
        for pl in pls:
            lins_pl = get_lins(pl, paths)
            lins_h += lins_pl

        lins[h] = lins_h

    # CPMs
    cpms, varis = quant_cpms(haz, pfs, rep_pri, lins)

    # Inference
    ## Variable Elimination order
    VE_ord = ['haz']
    for s in rep_pri:
        VE_ord += ['x'+s, 'n'+s, 'c'+s]
    for h, p in lins.items(): # Actually, VE order of houses does not matter as the marginal distribution is computed for "one" house at a time
        if len(p) < 2:
            VE_ord += [h]
        else: # there are more than 1 path
            VE_ord += [h+str(i) for i in range(len(p))] + [h]

    ## Get P(H*) for H* in h_list
    cond_names = ['haz']

    Mhs = {}
    for h in h_list:
        st=time.time()
        Mhs[h] = cpm.cal_Msys_by_cond_VE( cpms, varis, cond_names, VE_ord, h )
        en = time.time()

        print( h + " done. " + "Took {:.2f} sec.".format(en-st) )


    avg_cut_days = {}
    for h in h_list:
        cpm.get_means(Mhs[h], [h])
        avg_cut_days[h] = Mhs[h].means[0]

    # Plot
    if plot:

        # Get information about paths
        conn = get_pole_conn( paths, houses )
        locs = add_pole_loc( locs, conn )

        # failure prob. from marginal distribution
        pfs_mar = {}
        for s in pfs.keys():
            Mx = cpm.cal_Msys_by_cond_VE(cpms, varis, cond_names, VE_ord, 'x' + s)
            pf = cpm.get_prob(Mx, ['x'+s], [0])
            pfs_mar[s] = pf

        plot_result(locs, conn, avg_cut_days, pfs_mar)

    return Mhs, avg_cut_days


app.command()
def batch():

    Mhs, avg_cut_days = main(['h0'], HOME.joinpath("model.json"), plot=True)

if __name__ == "__main__":
    app()

