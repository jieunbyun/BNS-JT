#!/usr/bin/env python
import networkx as nx
import numpy as np
from pathlib import Path
import copy
import typer

from BNS_JT import trans, branch, variable, cpm, config, operation, brc

HOME = Path(__file__).parent

app = typer.Typer()

@app.command()
def setup():
    # # Illustrative example: Routine
    # Network
    node_coords = {'n1': (0, 0),
                   'n2': (1, 1),
                   'n3': (1, -1),
                   'n4': (2, 0)}

    arcs = {'e1': ['n1', 'n2'],
            'e2': ['n1', 'n3'],
            'e3': ['n2', 'n3'],
            'e4': ['n2', 'n4'],
            'e5': ['n3', 'n4']}
    n_arc = len(arcs)

    probs = {'e1': {0: 0.01, 1:0.99}, 'e2': {0:0.02, 1:0.98}, 'e3': {0:0.03, 1:0.97}, 'e4': {0:0.04, 1:0.96}, 'e5': {0:0.05, 1:0.95}}

    od_pair=('n1','n4')

    ODs = {'od1': od_pair}

    outfile = HOME.joinpath('./model.json')
    dic_model = trans.create_model_json_for_graph_network(arcs, node_coords, ODs, outfile)


@app.command()
def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))

    trans.plot_graph(cfg.infra['G'], HOME.joinpath('routine.png'))



def conn(comps_st, od_pair, arcs): # connectivity analysis
    G = nx.Graph()
    for k,x in comps_st.items():
        if x > 0:
            G.add_edge(arcs[k][0], arcs[k][1], capacity=1)


    if od_pair[0] in G.nodes and od_pair[1] in G.nodes:
        f_val, _ = nx.maximum_flow(G,od_pair[0],od_pair[1])
    else:
        f_val = 0

    if f_val > 0:
        sys_st = 's'

        p = nx.shortest_path( G, od_pair[0], od_pair[1] )

        min_comps_st = {}
        for i in range(len(p)-1):
            pair = [p[i], p[i+1]]
            if pair in arcs.values():
                a = list(arcs.keys())[list(arcs.values()).index(pair)]
            else:
                a = list(arcs.keys())[list(arcs.values()).index([pair[1], pair[0]])]
            min_comps_st[a] = 1

    else:
        sys_st = 'f'
        min_comps_st = None

    return f_val, sys_st, min_comps_st


@app.command()
def main():

    cfg = config.Config(HOME.joinpath('./config.json'))

    st_br_to_cs = {'f':0, 's':1, 'u': 2}

    od_pair = cfg.infra['ODs']['od1']

    probs = {'e1': {0: 0.01, 1:0.99}, 'e2': {0:0.02, 1:0.98}, 'e3': {0:0.03, 1:0.97}, 'e4': {0:0.04, 1:0.96}, 'e5': {0:0.05, 1:0.95}}

    varis = {}
    cpms = {}
    for k in cfg.infra['edges'].keys():
        varis[k] = variable.Variable(name=k, values=['f', 's'])

        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                          C = np.array([0, 1]).T, p = [probs[k][0], probs[k][1]])

    #sys_fun = lambda comps_st : conn(comps_st, od_pair, arcs)
    sys_fun = trans.sys_fun_wrap(cfg.infra['G'], od_pair, varis)

    brs, rules, sys_res, monitor = brc.run(varis, probs, sys_fun,
            max_sf=1000, max_nb=100)

    brc.plot_monitoring(monitor, HOME.joinpath('./monitor.png'))

    csys_by_od, varis_by_od = brc.get_csys(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable(name='sys', values=['f', 's', 'u'])
    cpms['sys'] = cpm.Cpm(variables = [varis[k] for k in ['sys'] + list(cfg.infra['edges'].keys())],
                          no_child = 1,
                          C = csys_by_od.copy(),
                          p = np.ones(csys_by_od.shape[0]))

    cpms_comps = {k: cpms[k] for k in cfg.infra['edges'].keys()}

    cpms_new = operation.prod_Msys_and_Mcomps(cpms['sys'], list(cpms_comps.values()))

    p_f = cpms_new.get_prob(['sys'], [0])
    p_s = cpms_new.get_prob(['sys'], [1])

    print(f'failure prob: {p_f:.5f}, survival prob: {p_s:.5f}')


def dummy():
    # # 1. Plotting of Branch and Bound results

    # In[15]:


    import matplotlib.pyplot as plt


    # In[22]:


    plt.plot(br_ns, pf_low, color='blue')
    plt.plot(br_ns, pf_up, color='blue')
    plt.xlabel('Number of branches')
    plt.ylabel('System failure probability bounds')

    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.show()


    # In[23]:


    plt.plot(sf_ns, pf_low, color='blue')
    plt.plot(sf_ns, pf_up, color='blue')
    plt.xlabel('Number of system function runs')
    plt.ylabel('System failure probability bounds')

    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.show()


    # In[24]:


    plt.plot(r_ns, pf_low, color='blue')
    plt.plot(r_ns, pf_up, color='blue')
    plt.xlabel('Number of rules')
    plt.ylabel('System failure probability bounds')

    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.show()


    # # 2. Inference -- New product function to not split composite states in a system event's C matrix

    # In[66]:


    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    result = brc.get_csys(brs, varis, st_br_to_cs)


    # In[27]:


    cpms = {}
    for e,v in arcs.items():
        cpms[e] = cpm.Cpm( variables=[varis[e]], no_child = 1, C=np.array([0,1]), p=[probs[e][0], probs[e][1]] )


    # In[67]:


    varis['sys'] = variable.Variable( name='sys', values=['f', 's', 'u'] ) #FIXME When trying to define B here, an error occurs. Can we revise this to give users liberty to define this?
    cpms['sys'] = cpm.Cpm( variables=[varis['sys']] + [varis[e] for e in arcs.keys()], no_child = 1, C=result[0], p=np.ones((len(result[0]),1)) )


    # In[68]:


    # New product function
    def prod_Msys_Mcomps(Msys, Mcomps_dict, varis):
        Cs = Msys.C
        ps = Msys.p
        v_sys = Msys.variables

        p_new = copy.deepcopy(ps)
        for i in range(len(v_sys)-Msys.no_child):
            c_name = v_sys[Msys.no_child+i].name
            M1 = Mcomps_dict[c_name]
            C1_list = [c[0] for c in M1.C] # TODO: For now this only works for marginal distributions of component evets
            for j in range(len(p_new)):
                st = Cs[j][Msys.no_child+i]
                st_set = varis[c_name].B[st]
                p_st = 0.0
                for k in st_set:
                    p_st += M1.p[C1_list.index(k)][0]

                p_new[j] *= p_st

        Mnew = cpm.Cpm( variables=v_sys, no_child=len(v_sys), C = Cs, p = p_new )

        return Mnew


    # In[70]:


    Msys = copy.deepcopy( cpms['sys'] )
    Mcomps = copy.deepcopy( {x:cpms[x] for x in arcs.keys()} )

    Mnew = prod_Msys_Mcomps(Msys, Mcomps, varis)
    print(Mnew)


    # In[78]:


    def get_prob( M, v_name, v_st ):

        v_names = [v.name for v in M.variables]
        v_loc = v_names.index(v_name)

        C_list = [c[v_loc] for c in M.C]
        prob = 0.0
        for i,c in enumerate(C_list):
            if c == v_st:
                prob += M.p[i]

        return prob


    # In[80]:


    pf = get_prob( Mnew, 'sys', 0 )
    print("failure prob: ", pf)

    ps = get_prob( Mnew, 'sys', 1 )
    print("survival prob: ", ps)



    # # 3. B&B + MCS

    # In[ ]:


    varis_c = {e:varis[e] for e in arcs.keys()}

    # Incomplete B&B (by setting max_br=10)
    brs2, rules2, sys_res2, _, _, _, _, _ = brc.run(sys_fun, varis_c, probs, max_br=8,
                                                                                          output_path=HOME, key='conn')

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    result2 = brc.get_csys(brs2, varis_c, st_br_to_cs)

    varis['sys2'] = variable.Variable( name='sys2', values=['f', 's', 'u'] ) #FIXME When trying to define B here, an error occurs. Can we revise this to give users liberty to define this?
    cpms['sys2'] = cpm.Cpm( variables=[varis['sys2']] + [varis[e] for e in arcs.keys()], no_child = 1, C=result2[0], p=np.ones((len(result2[0]),1)) )


    # In[89]:


    print(cpms['sys2'])


    # In[97]:


    def get_Msub( M, v_name, v_st ):

        v_names = [v.name for v in M.variables]
        v_loc = v_names.index(v_name)

        Csub = np.empty((0,len(v_names)), dtype=int)
        psub = np.empty((0,1), dtype=float)

        C_list = [c[v_loc] for c in M.C]
        for i,c in enumerate(C_list):
            if c == v_st:
                Csub=np.vstack([Csub, M.C[i]])
                psub =np.vstack([psub, M.p[i]])

        Msub = cpm.Cpm( variables=M.variables, no_child=M.no_child, C=Csub, p=psub )

        return Msub


    # In[98]:


    Msys2 = copy.deepcopy( cpms['sys2'] )
    Mnew2 = prod_Msys_Mcomps(Msys2, Mcomps, varis)

    pf2_bnb = get_prob( Mnew2, 'sys2', 0 )
    print(pf2_bnb)

    unk_pr = get_prob( Mnew2, 'sys2', 2 )
    print(unk_pr)

    Munk = get_Msub( Mnew2, 'sys2', 2 )
    print(Munk)


    # In[ ]:


    cov = 1.0
    cov_t = 0.05
    no_pf_mcs = 0
    no_mcs = 0
    while (cov > cov_t) or (no_pf_mcs < 1):

        no_mcs += 1
        


    # In[113]:


    import random
    Mmcs = Munk
    c_names = list(arcs.keys())


    # In[112]:


    samp_ind = random.choices( list(range(len(Mmcs.p))), weights=(k[0] for k in Mmcs.p) )
    print(samp_ind)


    v_names = [v.name for v in Mmcs.variables]
    c_locs = {c:v_names.index(c) for c in c_names}

    samp_e = Mmcs.C[samp_ind]
    samp = np.empty((0,len(1+c_names)), dtype=int)
    for c in c_names:
        c_st = samp_e[c_locs[c]]
        c_B = varis[c].B

        if len(c_B) < 2:
            samp[c_locs[c]] = c_B[0]
        else:
            ##############290224
            pass

if __name__=='__main__':

    app()
