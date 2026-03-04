import random
import warnings
import quimb as qu
import quimb.tensor as qtn
import math
import cotengra as ctg
import numpy as np
import os
import multiprocessing
import ray
ray.shutdown()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm
import autoray
from itertools import permutations
from more_itertools import distinct_permutations 
from tqdm.contrib.itertools import product
from itertools import islice
from quimb.tensor.tn2d.core import *
# from quimb.tensor.gating import maybe_factor_gate_into_tensor
import collections
from quimb.utils import frequencies 
from math import pi, log2, log10
import re
import time
import xyzpy as xyz
import pandas as pd

from collections import Counter
from quimb.tensor.belief_propagation.d2bp import (
    contract_d2bp,
    compress_d2bp,
    sample_d2bp,
)
from quimb.tensor.belief_propagation.l2bp import (
    contract_l2bp,
    compress_l2bp,
    L2BP,
)

from quimb.tensor.belief_propagation.l1bp import (
    contract_l1bp, L1BP,
)
from quimb.tensor.circuit import Gate

import math
from itertools import combinations
import autoray as ar

from quimb.utils import oset

from quimb.tensor.belief_propagation.bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    create_lazy_community_edge_map,
)



def add_diagonalu_tags(p, ):
    def diagonals_sum(Lx, Ly):
        """
        Diagonals with constant (x + y).
        Returns: dict {D: [(x,y), ...]}
        where D = x + y
        """
        diags = {d: [] for d in range(Lx + Ly - 1)}
        
        for x in range(Lx):
            for y in range(Ly):
                d = x + y
                diags[d].append((x, y))
        
        return diags


    for d, l in diagonals_sum(p.Lx, p.Ly).items():
        for count, cor in enumerate(l):
            p[cor].add_tag(f"Du{d}")
            p[cor].add_tag(f"du{count}")

    return p





def circ_hydro(Lx=14, Ly=4, steps = 14, dt = 0.25):
    from circuits import trotter_2D_square_inhomogeneous as circ
    from qiskit import transpile
    
    ## Functions to index 2d grid
    def pos_to_index(pos,dims):
        return int(pos[0]+dims[0]*pos[1])
    
    def index_to_pos(i,dims):
        x = int(np.mod(i,dims[0]))
        y = int((i-x)/dims[0])
        return (x,y)
    
    dims = Lx, Ly
    N = Lx*Ly
    J = -1
    h = 2
    hz = 0
    
    theta_min = np.arcsin(h/(4*J)) # mean-field (MF) ground-state angle
    delta_theta = 2*np.pi/9 # deviation from MF angle
    theta_cold = (theta_min+delta_theta)/2 #start everything other than the hot stripe in what we call the "intermediate temp quench" state
    theta_hot = np.pi/2 # start the hot stripe in the |1> state

    
    # hot stripe
    thetas = np.ones(N)*theta_cold # array of initial state angles for each site
    for y in range(Ly):
        hot_x_pos = Lx//2 # x-location of hot stripe
        j = pos_to_index((hot_x_pos,y),dims)
        thetas[j]=theta_hot
    
    # visualize thetas
    scale = 0.75
    plt.figure(figsize=(scale*Lx,scale*Ly))
    theta_grid = np.zeros((Lx,Ly))
    for j in range(N):
        (x,y) = index_to_pos(j,dims)
        theta_grid[x,y] = thetas[j]
    # plt.pcolor(theta_grid.T,vmin=0,vmax = np.pi,cmap = 'gnuplot2')
    # plt.colorbar()
    # plt.show()  
    qc = circ(Lx, Ly, steps, dt, J, h, hz, thetas, order = 'second')

    two_qubit_count = sum(1 for inst, qargs, _ in qc.data if len(qargs) == 2)
    # print("Two-qubit gates:", two_qubit_count)

    from qiskit.qasm2 import dumps
    basis_gates = [ 'rx', 'ry', 'rz', "rzz"]
    qc = transpile(qc, basis_gates=  basis_gates, optimization_level=0, seed_transpiler=1234)
    
    qasm_str = dumps(qc)
    circ = qtn.Circuit.from_openqasm2_str(qasm_str)

    return circ




def circ_info(Lx=4, Ly=5, steps=22):
    from floquet.circuits import (trotter_2D_square, circ_gates, Z1_ti, Z2_ti, mpo_z2_, simulate, transpile)
    import math
    
    
    x_dim = Lx
    y_dim = Ly
    L = x_dim*y_dim
    steps = steps
    cycle_peps = False
    theta, J, h, hz, dt = -math.pi / 36, 1, 2, 0.0, 0.25
    order = 'second'
    obs = {'Z': Z1_ti(x_dim,y_dim), 'Z2': Z2_ti(x_dim,y_dim)}
    circ_info = {"x_dim": x_dim, "y_dim": y_dim, "theta": theta, "J": J, "h": h, "dt": dt, "hz": hz, "order": order}
    basis_gates = [ 'rx', 'ry', 'rz', "rzz"]
    
    qc_ = trotter_2D_square(x_dim, y_dim, 1, dt, J, h, hz, 0, order = order)
    
    
    qc_0 = trotter_2D_square(x_dim, y_dim, 0, dt, J, h, hz, theta, order = order)
    qc_0 = transpile(qc_0, basis_gates=  basis_gates, optimization_level=0)
    
    # build-up the circuit init: |cos> + |sin>
    gate_l_0, where_l_0  = circ_gates(qc_0, 1, t_im=0)
    
    # build-up the repeated circuit
    gate_l, where_l  = circ_gates(qc_, steps, t_im=0, dt=dt)
    
    n_gate_0 = len(where_l_0[0])
    n_gate_ = len(where_l[0])
    
    
    gate_l = gate_l_0 + gate_l
    where_l = where_l_0 + where_l
    
    where_l = [x for group in where_l for x in group]
    gate_l = [x for group in gate_l for x in group]
    
    
    gate_info = { (where_l[count], count): gate  for count, gate in enumerate(gate_l)}
    depth_map = {n_gate_0+n_gate_*i:i  for i in range(steps+1)}

    return gate_info, depth_map, circ_info



def map_info(Lx, Ly, pepo):
    
    # 16, 0, 32, 4, 20, 24, 8, 36, 12, 28
    site = { (0,0): 16,  (0,1): 0, (0,2): 32, (0,3): 4, (0,4): 20,  (0,5): 24,   (0,6): 8,  (0,7): 36,  (0,8): 12,  (0,9): 28  }
    # 17, 1, 5, 21, 34, 25, 9, 13, 29, 38
    site |= { (1,0): 17,  (1,1): 1, (1,2): 5, (1,3): 21, (1,4): 34,  (1,5): 25,   (1,6): 9,  (1,7): 13,  (1,8): 29,  (1,9): 38  }
    # 18, 2, 33, 6, 22, 26, 10, 37, 14, 30
    site |= { (2,0):18,  (2,1):2, (2,2):33, (2,3):6, (2,4):22,  (2,5):26,   (2,6):10,  (2,7):37,  (2,8):14,  (2,9):30  }
    # 19, 3, 7, 23, 35, 27, 11, 15, 31, 39
    site |= { (3,0):19,  (3,1):3, (3,2):7, (3,3):23, (3,4):35,  (3,5):27,   (3,6):11,  (3,7):15,  (3,8):31,  (3,9):39  }
    
    site_ = {v: k for k, v in site.items()}
    for key, value in site.items():
        x, y = key
        t = pepo[f"I{x},{y}"]
        t.add_tag(f"G{value}")
    
    fix = { f"I{i},{j}":(i,j) for i in range(Lx) for j in range(Ly)}
    
    pos_map = {}
    pos_map_ = {}
    
    for count, i in enumerate(site.values()):
        pos_map[i] = count
        pos_map_[count] = i
        

    return site, site_, pos_map, pos_map_, fix



def map_info_(Lx=9, Ly=10, pepo=None):

    cor = [ 36, 0, 1, 72, 6, 7, 42, 48, 12, 13, 78, 18, 19, 54, 60, 24, 25, 84, 30, 31, 66 ]
    cor += [ 37, 38, 2, 3, 73, 8, 9, 43, 44, 75, 49, 50, 14, 15, 79, 20, 21, 55, 56, 81, 61, 62, 26, 27, 85, 32, 33, 68, 67, 87 ]
    cor += [39, 40,41, 4, 5, 74, 10,11,47,77, 46,45,76,51,52,53,16,17,80,22,23,57,58,59,83,82,63,64,65, 28,29,86,34,35,69,70,71,88,89 ]

    site = {}
    for i in range(Lx):
        for j in range(Ly):
            site[(i,j)] = cor[i*Ly + j]   
    
    
    # # 16, 0, 32, 4, 20, 24, 8, 36, 12, 28
    # site = { (0,0): 16,  (0,1): 0, (0,2): 32, (0,3): 4, (0,4): 20,  (0,5): 24,   (0,6): 8,  (0,7): 36,  (0,8): 12,  (0,9): 28  }
    # # 17, 1, 5, 21, 34, 25, 9, 13, 29, 38
    # site |= { (1,0): 17,  (1,1): 1, (1,2): 5, (1,3): 21, (1,4): 34,  (1,5): 25,   (1,6): 9,  (1,7): 13,  (1,8): 29,  (1,9): 38  }
    # # 18, 2, 33, 6, 22, 26, 10, 37, 14, 30
    # site |= { (2,0):18,  (2,1):2, (2,2):33, (2,3):6, (2,4):22,  (2,5):26,   (2,6):10,  (2,7):37,  (2,8):14,  (2,9):30  }
    # # 19, 3, 7, 23, 35, 27, 11, 15, 31, 39
    # site |= { (3,0):19,  (3,1):3, (3,2):7, (3,3):23, (3,4):35,  (3,5):27,   (3,6):11,  (3,7):15,  (3,8):31,  (3,9):39  }
    
    site_ = {v: k for k, v in site.items()}
    for key, value in site.items():
        x, y = key
        t = pepo[f"I{x},{y}"]
        t.add_tag(f"G{value}")
    
    fix = { f"I{i},{j}":(i,j) for i in range(Lx) for j in range(Ly)}
    
    pos_map = {}
    pos_map_ = {}
    
    for count, i in enumerate(site.values()):
        pos_map[i] = count
        pos_map_[count] = i
        

    return site, site_, pos_map, pos_map_, fix










    
def circ_gates_to_list(circ, pos_map, site_, reverse=False, limit=None):

    gate_l = []
    gate_l_ = []
    gate_label = []
    where_l = []
    where2d = []
    
    if limit:
        gates = circ.gates[:limit]
    else:
        gates = circ.gates
    
    
    for gate in gates:
        
        where = gate.qubits
        
        gate_label.append(gate.label)
        
        if len(where)==2:
            x, y = where
            where_l.append( (pos_map[x], pos_map[y])  ) 
            where2d.append( (site_[x], site_[y])  ) 
            gate_l.append(gate.array.reshape(2,2,2,2))
    
        if len(where)==1:
            x,  = where
            where_l.append( (pos_map[x], )  ) 
            where2d.append( (site_[x], )  ) 
            gate_l.append(gate.array)

    if reverse:
        gate_l.reverse()
        gate_l_.reverse()
        where2d.reverse()
        where_l.reverse()
        gate_label.reverse()

    return  gate_l, where_l, where2d, gate_label






def req_backend(chi=None, threads=None, progbar=False, max_repeats=2**8, alpha="flops"):

    to_backend = "numpy-cpu-double"
    to_backend_ = get_to_backend(to_backend)

    #print("to_backend:=", to_backend)
    # set single thread
    #os.environ['OMP_NUM_THREADS'] = f'{threads}'
    #os.environ['MKL_NUM_THREADS'] = f'{threads}'
    if threads:
        os.environ['OPENBLAS_NUM_THREADS'] = f'{threads}'
        os.environ["NUMBA_NUM_THREADS"] = f'{threads}'

    target_size = 2**34

    #opt_ = get_optimizer_exact(target_size=target_size, 
    #                               parallel=True, 
    #                               progbar=progbar, 
    #                               minimize="flops",
                                   #max_time
    #                               max_repeats=max_repeats,
    #                               )

    opt = opt_contraction_path(alpha =alpha, 
                               max_time="rate:1e8",
                               progbar=progbar,
                               max_repeats=max_repeats)

    copt= None
    if chi:
        copt = ctg.ReusableHyperCompressedOptimizer(
            chi,
            max_repeats=max_repeats,
            minimize='combo-compressed', 
            max_time="rate:1e8",
            progbar=progbar,
            on_trial_error='ignore',
            hash_method = "b",
            overwrite = True,
            parallel=True,
            #directory="cash/",
            )

    res = {"opt":opt, "backend": to_backend, "backend_": to_backend_, "copt":copt}
    return  res



def mps_prep(theta, L):
    to_backend, opt, opt_ = req_backend()
    to_backend_ = get_to_backend(to_backend)
    
    mps = qtn.MPS_computational_state([0]*L)
    
    for t in mps:
        vec = np.array([math.cos(theta), math.sin(theta)]) 
        shape = t.shape
        t.modify(data = vec.reshape(shape))
    
    mps.apply_to_arrays(to_backend_)
    
    return mps



def prod_pepso(x_dim = 3, y_dim = 3, theta=0, dt_=0.2, 
               cycle_peps = True, step_=4, bnd = 4, chi_bmps= 40, dtype="complex64",
              ):
    from floquet.circuits import (trotter_2D_square, circ_gates, Z1_ti, Z2_ti)

    res = req_backend(progbar=True)
    opt = res["opt"]
    x_dim = x_dim
    y_dim = y_dim
    steps = 16
    theta, J, h, hz, dt = 0, 1, 2, 0.0, dt_
    order = 'second'
    obs = {'Z': Z1_ti(x_dim,y_dim), 'Z2': Z2_ti(x_dim,y_dim)}
    circ_info = {"x_dim":x_dim, "y_dim":y_dim, "theta":theta, "J":J, "h":h, "dt":dt, "hz":hz, "order":order}
    qc = trotter_2D_square(x_dim, y_dim, steps, dt, J, h, hz, theta, order = order, obs = obs)
    qc_ = trotter_2D_square(x_dim, y_dim, 1, dt, J, h, hz, 0, order = order)
    gate_l, where_l  = circ_gates(qc_, 1, t_im=0, dt=dt)
    where_l_2d = rotate_to_2d_(where_l[0], x_dim, y_dim)
    pepo_l, bonds = gate_pepo(gate_l[0], where_l_2d, x_dim, y_dim, cycle_peps)
    pepo = pepo_l[0][0]
    pepo.add_tag("pepo")
    
    Lx = x_dim
    Ly = y_dim
    L = Lx * Ly
    inds_b = { f"b{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}
    inds_k = { f"k{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}
    peps = peps_I(Lx, Ly, theta = 0, dtype = dtype)
    if cycle_peps:
        peps = peps_cycle(peps, int(1))
    
    peps.astype_(dtype)
    pepo.astype_(dtype)
    # peps.apply_to_arrays(to_backend_)



    
    
    
    for step in tqdm(range(step_)):
        #peps = peps_normalize(peps, opt, chi=chi_bmps)
        peps_fix = peps & pepo
        peps_fix = peps_fix.flatten(fuse_multibonds=True, inplace=False)
        peps_fix.reindex_(inds_b)
        
        peps = peps_fix.compress_all(inplace=False, **{"max_bond":bnd, "canonize_distance":2, "cutoff":1e-14})
        # print(step, quf.loss_peps(peps, peps_fix, opt, "fid", 1., chi_bmps= 120, mode = "exact"))


    return peps, pepo


def normalize_peps_hyper(peps, chi= 40, to_backend=None, progbar=True):
    
    res = req_backend(progbar=True, chi=chi)
    copt = res["copt"]
    
    
    tn = peps.H & peps
    #overlap = tn.contract(all, optimize=opt)

    if to_backend:
        tn.apply_to_arrays(to_backend)
    
    overlap, (flops, peak) = apply_hyperoptimized_compressed(tn, copt, chi, cutoff=0, progbar=progbar)
    main, exp=(overlap.contract(), overlap.exponent)
    norm = complex(main * 10**(exp))
    print("chi", chi, "norm", norm)
    peps  = peps * (norm**-0.5)

    
    return peps

def pro_projs(peps, pepo, bnd=3):
    psi0 = peps & pepo
    psi0 = psi0.copy()
    bp = L2BP(psi0)
    bp.run(progbar=True)
    mantissa, norm_exponent = bp.contract(strip_exponent=True)
    site_tags = bp.site_tags
    psi0 = peps & pepo
    psi0 = psi0.copy()
    
    #psi0.reindex_(inds_b)
    tn = bp.compress(psi0, max_bond=bnd, cutoff=1.e-12, lazy=True)
    #fix = {f"I{i},{j}":(i,j) for i,j in itertools.product(range(Lx), range(Ly)) }
    #tn.draw(psi0.site_tags, fix=fix, legend = False, show_tags=False)
    #fix = {f"I{i},{j}":(i,j) for i,j in itertools.product(range(Lx), range(Ly)) }
    # tn.draw(list(psi0.site_tags)+["proj"], legend = False, node_outline_darkness=0.2, node_shape='s',    node_color=None,
    #     node_scale=1.0,show_tags=False, 
    #     node_size=1.2,)
    
    tn_proj = tn.select(["proj"], "any")
    tn_constant = tn.select(["proj"], "!any")
    tn_proj.drop_tags("proj")
    tn_constant.drop_tags("pepo")
    
    return tn_proj, tn_constant





def loss_peps_hyper(tn_proj, tn_constant, psi0, chi, site_tags, copt, opt):
    
    peps = (tn_constant | tn_proj)
    for st in site_tags:
            peps.contract_tags_(st)


    def apply_hyperoptimized_compressed(tn, copt, chi, output_inds=None, tree_gauge_distance=2, progbar=False, 
                                        cutoff=1.e-12, equalize_norms=1.0):
        
        tn.full_simplify_(seq='R', split_method='svd', inplace=True)
        
        tree = tn.contraction_tree(copt)
        
        tn_ = tn.copy()
        
        flops = tree.contraction_cost(log=10)
        peak = tree.peak_size(log=2)
        
        tn_.contract_compressed_(
            optimize=tree,
            output_inds=output_inds,
            max_bond=chi,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
        )
        return tn_, (flops, peak)


    tn = peps.H & peps
    #overlap = tn.contract(all, optimize=opt)

    overlap, (flops, peak) = apply_hyperoptimized_compressed(tn, copt, chi, cutoff=0)
    main, exp=(overlap.contract(all), overlap.exponent)
    overlap = main * 10**(exp)
    
    tn = peps.H & psi0
    #overlap_ = tn.contract(all, optimize=opt)

    overlap_, (flops, peak) = apply_hyperoptimized_compressed(tn, copt, chi, cutoff=0)
    main, exp = (overlap_.contract(), overlap_.exponent)
    overlap_ = main * 10**(exp)


    return 1 - abs(overlap_**2)/abs(overlap)









# build-up a projector into the excited subspace of the bp massages:   I - |tmi><tmj|  
def projector(tmi, tmj):
    to_backend_ = ar.infer_backend(tmi)
    to_backend = get_to_backend(to_backend_)
    dtype = ar.get_dtype_name(tmi.data)
    backend = ar.infer_backend(tmi.data)
    
    tmi_ = tmi.data*1.
    tmj_ = tmj.data*1.
    
    #store original shape
    shape_ =  ar.do("shape", tmi_)
    
    # P = |mi> <mj|
    mi_vector = ar.do("reshape", tmi_, (-1,))
    mj_vector = ar.do("reshape", tmj_, (-1,))
    Pr_ij = ar.do("outer", mi_vector, mj_vector)
    
    # Create the identity matrix of the same shape as the projector
    I = ar.do("eye", Pr_ij.shape[0])  
    I = ar.do('asarray', I, like=backend)
    
    
    # Subtract the projector from the identity matrix: I - |mi> <mj|
    Pr_excited = I - Pr_ij

    #back to original shape
    Pr_excited = ar.do("reshape", Pr_excited, shape_+shape_)

    #print(Pr_excited @ mi_vector, "==0", mj_vector @ Pr_excited, "==0")
    
    Pr_ij = ar.do("reshape", Pr_ij, shape_+shape_)

    return Pr_excited

def mps_prepare(bnd = 16, b = [8] * 7, theta = - 2*math.pi/48):
    
    to_backend, opt, opt_ = req_backend(chi=None, threads=None)
    # block structure of MPS
    block_l = b

    # total system size or number of qubits
    L = sum(block_l) 

    #MPS length
    L_mps = len(block_l)

    # irregular bnds: [max_bond] * (L_mps-1)
    bnds = [bnd] * (L_mps-1)

    basis = "+"
    list_basis = [basis] * L
    #MPS:
    psi, inds_fuse= Init_bmps(block_l, L_mps, list_basis, 'auto-hq', theta = theta)
    #psi = quf.iregular_bnd(psi, bnds, rand_strength=0.0)

    to_backend_ = get_to_backend(to_backend)
    psi.apply_to_arrays(to_backend_)


    return psi, block_l, inds_fuse, bnds 


def rotate_to_2d(where, lx, ly, lz):
    dic_ = {}
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                dic_ |= { f"{i*ly*lz + j*lz + k}":(i,j)} 
    where_2d = []
    for where_ in where:
        if len(where_)==1:
            x, = where_
            where_2d.append((dic_[f"{x}"],))
        if len(where_)==2:
            x, y= where_
            where_2d.append((dic_[f"{x}"],dic_[f"{y}"]))
    return where_2d


def rotate_to_2d_(where, lx, ly):
    dic_ = {}
    for i in range(lx):
        for j in range(ly):
                dic_ |= { f"{j*lx + i}":(i,j)} 
    where_2d = []
    for where_ in where:
        if len(where_)==1:
            x, = where_
            where_2d.append((dic_[f"{x}"],))
        if len(where_)==2:
            x, y= where_
            where_2d.append((dic_[f"{x}"],dic_[f"{y}"]))
    return where_2d



def obe_measure(rho, chi_, output_inds, to_backend_, obs_ = qu.pauli("Z") & qu.pauli("Z"), progbar=False, f_max = 15 , peak_max=34 ):
    
    end_inds_k = [ i for i in output_inds if i.startswith("k")]
    
    to_backend, opt_, opt = req_backend()
    to_backend_ = get_to_backend(to_backend) #"numpy-single"


    if chi_:
        _, _, _, copt = req_backend(chi_, progbar=progbar, max_repeats=2**9)
        rho_appro, (f, peak_) = apply_hyperoptimized_compressed(rho, copt, chi_, output_inds=output_inds,
                                            tree_gauge_distance=4, 
                                            progbar=progbar, f_max = f_max , peak_max=peak_max, 
                                            cutoff=1.e-12
                                         )
    else:
        rho_appro = rho.contract(all, optimize=opt)
        rho_appro = qtn.TensorNetwork([rho_appro])

    if len(output_inds)==4:
        for t in rho_appro:
            t.transpose_(*output_inds)
            rho_d = t.data
            rho_d = ar.do("reshape", rho_d, (4,4))
            rho_d = (rho_d + ar.dag(rho_d))*0.5
            rho_d = ar.do("reshape", rho_d, (2,2,2,2))
            t.transpose_(*output_inds)
            t.modify(data=rho_d)


    if len(output_inds)==2:
        for t in rho_appro:
            t.transpose_(*output_inds)
            rho_d = t.data
            rho_d = ar.do("reshape", rho_d, (2,2))
            rho_d = (rho_d + ar.dag(rho_d))*0.5
            rho_d = ar.do("reshape", rho_d, (2,2))
            t.transpose_(*output_inds)
            t.modify(data=rho_d)

    obs_ = to_backend_(obs_)
    dic_end = { i:i.replace("k", "b")  for i in end_inds_k}
    rho_ = rho_appro.reindex(dic_end, inplace=False)
    
    rhoz=qtn.tensor_network_gate_inds(rho_appro, obs_, end_inds_k, contract=True,  inplace=False)
    rhoz.reindex(dic_end, inplace=True)
    norm = rho_.contract(all, optimize=opt)
    z = rhoz.contract(all, optimize=opt)
    z = complex(z/norm)

    return z, (f, peak_)
    


def obe_measure_(rho, chi_, output_inds, to_backend_, obs_ = qu.pauli("Z") & qu.pauli("Z"), progbar=False ):


    end_inds_k = [ i for i in output_inds if i.startswith("k")]
    to_backend, opt_, opt, copt = req_backend(chi_)
    obs_ = to_backend_(obs_)
    rhoz=qtn.tensor_network_gate_inds(rho, obs_, end_inds_k, contract=True,  inplace=False)
    
    dic_end = { i:i.replace("k", "b")  for i in end_inds_k}
    rhoz = rhoz.reindex(dic_end, inplace=False)
    rho = rho.reindex(dic_end, inplace=False)


    if chi_:
        _, _, _, copt = req_backend(chi_, progbar=progbar)
        rhoz.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)
        res_Z, (f, peak_) = apply_hyperoptimized_compressed(rhoz, copt, chi_, output_inds={},
                                            tree_gauge_distance=4, 
                                            progbar=progbar, 
                                            cutoff=1.e-12
                                         )
        main, exp=(res_Z.contract(), res_Z.exponent)
        z = (main) * 10**(exp)

        _, _, _, copt = req_backend(chi_, progbar=progbar)
        rho.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)
        norm, (f, peak_) = apply_hyperoptimized_compressed(rho, copt, chi_, output_inds={},
                                            tree_gauge_distance=4, 
                                            progbar=progbar, 
                                            cutoff=1.e-12
                                         )
        main, exp=(norm.contract(), norm.exponent)
        norm = (main) * 10**(exp)

    else:
        z = rhoz.contract(all, optimize=opt)
        norm = rho.contract(all, optimize=opt)

    
    z = complex(z/norm)

    return z


    


def prepare_gates( depth_=4, 
                  lx=6, ly=6, lz=1, opt=None, depth_r=None, 
                  cycle_gates = True,
                  cycle_peps = True, 
                  swap_gate = False,
                  label = "", 
                  exact_cal = False,
                  lightcone = None,
                  theta = - 2*math.pi/48,
                  h_ = -1,
                  J_ = -1,
                  hz = -0.5,
                  trotter = 1,
                  pauli = "Z",
                  site_ = 14,
                  delta_t = 0.3,
                  ):

    L = lx * ly * lz
    #label += f"-{depth_r}"
    to_backend, opt, opt_ = req_backend()

    delta_t =  delta_t    #math.pi/4 #1.01 #1.01 #0.39   #math.pi/16
    J_ =  J_
    h_ =  h_
    to_backend_ = get_to_backend(to_backend)



    peps = peps_I(lx, ly, theta = theta)
    if cycle_peps:
        peps = peps_cycle(peps, int(1))

    peps.apply_to_arrays(to_backend_)

        
    gate_l, where_l, gate_round, circ= trotter_gates(dt=delta_t, J=J_,
                                                    h=h_, hz=hz, Lx=lx, Ly=ly, Lz=lz, 
                                                    cycle = cycle_gates,
                                                    swap_gate = swap_gate, 
                                                    depth_= depth_, 
                                                    list_basis = ["0"]*L, 
                                                   lightcone = lightcone, #site, 
                                                   theta = theta,
                                                   trotter=trotter,
                                                   tags="ROUND_0",
                                                     )


    #x_mpo = None
    #x_pepo = pepo_init(Lx, Ly, to_2d(site, Lx, Ly, Lz), O_label,to_backend_, cycle=cycle_peps)
    # if lightcone:
    #     x_mpo, x_pepo = pepoXmpo_prep(lx, ly, to_backend, pauli, lightcone, cycle_peps = cycle_peps)
    # else:
    
    print("site_mpo/pepo", site_, site_+1)
    x_mpo, x_pepo = pepoXmpo_prep(lx, ly, to_backend, pauli, [site_], cycle_peps = cycle_peps)
    print("gate_round", gate_round)



    gate_l, where_l = gate_chunk(gate_l, where_l, gate_round)   
    
    where_l, gate_l= abs_sqg_fromlist(where_l[0], gate_l[0], iterations=4*L)
    #where_l = where_l[0] 
    #gate_l = gate_l[0]
    
    where_l_2d = rotate_to_2d(where_l,lx,ly,lz)
    pepo_l, bonds = gate_pepo(gate_l, where_l_2d, lx, ly, 
                        cycle_peps, depth_r = depth_r,
                        )
    pepo_l = l_to_backend(pepo_l, to_backend_) 
    pepo_l = pepo_l * depth_
    gate_round = [len(where_l)] *  depth_
    
    where_l = [where_l] *  depth_
    gate_l = [gate_l] *  depth_
    
    print("gate_round_f", gate_round)
    print("bonds", bonds)

    gate_l = gate_to_backend(gate_l, to_backend_)

    if exact_cal:
        site = 0
        z_cal = circ.local_expectation(qu.pauli("Z"), site, 
                                    optimize=opt, simplify_atol=1e-14)   
        print("<Z>",  site, z_cal)

    return circ, gate_l, where_l, (peps, x_mpo, pepo_l, x_pepo, label) 


def gate_trans(gate_l):
    gate_l_ = []
    for G in gate_l:
        if len(G.shape)==2:
            G = np.transpose(G, (1,0))
        if len(G.shape)==4:
            G = np.transpose(G, (2,3,0,1))
        gate_l_.append(G)

    
    return gate_l_


def abs_sqg_fromlist(where_l_, gate_l_, iterations=20):
    where_l_ = where_l_.copy()
    gate_l_ = gate_l_.copy()

    for _ in range(iterations):
        #print_(where_l_, gate_l_)
        where_l_, gate_l_ = reduce_(where_l_, gate_l_)

    where_l_.reverse()
    gate_l_.reverse()
    gate_l_ = gate_trans(gate_l_)
    for _ in range(iterations):
        #print_(where_l_, gate_l_)
        where_l_, gate_l_ = reduce_(where_l_, gate_l_)

    where_l_.reverse()
    gate_l_.reverse()
    gate_l_ = gate_trans(gate_l_)
    return where_l_, gate_l_



def find(where_l, cor, count):
    if len(cor) == 1:
        x, = cor
        for count_, where in enumerate(where_l):
            if count_>count:
                if x in where:
                    return count_ 
        return None
    elif len(cor) == 2:
        x, y = cor
        for count_, where in enumerate(where_l):
            if count_>count:
                if len(where) == 2:
                    x_, y_ = where
                    if (x_ in (x,y)) and (y_ in (x,y)):
                        return count_ 
                    elif x_ in (x,y) or y_ in (x,y):
                        return None   
                elif len(where) == 1:
                    x_, = where
                    if x_ in (x,y):
                        return count_
    
        return None


def run_bp(tn, site_tags, normalize=True, progbar=False, max_iterations=520, tol=1.e-7,damping=0.01, update='sequential',):
    
    tn0 = tn.copy()
    res = req_backend(progbar=False)
    opt = res["opt"]
    
    bp = L1BP(tn, optimize=opt, site_tags=site_tags, damping=damping)
    bp.update='sequential'
    bp.run(tol=tol, max_iterations=20, progbar=progbar)
    bp.update=update
    bp.run(tol=tol, max_iterations=max_iterations, progbar=progbar)

    mantissa, norm_exponent = bp.contract(strip_exponent=True)
    
    est_norm = complex(mantissa * 10**norm_exponent)
    
    bp.normalize_messages()
    bp.cal_projects()
    if normalize:
        normalize_bp(tn, bp, [[]], site_tags)

    res = {"tn0":tn0, "bp":bp, "norm": est_norm, "tn":tn}
    return res

def env_exact(tn, reg_reindex, reg_tags, inds_rho):
    tn_=tn.copy()
    tn_bra = tn_.select(["BRA"], which="any")
    tn_bra.reindex_(reg_reindex)
    
    tags_ = list(reg_tags.keys())
    tn_left = tn_.select([tags_[0], "KET"], which="all")
    tn_right = tn_.select([tags_[1], "KET"], which="all")
    indx_ = [ i for i in tn_left.outer_inds() if i in tn_right.outer_inds()  ]
    
    map_ket_l = {indx_[0]:"up_l"}
    map_ket_r = {indx_[0]:"up_r"}
    
    
    tn_left.reindex_(map_ket_l)
    tn_right.reindex_(map_ket_r)
    
    tags_ = list(reg_tags.keys())
    
    
    tn_left = tn_.select([tags_[0], "BRA"], which="all")
    tn_right = tn_.select([tags_[1], "BRA"], which="all")
    indx_bra = [ i for i in tn_left.outer_inds() if i in tn_right.outer_inds()  ]
    map_bra_l = {indx_bra[0]:"low_l"}
    map_bra_r = {indx_bra[0]:"low_r"}
    
    tn_left.reindex_(map_bra_l)
    tn_right.reindex_(map_bra_r)
    inds_l = [inds_rho[0], inds_rho[1]]+["up_l","up_r"]+[inds_rho[2], inds_rho[3]]+["low_l","low_r"]
    N_exact = tn_.contract(all, output_inds=inds_l)
    
    return N_exact


# Split combs_2 into n approximately equal parts
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    print("cpus", n, "chunk", chunk_size)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# @ray.remote
# def filter_combin_remote(sublist, sites):
#     import quf  # Import inside the function
#     return quf.filter_combin(sublist, sites)


@ray.remote
def filter_combin_remote(sublist, sites):
    import quf
    try:
        return quf.filter_combin(sublist, sites)
    except Exception as e:
        print(f"Error in filter_combin_remote: {e}")
        raise



def filter_combin(combs_2, sites):
    comb_two_f = []
    for combs_2_ in tqdm(combs_2):
        
        flattened = [element for edge in combs_2_ for element in edge]
        # Check if each element appears exactly two times
        counts = Counter(flattened)
        values_list = list(counts.values())
        
        if 1 in values_list:
    
            # if sites exist, then we are intersted in the density matrix: dangling site alone does not result in zero;
            if sites:
                # if dangling sites are siting on the density matrix positions, then they are valid; as loops can be non-zero
                sublist = [x for x in flattened if counts[x] == 1]
                # all dangling sites should be sitting on a set or a subset of the density matrix positions
                is_sublist = all(elem in sites for elem in sublist)
                
                if is_sublist:
                    comb_two_f.append(combs_2_)    
        else: 
            comb_two_f.append(combs_2_)
    
    comb_two_f = [  tuple(sorted(a)) for a in comb_two_f ]
    return comb_two_f




def combine_elements(edges, length,   sites = []):
    
    combs_2 = list(itertools.combinations(edges, length))
    #filter non-loops edges:
    comb_two_f = filter_combin(combs_2, sites)
    # Flatten the loop
    if length == 0:
        comb_two_f = [[]]
    
    return comb_two_f
    

def combine_elements_ray(edges, length,   sites = [], num_cpus = 10):
    import ray
    import math




    def filter_combin(combs_2, sites):
        comb_two_f = []
        for combs_2_ in tqdm(combs_2):
            
            flattened = [element for edge in combs_2_ for element in edge]
            # Check if each element appears exactly two times
            counts = Counter(flattened)
            values_list = list(counts.values())
            
            if 1 in values_list:
        
                # if sites exist, then we are intersted in the density matrix: dangling site alone does not result in zero;
                if sites:
                    # if dangling sites are siting on the density matrix positions, then they are valid; as loops can be non-zero
                    sublist = [x for x in flattened if counts[x] == 1]
                    # all dangling sites should be sitting on a set or a subset of the density matrix positions
                    is_sublist = all(elem in sites for elem in sublist)
                    
                    if is_sublist:
                        comb_two_f.append(combs_2_)    
            else: 
                comb_two_f.append(combs_2_)
        
        comb_two_f = [  tuple(sorted(a)) for a in comb_two_f ]
        return comb_two_f
    
    
    @ray.remote
    def filter_combin_remote(sublist, sites):
        from collections import Counter
        import tqdm as tqdm
        try:
            return filter_combin(sublist, sites)
        except Exception as e:
            print(f"Error in filter_combin_remote: {e}")
            raise
    

    combs_2 = list(itertools.combinations(edges, length))
    print("start: loop", )

    # Initialize Ray
    ray.init(num_cpus=num_cpus)

    
    # Split the list
    combs_2_sub = split_list(combs_2, int(num_cpus))
    print(len(combs_2_sub), len(combs_2)/len(combs_2_sub))
    
    # Distribute the sublists across CPUs using Ray
    future_results = [filter_combin_remote.remote(sublist, sites=sites) for sublist in combs_2_sub]
    
    # Gather the results
    filtered_sublists = ray.get(future_results)
    
    # Combine the results into a single list
    filtered_combs = [item for sublist in filtered_sublists for item in sublist]
    
    # Shutdown Ray when done
    ray.shutdown()
    
    #print("Filtered combinations:", filtered_combs)

    return filtered_combs




def gen_loop_pair_(edges, region_tag=[], site_tags=[], length = 4, sites=None):

    res = {f"loop_{0}":[[]], f"length_{0}":1}
    all_combinations = []
    for r in range(1, length + 1):  # r is the size of each combination
        
        two_combinations = list(itertools.combinations(edges, r))
        
        # Flatten the loop
        two_combinations_f = []
        for two_combinations_ in two_combinations:
            
            flattened = [element for edge in two_combinations_ for element in edge]
            # Check if each element appears exactly two times
            counts = Counter(flattened)
            
            valid_loop = True
            if 1 in counts.values():

                # if sites exist, then we are intersted in the density matrix: dangling site alone does not result in zero;
                if sites:
                    # if dangling sites are siting on the density matrix positions, then they are valid; as loops can be non-zero
                    sublist = [x for x in flattened if counts[x] == 1]
                    # all dangling sites should be sitting on a set or a subset of the density matrix positions
                    is_sublist = all(elem in sites for elem in sublist)
                    
                    if is_sublist:
                        valid_loop = True
                    else:
                        valid_loop = False
                else:
                    valid_loop = False
            
            if valid_loop:
                two_combinations_f.append(two_combinations_)

        two_combinations_f = [  tuple(sorted(a)) for a in two_combinations_f]
        
        res |= {f"length_all_{r}":len(two_combinations)}
        
        if region_tag:
            two_combinations_f_ = []
            for loop in two_combinations_f:
                loop_flat = [item for tup in loop for item in tup]
                if has_common_elements(loop_flat, region_tag):
                     two_combinations_f_.append(loop)
            two_combinations_f = two_combinations_f_

            
        res |= {f"length_{r}":len(two_combinations_f), f"length_all_{r}":len(two_combinations)}
        res |= {f"loop_{r}": two_combinations_f}


        all_combinations.append(two_combinations_f)
    
            
    loop_l = []
    for i in range(len(all_combinations)):
            loop_l += all_combinations[i]    

            
    # loop_f = []
    # for loop in loop_l:
    #     loop = [  tuple(sorted(a)) for a in loop]
    #     loop_f.append(loop)

    # if region_tag:
    #     loop_f_ = []
    #     for loop in loop_f:
    #         loop_flat = [item for tup in loop for item in tup]
    #         if has_common_elements(loop_flat, region_tag):
    #              loop_f_.append(loop)

    #     loop_f = loop_f_


    
    loop_l = elm_red_uniq(loop_l)
    
    loop_l = [[]] + loop_l
    
    res |= {"loops":loop_l}
    return res




def gen_loop_pair(tn_flat, region_tag=[], str_order="1-4-6-7-8-go-10", site_tags=[]):
    
    
    site_tags = list(site_tags)
    # make up all loops, you want
    length_info = {}
    Lx = tn_flat.Lx
    Ly = tn_flat.Ly
    
    #start with vaccume
    loop_pair= []
    
    if "0" in str_order:
        loop_pair += [[]]
        length_info |= {"0":1}

    # if "1" in str_order:
    #     loop_pair_ = produce_line(Lx, Ly)
    #     loop_pair += loop_pair_
    #     length_info |= {"1":len(loop_pair)}
    
    if "4" in str_order:
        res = gen_loop(tn_flat, loop_length=4, intersect=True, region_tag=region_tag, site_tags=site_tags)
        loop_pair += res["loop_pair"]
        length_info |= {"4":len(loop_pair)}
    
    if "6" in str_order:
    
        res = gen_loop(tn_flat, loop_length=6, intersect=True, region_tag=region_tag, site_tags=site_tags)
        loop_pair += res["loop_pair"]
        length_info |= {"6":len(loop_pair)}

    if "7" in str_order:
        res = gen_loop_(tn_flat, loop_length=7, region_tag=region_tag, site_tags=site_tags)
        loop_pair += res["loop_pair"]
        length_info |= {"7":len(loop_pair)}
    

    if "8" in str_order:

        res = gen_loop(tn_flat, loop_length=8, intersect=True, region_tag=region_tag, site_tags=site_tags)
        loop_pair += res["loop_pair"]
        length_info |= {"8":len(loop_pair)}
    
    if "10" in str_order:
    
        res = gen_loop_(tn_flat, loop_length=10, region_tag=region_tag, site_tags=site_tags)
        loop_pair += res["loop_pair"]
        length_info |= {"10":len(loop_pair)}


    if "co" in str_order:
    
        loop_pair_ =produce_correlatedloops(Lx, Ly, region_tag=region_tag)
        loop_pair += loop_pair_
        length_info |= {"co":len(loop_pair)}
    
    if "cc" in str_order:
    
        loop_pair_ =produce_correlatedloops_(Lx, Ly, region_tag=region_tag)
        loop_pair += loop_pair_
        length_info |= {"cc":len(loop_pair)}
    
    
    if "go" in str_order:

        # handy build-up loops
        loop_pair_ = produce_gloops(Lx, Ly, region_tag=region_tag)
        loop_pair += loop_pair_
        length_info |= {"go":len(loop_pair)}
    
    if "gc" in str_order:

        # handy build-up loops
        loop_pair_ = produce_gloops_(Lx, Ly, region_tag=region_tag)
        loop_pair += loop_pair_
        length_info |= {"gc":len(loop_pair)}

    
    if "44" in str_order:
    
        loop_pair_ = multi_loop(tn_flat, lenght_gs=4, lenght_exc=4)
        loop_pair += loop_pair_
        length_info |= {"4-4":len(loop_pair)}
    
    if "46" in str_order:
    
        loop_pair_ = multi_loop(tn_flat, lenght_gs=4, lenght_exc=6)
        loop_pair += loop_pair_
        length_info |= {"4-6":len(loop_pair)}
    
    if "74" in str_order:
        
        loop_pair_ = multi_loop_(tn_flat, lenght_gs=7, lenght_exc=4)
        loop_pair += loop_pair_
        length_info |= {"7-4":len(loop_pair)}
    
    if "66" in str_order:
        loop_pair_ = multi_loop(tn_flat, lenght_gs=6, lenght_exc=6)
        loop_pair += loop_pair_
        length_info |= {"6-6":len(loop_pair)}
    
    loop_pair = elm_red_uniq(loop_pair)

    return loop_pair, length_info


def gen_loop(tn, loop_length=4, intersect=True, region_tag=[], site_tags=[]):
    gen = qu.tensor.networking.gen_paths_loops(tn, max_loop_length=loop_length, intersect=intersect, tids=None, inds=None, paths=None)
    loop_pair = []
    tags_loops = []
    inds_loop = []
    for g in gen:
        
        if len(g.inds) == loop_length:
            tags = [list(tn.tensor_map[tid].tags) for tid in g.tids]
            tags = [item for sublist in tags for item in sublist]
            inds = list(g.inds) 
            tids_pair = [    list(tn.ind_map[indx])   for indx in inds]
            pair_tags = []
            for t_pair in tids_pair:
                tid1, tid2 = t_pair
                tags1 = tn.tensor_map[tid1].tags            
                tags2 = tn.tensor_map[tid2].tags            
                pattern = re.compile(rf"I-?\d+(\.\d+)?(,-?\d+(\.\d+)?)*")
        
                
                # Filter the list
                tags1 = [elem for elem in tags1 if pattern.fullmatch(elem)]
                # Filter the list
                tags2 = [elem for elem in tags2 if pattern.fullmatch(elem)]

                if site_tags:
                    tags1 = [tag for tag in tags1 if tag in site_tags]
                    tags2 = [tag for tag in tags2 if tag in site_tags]
                    
                
                pair_tags.append(  tuple(sorted((tags1[0], tags2[0])))     )

            #set(region_tag).issubset(set(tags))
            if region_tag and has_common_elements(region_tag, tags):
                
                inds_loop.append(inds)
                loop_pair.append(pair_tags)
                pair_tags_flat = list(itertools.chain.from_iterable(pair_tags))
                tags = [list(tn.tensor_map[tid].tags) for tid in g.tids]
                tags = [item for sublist in tags for item in sublist]
                tags = [tag for tag in tags if tag.startswith("I")]
                tags_loops.append(tags)
                
            if not region_tag:
                inds_loop.append(inds)
                loop_pair.append(pair_tags)
                pair_tags_flat = list(itertools.chain.from_iterable(pair_tags))
                tags = [list(tn.tensor_map[tid].tags) for tid in g.tids]
                tags = [item for sublist in tags for item in sublist]
                tags = [tag for tag in tags if tag.startswith("I")]
                tags_loops.append(tags)
            
        
    res = {"loop_pair":loop_pair, "tags_loops":tags_loops, "inds_loop":inds_loop}
    return res


def multi_loop(tn_flat, lenght_gs=4, lenght_exc=4):
    res = gen_loop(tn_flat, loop_length=lenght_gs, intersect=True)

    loop_pair = res["loop_pair"]
    
    # final unique loops
    loop_pair_f = []

    for corner_loop in loop_pair:
        loop_pair_ = produce_disjoint_loops(tn_flat, loop_length=lenght_exc, index=0, corner_loop=corner_loop)

        loop_pair_f.append(loop_pair_)
    

    # flatten list
    loop_pair_f = [item for sublist in loop_pair_f for item in sublist]
    
    
    
    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_pair_f]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_pair_f = [list(tup) for tup in unique_sublists]
    return loop_pair_f

def multi_loop_(tn_flat, lenght_gs=4, lenght_exc=4):
    res = gen_loop_(tn_flat, loop_length=lenght_gs)

    loop_pair = res["loop_pair"]
    
    # final unique loops
    loop_pair_f = []

    for corner_loop in loop_pair:
        loop_pair_ = produce_disjoint_loops(tn_flat, loop_length=lenght_exc, index=0, corner_loop=corner_loop)

        loop_pair_f.append(loop_pair_)
    

    # flatten list
    loop_pair_f = [item for sublist in loop_pair_f for item in sublist]
    
    
    
    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_pair_f]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_pair_f = [list(tup) for tup in unique_sublists]
    return loop_pair_f


def produce_line(Lx, Ly):
    
    #loop_8v = [[]]
    loop_8v = [[('I0,0', 'I1,0')],[('I0,0', 'I0,1')]]
    loop_8v += [[('I0,0', 'I1,0'),('I1,0', 'I1,1')],[('I0,0', 'I0,1'),('I0,1', 'I1,1')]]
    loop_8v += [[('I1,0', 'I2,0'),('I1,0', 'I1,1')],[('I1,0', 'I1,1'),('I1,1', 'I0,1')]]
    loop_8v += [[('I0,0', 'I1,0'),('I1,0', 'I2,0')],[('I0,0', 'I0,1'),('I0,1', 'I0,2')]]

    loop_8v += [[('I0,0', 'I1,0'),('I1,0', 'I2,0'),('I2,0', 'I3,0')],[('I0,0', 'I0,1'),('I0,1', 'I0,2'),('I0,2', 'I0,3')]]
    loop_8v += [[('I0,0', 'I1,0'),('I1,0', 'I1,1'),('I1,1', 'I0,1')],[('I0,0', 'I1,0'),('I1,0', 'I1,1'),('I1,1', 'I2,1')]]
    loop_8v += [[('I0,0', 'I1,0'),('I1,0', 'I1,1'),('I1,1', 'I1,2')],[('I1,0', 'I2,0'),('I1,0', 'I1,1'),('I1,1', 'I0,1')]]
    loop_8v += [[('I1,0', 'I2,0'),('I1,0', 'I1,1'),('I1,1', 'I2,1')],[('I1,0', 'I2,0'),('I1,0', 'I1,1'),('I1,1', 'I1,2')]]
    loop_8v += [[('I0,0', 'I0,1'),('I0,1', 'I0,2'),('I0,2', 'I1,2')],[('I1,0', 'I1,1'),('I1,1', 'I1,2'),('I1,2', 'I0,2')]]
    loop_8v += [[('I0,0', 'I0,1'),('I0,1', 'I1,1'),('I1,1', 'I1,0')],[('I0,0', 'I0,1'),('I0,1', 'I1,1'),('I1,1', 'I1,2')]]
    loop_8v += [[('I0,0', 'I0,1'),('I0,1', 'I0,2'),('I0,1', 'I1,1')],[('I1,0', 'I1,1'),('I1,1', 'I1,2'),('I1,1', 'I0,1')]]

    # Parameters
    loop_f = []
    for loop in loop_8v:
        for dx, dy in itertools.product(range(Lx), range(Ly)):
            res = translate_loop(loop, dx, dy, Lx, Ly)
            if res:
                loop_f.append(res)
    
    loop_f_ = []
    for loop in loop_f:
        loop = [  tuple(sorted(a)) for a in loop]
        loop_f_.append(loop)



    
    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_f_]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_f_ = [list(tup) for tup in unique_sublists]
    return loop_f_





def produce_correlatedloops(Lx, Ly, region_tag=[],):
    
    loop_8v = [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I2,1'),('I2,1', 'I2,2')    ]             ]
    
    loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I1,2'),('I1,2', 'I2,2')    ]             ]

    # loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I1,2'),('I1,2', 'I1,3'),('I1,3', 'I2,3')    ]             ]


    # loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,0', 'I2,0'),('I2,0', 'I2,1'),('I2,1', 'I2,2')    ]             ]


    loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I2,1'),('I2,1', 'I2,2'),('I1,1', 'I1,2'),('I1,2', 'I2,2')    ]             ]



    loop_8v += [[('I2,0', 'I3,0'), ('I2,0', 'I2,1'), ('I2,1', 'I3,1'), ('I3,0', 'I3,1'),('I0,2', 'I0,3'), ('I0,2', 'I1,2'), ('I1,2', 'I1,3'), ('I0,3', 'I1,3'),('I1,1', 'I2,1'),('I2,1', 'I2,2'),('I1,1', 'I1,2'),('I1,2', 'I2,2')    ]             ]



    loop_8v += [[('I2,0', 'I3,0'), ('I2,0', 'I2,1'), ('I2,1', 'I3,1'), ('I3,0', 'I3,1'),('I0,2', 'I0,3'), ('I0,2', 'I1,2'), ('I1,2', 'I1,3'), ('I0,3', 'I1,3'),('I2,1', 'I2,2'),('I1,2', 'I2,2')    ]             ]



    loop_8v += [[('I2,0', 'I3,0'), ('I2,0', 'I2,1'), ('I2,1', 'I3,1'), ('I3,0', 'I3,1'),('I0,2', 'I0,3'), ('I0,2', 'I1,2'), ('I1,2', 'I1,3'), ('I0,3', 'I1,3'),('I1,1', 'I2,1'),('I1,1', 'I1,2')   ]             ]



    # Parameters
    loop_f = []
    for loop in loop_8v:
        for dx, dy in itertools.product(range(Lx), range(Ly)):
            res = translate_loop(loop, dx, dy, Lx, Ly)
            if res:
                loop_f.append(res)
    
    
    
    loop_f_ = []
    for loop in loop_f:
        loop = [  tuple(sorted(a)) for a in loop]
        loop_f_.append(loop)

    if region_tag:
        loop_f = []
        for loop in loop_f_:
            loop_flat = [item for tup in loop for item in tup]
            if has_common_elements(loop_flat, region_tag):
                 loop_f.append(loop)

    loop_f_ = loop_f


    
    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_f_]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_f_ = [list(tup) for tup in unique_sublists]
    return loop_f_

def produce_correlatedloops_(Lx, Ly,region_tag=[],):
    
    loop_8v = [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I2,1'),('I2,1', 'I2,2')    ]             ]
    
    loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I1,2'),('I1,2', 'I2,2')    ]             ]

    # loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I1,2'),('I1,2', 'I1,3'),('I1,3', 'I2,3')    ]             ]


    # loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,0', 'I2,0'),('I2,0', 'I2,1'),('I2,1', 'I2,2')    ]             ]


    loop_8v += [[('I0,0', 'I0,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I0,1', 'I1,1'),('I2,2', 'I3,2'), ('I2,2', 'I2,3'), ('I2,3', 'I3,3'), ('I3,2', 'I3,3'),('I1,1', 'I2,1'),('I2,1', 'I2,2'),('I1,1', 'I1,2'),('I1,2', 'I2,2')    ]             ]



    loop_8v += [[('I2,0', 'I3,0'), ('I2,0', 'I2,1'), ('I2,1', 'I3,1'), ('I3,0', 'I3,1'),('I0,2', 'I0,3'), ('I0,2', 'I1,2'), ('I1,2', 'I1,3'), ('I0,3', 'I1,3'),('I1,1', 'I2,1'),('I2,1', 'I2,2'),('I1,1', 'I1,2'),('I1,2', 'I2,2')    ]             ]



    loop_8v += [[('I2,0', 'I3,0'), ('I2,0', 'I2,1'), ('I2,1', 'I3,1'), ('I3,0', 'I3,1'),('I0,2', 'I0,3'), ('I0,2', 'I1,2'), ('I1,2', 'I1,3'), ('I0,3', 'I1,3'),('I2,1', 'I2,2'),('I1,2', 'I2,2')    ]             ]



    loop_8v += [[('I2,0', 'I3,0'), ('I2,0', 'I2,1'), ('I2,1', 'I3,1'), ('I3,0', 'I3,1'),('I0,2', 'I0,3'), ('I0,2', 'I1,2'), ('I1,2', 'I1,3'), ('I0,3', 'I1,3'),('I1,1', 'I2,1'),('I1,1', 'I1,2')   ]             ]



    # Parameters
    loop_f = []
    for loop in loop_8v:
        for dx, dy in itertools.product(range(Lx), range(Ly)):
            res = translate_loop_(loop, dx, dy, Lx, Ly)
            if res:
                loop_f.append(res)
    
    
    
    loop_f_ = []
    for loop in loop_f:
        loop = [  tuple(sorted(a)) for a in loop]
        loop_f_.append(loop)

    if region_tag:
        loop_f = []
        for loop in loop_f_:
            loop_flat = [item for tup in loop for item in tup]
            if has_common_elements(loop_flat, region_tag):
                 loop_f.append(loop)

    loop_f_ = loop_f

    
    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_f_]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_f_ = [list(tup) for tup in unique_sublists]
    return loop_f_







def produce_gloops(Lx, Ly, region_tag=[],):
    loop_8v = [[('I1,0', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I1,1', 'I1,2'), ('I0,1', 'I1,1'), ('I0,0', 'I0,1')]]
    loop_8v += [[('I1,1', 'I2,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I1,1', 'I1,2'), ('I0,1', 'I1,1'), ('I0,0', 'I0,1')]]
    
    loop_8v += [[('I1,0', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I1,1', 'I2,1'), ('I1,1', 'I1,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    loop_8v += [[('I0,1', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I1,1', 'I2,1'), ('I1,1', 'I1,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    
    loop_8v += [[('I1,1', 'I1,2'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I1,1', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    loop_8v += [[('I0,1', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I1,1', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    
    loop_8v += [[('I1,1', 'I1,2'),('I0,1', 'I1,1'), ('I1,0', 'I1,1'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2')]]
    loop_8v += [[('I1,1', 'I2,1'),('I0,1', 'I1,1'), ('I1,0', 'I1,1'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2')]]




    loop_8v += [[('I0,0', 'I0,1'), ('I0,2', 'I1,2'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I1,1', 'I1,2'), ('I0,1', 'I0,2'), ('I0,3', 'I1,3')]]
    loop_8v += [[('I0,0', 'I0,1'), ('I0,1', 'I1,1'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I1,1', 'I1,2'), ('I0,1', 'I0,2'), ('I0,3', 'I1,3')]]
    
    loop_8v += [[('I0,0', 'I0,1'), ('I0,1', 'I1,1'), ('I0,2', 'I1,2'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I0,1', 'I0,2'), ('I0,3', 'I1,3')]]
    loop_8v += [[('I0,0', 'I0,1'), ('I0,1', 'I1,1'), ('I0,2', 'I1,2'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I1,1', 'I1,2'), ('I0,3', 'I1,3')]]



    loop_8v += [[('I1,0', 'I1,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I1,0', 'I2,0'),('I1,0', 'I2,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I1,1', 'I2,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]
    loop_8v += [[('I2,0', 'I2,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I1,0', 'I2,0'),('I1,0', 'I2,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I1,1', 'I2,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]
    
    loop_8v += [[('I1,0', 'I1,1'), ('I2,0', 'I2,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I1,0', 'I2,0'),('I1,0', 'I2,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]
    loop_8v += [[('I1,0', 'I1,1'), ('I2,0', 'I2,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I1,1', 'I2,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]




 
    # Parameters
    loop_f = []
    for loop in loop_8v:
        for dx, dy in itertools.product(range(Lx), range(Ly)):
            res = translate_loop(loop, dx, dy, Lx, Ly)
            if res:
                loop_f.append(res)
    
    
    
    loop_f_ = []
    for loop in loop_f:
        loop = [  tuple(sorted(a)) for a in loop]
        loop_f_.append(loop)


    if region_tag:
        loop_f = []
        for loop in loop_f_:
            loop_flat = [item for tup in loop for item in tup]
            if has_common_elements(loop_flat, region_tag):
                 loop_f.append(loop)

    loop_f_ = loop_f

    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_f_]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_f_ = [list(tup) for tup in unique_sublists]
    return loop_f_




def produce_gloops_(Lx, Ly, region_tag=[]):
    loop_8v = [[('I1,0', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I1,1', 'I1,2'), ('I0,1', 'I1,1'), ('I0,0', 'I0,1')]]
    loop_8v += [[('I1,1', 'I2,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I1,1', 'I1,2'), ('I0,1', 'I1,1'), ('I0,0', 'I0,1')]]
    
    loop_8v += [[('I1,0', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I1,1', 'I2,1'), ('I1,1', 'I1,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    loop_8v += [[('I0,1', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I1,1', 'I2,1'), ('I1,1', 'I1,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    
    loop_8v += [[('I1,1', 'I1,2'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I1,1', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    loop_8v += [[('I0,1', 'I1,1'), ('I0,0', 'I1,0'), ('I1,0', 'I1,1'), ('I1,1', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2'), ('I0,0', 'I0,1')]]
    
    loop_8v += [[('I1,1', 'I1,2'),('I0,1', 'I1,1'), ('I1,0', 'I1,1'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2')]]
    loop_8v += [[('I1,1', 'I2,1'),('I0,1', 'I1,1'), ('I1,0', 'I1,1'), ('I1,0', 'I2,0'), ('I2,0', 'I2,1'), ('I2,1', 'I2,2'), ('I1,2', 'I2,2'), ('I0,2', 'I1,2'), ('I0,1', 'I0,2')]]




    loop_8v += [[('I0,0', 'I0,1'), ('I0,2', 'I1,2'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I1,1', 'I1,2'), ('I0,1', 'I0,2'), ('I0,3', 'I1,3')]]
    loop_8v += [[('I0,0', 'I0,1'), ('I0,1', 'I1,1'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I1,1', 'I1,2'), ('I0,1', 'I0,2'), ('I0,3', 'I1,3')]]
    
    loop_8v += [[('I0,0', 'I0,1'), ('I0,1', 'I1,1'), ('I0,2', 'I1,2'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I0,1', 'I0,2'), ('I0,3', 'I1,3')]]
    loop_8v += [[('I0,0', 'I0,1'), ('I0,1', 'I1,1'), ('I0,2', 'I1,2'),('I0,0', 'I1,0'), ('I0,2', 'I0,3'), ('I1,2', 'I1,3'),('I1,0', 'I1,1'), ('I1,1', 'I1,2'), ('I0,3', 'I1,3')]]



    loop_8v += [[('I1,0', 'I1,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I1,0', 'I2,0'),('I1,0', 'I2,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I1,1', 'I2,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]
    loop_8v += [[('I2,0', 'I2,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I1,0', 'I2,0'),('I1,0', 'I2,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I1,1', 'I2,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]
    
    loop_8v += [[('I1,0', 'I1,1'), ('I2,0', 'I2,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I1,0', 'I2,0'),('I1,0', 'I2,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]
    loop_8v += [[('I1,0', 'I1,1'), ('I2,0', 'I2,1'), ('I0,0', 'I0,1'),('I0,0', 'I1,0'),('I2,0', 'I3,0'),('I0,1', 'I1,1'),('I1,1', 'I2,1'),('I2,1', 'I3,1'),('I3,0', 'I3,1') ]]




    
    # Parameters
    loop_f = []
    for loop in loop_8v:
        for dx, dy in itertools.product(range(Lx), range(Ly)):
            res = translate_loop_(loop, dx, dy, Lx, Ly)
            if res:
                loop_f.append(res)
    

    
    
    loop_f_ = []
    for loop in loop_f:
        loop = [  tuple(sorted(a)) for a in loop]
        loop_f_.append(loop)

    if region_tag:
        loop_f = []
        for loop in loop_f_:
            loop_flat = [item for tup in loop for item in tup]
            if has_common_elements(loop_flat, region_tag):
                 loop_f.append(loop)

    loop_f_ = loop_f


    
    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_f_]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_f_ = [list(tup) for tup in unique_sublists]
    return loop_f_






def loss_peps(peps, peps_fix, opt, copt, cost_f, val_=1., chi_bmps= 60, mode = "appr",cutoff=0.0 ):
    

    def apply_hyperoptimized_compressed(tn, copt, chi, output_inds=None, tree_gauge_distance=4, progbar=False, 
                                        cutoff=1.e-12, equalize_norms=False):
        
        tn.full_simplify_(seq='R', split_method='svd', inplace=True)
        
        tree = tn.contraction_tree(copt)
        tn_ = tn.copy()
        
        flops = tree.contraction_cost(log=10)
        peak = tree.peak_size(log=2)
        
        tn_.contract_compressed_(
            optimize=tree,
            output_inds=output_inds,
            max_bond=chi,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
        )
        return tn_, (flops, peak)

    
    
    
    #print((peps.H & peps).contract(all, optimize=opt))
    peps.add_tag('KET')
    peps_fix.add_tag('KET')
    
    pepsH = peps.conj().retag({'KET': 'BRA'})

    norm = pepsH | peps
    norm_ = pepsH | peps_fix
    if mode == "mps":

        val_0 = norm.contract_boundary(max_bond=chi_bmps, 
                                           final_contract_opts={"optimize": opt}, 
                                           #cutoff=1e-14,
                                           progbar=True,
                                           layer_tags=['KET', 'BRA'],
                                           #max_separation=1,
                                          )
        val_1 = norm_.contract_boundary(max_bond=chi_bmps, 
                                           final_contract_opts={"optimize": opt}, 
                                           #cutoff=1e-14,
                                           progbar=True,
                                           layer_tags=['KET', 'BRA'],
                                           #max_separation=1,
                                       )

    if mode == "hyper":
    
        overlap, (flops, peak) = apply_hyperoptimized_compressed(norm, copt, chi_bmps, cutoff=cutoff)
        # main, exp=(overlap.contract(all), overlap.exponent)
        val_0 = overlap^all 
        print(val_0)
        overlap, (flops, peak) = apply_hyperoptimized_compressed(norm_, copt, chi_bmps, cutoff=cutoff)
        # main, exp=(overlap.contract(all), overlap.exponent)
        val_1 = overlap^all #* 10**(exp)
        print(val_1, overlap.exponent)
    
    
    
    if mode == "exact":
        val_0 =  norm.contract(all, optimize=opt) 
        val_1 =  norm_.contract(all, optimize=opt) 
    
    if cost_f == "fid":
        val_0 = autoray.do("abs", val_0)
        val_3 = autoray.do("sqrt", val_0)
        val_4 = autoray.do("sqrt", val_)
        val_1 = autoray.do("abs", val_1)
        return 1 - ( val_1/( val_3 * val_4 ) )
    
    elif cost_f == "logfidelity":
        val_0 = autoray.do("abs", val_0)
        val_0 = autoray.do("log", val_0)
        
        val_4 = autoray.do("log", val_)
        
        val_1 = autoray.do("abs", val_1)
        val_1 = autoray.do("log", val_1)
        return -val_1 + (val_0 + val_4) * 0.5

    elif cost_f == "dis":
        val_0 = autoray.do("abs", val_0)
        val_2 = autoray.do("conj", val_1)
        return abs(val_ + val_0 -  val_1 - val_2)




def loss_peps_g(peps, peps_fix, opt=None, copt=None, cost_f="fid", chi_bmps= 60, mode = "appr",cutoff=0.0, progbar=False ):
    

    def apply_hyperoptimized_compressed(tn, copt, chi, output_inds=None, tree_gauge_distance=4, progbar=False, 
                                        cutoff=1.e-12, equalize_norms=False):
        
        tn.full_simplify_(seq='R', split_method='svd', inplace=True)
        
        tree = tn.contraction_tree(copt)
        tn_ = tn.copy()
        
        flops = tree.contraction_cost(log=10)
        peak = tree.peak_size(log=2)
        
        tn_.contract_compressed_(
            optimize=tree,
            output_inds=output_inds,
            max_bond=chi,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
        )
        return tn_, (flops, peak)

    
    
    
    #print((peps.H & peps).contract(all, optimize=opt))
    peps.add_tag('KET')
    peps_fix.add_tag('KET')
    
    pepsH = peps.conj().retag({'KET': 'BRA'})
    peps_fixH = peps_fix.conj().retag({'KET': 'BRA'})

    norm = pepsH | peps
    norm_ = pepsH | peps_fix
    norm_fix = peps_fixH | peps_fix

    if mode == "mps":

        val_0 = norm.contract_boundary(max_bond=chi_bmps, 
                                           final_contract_opts={"optimize": opt}, 
                                           #cutoff=1e-14,
                                           progbar=progbar,
                                           layer_tags=['KET', 'BRA'],
                                           #max_separation=1,
                                          )
        val_1 = norm_.contract_boundary(max_bond=chi_bmps, 
                                           final_contract_opts={"optimize": opt}, 
                                           #cutoff=1e-14,
                                           progbar=progbar,
                                           layer_tags=['KET', 'BRA'],
                                           #max_separation=1,
                                       )
        val_2 = norm_fix.contract_boundary(max_bond=chi_bmps, 
                                           final_contract_opts={"optimize": opt}, 
                                           #cutoff=1e-14,
                                           progbar=progbar,
                                           layer_tags=['KET', 'BRA'],
                                           #max_separation=1,
                                       )

        
    if mode == "hyper":
    
        overlap, (flops, peak) = apply_hyperoptimized_compressed(norm, copt, chi_bmps, cutoff=cutoff)
        # main, exp=(overlap.contract(all), overlap.exponent)
        val_0 = overlap^all 
        overlap, (flops, peak) = apply_hyperoptimized_compressed(norm_, copt, chi_bmps, cutoff=cutoff)
        # main, exp=(overlap.contract(all), overlap.exponent)
        val_1 = overlap^all #* 10**(exp)

        overlap, (flops, peak) = apply_hyperoptimized_compressed(norm_fix, copt, chi_bmps, cutoff=cutoff)
        # main, exp=(overlap.contract(all), overlap.exponent)
        val_2 = overlap^all #* 10**(exp)
    
    
    if mode == "exact":
        norm.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)
        norm_.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)
        norm_fix.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)

        val_0 =  norm.contract(all, optimize=opt) 
        val_1 =  norm_.contract(all, optimize=opt) 
        val_2 =  norm_fix.contract(all, optimize=opt) 
    
    if cost_f == "fid":
        val_0 = autoray.do("sqrt", val_0)
        val_2 = autoray.do("sqrt", val_2)
        val_1 = autoray.do("abs", val_1)
        return 1 - ( val_1 / autoray.do("abs", val_0 * val_2) )
    
    elif cost_f == "logfidelity":
        val_0 = autoray.do("abs", val_0)
        val_0 = autoray.do("log", val_0)
        
        val_4 = autoray.do("log", val_)
        
        val_1 = autoray.do("abs", val_1)
        val_1 = autoray.do("log", val_1)
        return -val_1 + (val_0 + val_4) * 0.5

    elif cost_f == "dis":
        val_0 = autoray.do("abs", val_0)
        val_2 = autoray.do("conj", val_1)
        return abs(val_ + val_0 -  val_1 - val_2)



def elm_red_uniq(loop_f):
    loop_f_ = []
    for loop in loop_f:
        loop = [  tuple(sorted(a)) for a in loop]
        loop_f_.append(loop)
    
    # Convert sublists to sorted tuples and create a DataFrame
    df = pd.DataFrame({'sublists': [tuple(sorted(sublist)) for sublist in loop_f_]})
    
    # Drop duplicates and retrieve original sublists
    unique_sublists = df.drop_duplicates()['sublists'].tolist()
    
    # Convert tuples back to lists
    loop_f_ = [list(tup) for tup in unique_sublists]
    return loop_f_








def produce_disjoint_loops(tn_flat, loop_length=4, intersect=True, index=0, corner_loop=[]):

    res = gen_loop(tn_flat, loop_length=loop_length, intersect=intersect)
    loop_pair = res["loop_pair"]
    
    if not corner_loop:
        corner_loop = loop_pair[index]

        corner_loop = loop_pair[index]
    
        #sort edges ina descing order
        corner_loop = [tuple(sorted(i)) for i in corner_loop]
        #include verticies of currunt loop 
        corner_loop_ = [item for tup in corner_loop for item in tup]
    else:    
        #sort edges ina descing order
        corner_loop = [tuple(sorted(i)) for i in corner_loop]
        #include verticies of currunt loop 
        corner_loop_ = [item for tup in corner_loop for item in tup]

    
    loop_f = []
    for loop in loop_pair:
        loop = [tuple(sorted(i)) for i in loop]
        loop_ = [item for tup in loop for item in tup]
        
        common_elements = set(loop_).intersection(set(corner_loop_))

        #if there is no common verticies then add loop + loop_corner
        if not common_elements:
            loop_f.append( loop+corner_loop )

    return loop_f


def produce_disjoint_loops_(tn_flat, loop_length=4, index=0, corner_loop = []):
    res = gen_loop_(tn_flat, loop_length=loop_length)
    loop_pair = res["loop_pair"]

    if not corner_loop:

        corner_loop = loop_pair[index]
    
        #sort edges ina descing order
        corner_loop = [tuple(sorted(i)) for i in corner_loop]
        #include verticies of currunt loop 
        corner_loop_ = [item for tup in corner_loop for item in tup]
    else:    
        #sort edges ina descing order
        corner_loop = [tuple(sorted(i)) for i in corner_loop]
        #include verticies of currunt loop 
        corner_loop_ = [item for tup in corner_loop for item in tup]
    
    loop_f = []
    for loop in loop_pair:
        loop = [tuple(sorted(i)) for i in loop]
        loop_ = [item for tup in loop for item in tup]
        
        common_elements = set(loop_).intersection(set(corner_loop_))
    
        #if there is no common verticies then add loop + loop_corner
        if not common_elements:
            loop_f.append( loop+corner_loop )

    return loop_f








def env_peps_xl(peps, tn_left, x, Lx=4, Ly=4, opt=None, chi = 4, can_dis=4, cutoff=1e-14):

    peps = peps.copy()
    tag = f"X{x}"
    
    if x == 0:
        mps_up = peps.select([tag,"KET"],which="all" )
        mps_down = peps.select([tag,"BRA"],which="all" )
        mps_down = mps_down*1.0
     
        mps_left = mps_up*1.

        if "pepo" in peps.tags:
            mpo = peps.select([tag,"pepo"],which="all" )
        
            mps_left = (mps_left & mpo)
            for j in range(Ly):
                mps_left.contract_tags_(f"Y{j}", optimize=opt)
            mps_left.fuse_multibonds_(inplace=True)
            mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
        
    
        
        mps_left = (mps_down & mps_left)

        for j in range(Ly):
            mps_left.contract_tags_(f"Y{j}", optimize=opt, )
        mps_left.fuse_multibonds_(inplace=True)
        
        

        mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
        
        tn_left[x] = mps_left
    
    
    else:
            
        mps_left_ = tn_left[x-1].copy()
    

        mps_up = peps.select([tag,"KET"],which="all" )
        mps_down = peps.select([tag,"BRA"],which="all" )
   
    
        mps_left = (mps_up & mps_left_)
        for j in range(Ly):
            mps_left.contract_tags_(f"Y{j}", optimize=opt)
        mps_left.fuse_multibonds_(inplace=True)
        mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
    
    
        if "pepo" in peps.tags:
            mpo = peps.select([tag, "pepo"],which="all" )
            mps_left = (mps_left & mpo)
            for j in range(Ly):
                mps_left.contract_tags_(f"Y{j}", optimize=opt, )
            mps_left.fuse_multibonds_(inplace=True)
    
    
            mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
    
    
        mps_left = (mps_down & mps_left)
        for j in range(Ly):
            mps_left.contract_tags_(f"Y{j}", optimize=opt, )
        mps_left.fuse_multibonds_(inplace=True)
        
            
        mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
        
        tn_left[x] = mps_left


def env_peps_xr(peps, tn_left, x, Lx=4, Ly=4, opt=None, chi = 4, can_dis=4, cutoff=1e-14):

    peps = peps.copy()
    tag = f"X{x}"
    
    if x == Lx-1:
        mps_up = peps.select([tag,"KET"],which="all" )
        mps_down = peps.select([tag,"BRA"],which="all" )
    
        mps_left = mps_up*1.0
        mps_down = mps_down*1.0
        if "pepo" in peps.tags:
            mpo = peps.select([tag,"pepo"],which="all" )
            mpo = mpo * 1.
            mps_left = (mps_left & mpo)
            for j in range(Ly):
                mps_left.contract_tags_(f"Y{j}", optimize=opt, )
            mps_left.fuse_multibonds_(inplace=True)
            mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
        
    
        
        mps_left = (mps_down & mps_left)

        for j in range(Ly):
            mps_left.contract_tags_(f"Y{j}", optimize=opt, )
        mps_left.fuse_multibonds_(inplace=True)
        
        

        mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
        tn_left[x] = mps_left
    
    
    else:
            
        mps_left_ = tn_left[x+1].copy()
    

        mps_up = peps.select([tag,"KET"],which="all" )
        mps_down = peps.select([tag,"BRA"],which="all" )
        mps_down = mps_down*1.0
   
    
        mps_left = (mps_up & mps_left_)
        for j in range(Ly):
            mps_left.contract_tags_(f"Y{j}", optimize=opt)
        mps_left.fuse_multibonds_(inplace=True)
        mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
    
    
        if "pepo" in peps.tags:
            mpo = peps.select([tag,"pepo"],which="all" )
            mps_left = (mps_left & mpo)
            for j in range(Ly):
                mps_left.contract_tags_(f"Y{j}", optimize=opt, )
            mps_left.fuse_multibonds_(inplace=True)
    
    
            mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
    
    
        mps_left = (mps_down & mps_left)
        for j in range(Ly):
            mps_left.contract_tags_(f"Y{j}", optimize=opt, )
        mps_left.fuse_multibonds_(inplace=True)
        
            
        mps_left.compress_all(inplace=True, **{"max_bond":chi, "canonize_distance":can_dis, "cutoff":cutoff})
        
        tn_left[x] = mps_left
    

def has_common_elements(list1, list2):
    return any(elem in list2 for elem in list1)



def gen_loop_(tn, loop_length=4, region_tag=[], site_tags=[]):

    gen = qu.tensor.networking.gen_loops(tn, max_loop_length=loop_length)
    inds_loop = []
    loop_pair = []
    tags_loops = []
    for count,  g in enumerate(gen):
        tn_l = []
        inds_l = []
        for tid in g:
            tn_ = tn.tensor_map[tid]
            tn_l.append(tn_)
            inds_l.append(list(tn_.inds))
        
        inds_l = list(itertools.chain.from_iterable(inds_l))
        element_counts = Counter(inds_l)
        inds_l = [elem for elem, count in element_counts.items() if count == 2]
    
        if len(inds_l)==loop_length:
            inds = inds_l 
            
            tids_pair = [    list(tn.ind_map[indx])   for indx in inds]
            pair_tags = []
            for t_pair in tids_pair:
                tid1, tid2 = t_pair
                tags1 = tn.tensor_map[tid1].tags            
                tags2 = tn.tensor_map[tid2].tags            
                pattern = re.compile(rf"I-?\d+(\.\d+)?(,-?\d+(\.\d+)?)*")
                # Filter the list
                tags1 = [elem for elem in tags1 if pattern.fullmatch(elem)]
                # Filter the list
                tags2 = [elem for elem in tags2 if pattern.fullmatch(elem)]
                
                if site_tags:
                    tags1 = [tag for tag in tags1 if tag in site_tags]
                    tags2 = [tag for tag in tags2 if tag in site_tags]
                
                
                
                pair_tags.append(  tuple(sorted((tags1[0], tags2[0])))     )
            
            tags= [list(tn.tensor_map[tid].tags) for tid in g]
            #set(region_tag).issubset(set(tags))
            tags = [item for sublist in tags for item in sublist]
            if region_tag and has_common_elements(region_tag, tags):
                inds_loop.append(inds)
                loop_pair.append(pair_tags)
                pair_tags_flat = list(itertools.chain.from_iterable(pair_tags))
                
                tags = [item for sublist in tags for item in sublist]
                tags = [tag for tag in tags if tag.startswith("I")]
                tags_loops.append(tags)
                #tn.draw(tags, fix=fix, highlight_inds=inds, highlight_inds_color="red", show_tags=False)
            if not region_tag:
                inds_loop.append(inds)
                loop_pair.append(pair_tags)
                pair_tags_flat = list(itertools.chain.from_iterable(pair_tags))
                
                tags = [item for sublist in tags for item in sublist]
                tags = [tag for tag in tags if tag.startswith("I")]
                tags_loops.append(tags)
                #tn.draw(tags, fix=fix, highlight_inds=inds, highlight_inds_color="red", show_tags=False)

    res = {"loop_pair":loop_pair, "tags_loops":tags_loops, "inds_loop":inds_loop}
    return res


def env_rho(info_pass, tree_gauge_distance=4, f_max = 14 , peak_max=34, prgbar=True, 
            external_opt=None):

    tn_l = info_pass["tn_l"]
    reg_reindex = info_pass["reg_reindex"]
    reg_tags = info_pass["reg_tags"]
    chi = info_pass["chi"]
    inds_rho = info_pass["inds_rho"]

    leftinds_rho = info_pass["leftinds_rho"]
    rightinds_rho = info_pass["rightinds_rho"]
    
    rho_l = []
    rhodata_l = []
    flops_l = []
    peak_l = []
    tn_rho_l = []


    with tqdm(total=len(tn_l),  desc="rho", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
    
        for count, tn_ in enumerate(tn_l):
            tn_ = tn_.copy()
        
            if reg_reindex:
                tn_bra = tn_.select(["BRA"], which="any")
                tn_bra.reindex_(reg_reindex)
            
            # tn_.draw(["proj"]+["Mg"], 
            #          edge_alpha=1.0, edge_scale=1.0, 
            #     fix=info_pass["fix"], node_outline_darkness=0.20, node_outline_size=1.0,     
            #     edge_color='gray', highlight_inds_color="darkred",
            #          show_tags=False, legend=False, node_scale=1.2, figsize=(4,4))
    
            tn_rho_l.append(tn_)
            tn_.full_simplify_(seq='R', split_method='svd', inplace=True)
        
            if not inds_rho:
                inds_rho = None
        

            
            res_ = req_backend(progbar=False, chi=chi, max_repeats=2**8)
            opt = res_["opt"]
            copt = res_["copt"]

            if external_opt:
                opt = external_opt

            
            tree = tn_.contraction_tree(optimize=opt, output_inds=inds_rho)
            flops = tree.contraction_cost()
            peak = tree.peak_size(log=2)
            if flops<1:
                flops = 1
            flops = np.log10(flops)
            
            if chi and flops>10:
                rho, (flops, peak) = apply_hyperoptimized_compressed(tn_, copt, chi, output_inds=inds_rho, 
                                                         tree_gauge_distance=tree_gauge_distance, progbar=False, 
                                                     equalize_norms=None, f_max = f_max , peak_max=peak_max)
            
                if flops<1:
                    flops = 1
                flops = np.log10(flops)
                if rho:
                    rho = rho^all 

            elif flops<=10:
                rho = tn_.contract(optimize=opt, output_inds=inds_rho)
                
            if flops<1.:
                flops = 1.

            flops_l.append(flops)
            peak_l.append(peak)


            pbar.set_postfix({"flops": max(flops_l), 
                              "peak": max(peak_l),
                              })
            pbar.refresh()
            pbar.update(1)

            
            # rho = tn_.contract(all, optimize=opt_)
        
            if inds_rho and rho:
                
                rho.transpose(*inds_rho, inplace=True)    
                rho_data = rho.data.reshape(2**len(leftinds_rho),2**len(leftinds_rho))
                rho_data = (rho_data + ar.dag(rho_data))*0.5
                
                rhodata_l.append(rho_data)
                
                rho_data = rho_data*1.
                
                shape_ = (2,)*(2*len(leftinds_rho))
                rho_data = rho_data.reshape(shape_)
                rho.modify(data=rho_data)
                rho_l.append(rho)
            
    
    if rhodata_l:
        rho_data = sum(rhodata_l)
        norm_loop = ar.do("trace", rho_data)    
        rho_data /= norm_loop

        res = {"rho":rho, "rho_l":rho_l, "norm_loop":norm_loop, "rhodata_l":rhodata_l, "rho_data":rho_data, "flops_l":flops_l, "peak_l":peak_l, "tn_rho_l":tn_rho_l}
    
        return res
    else:
        return None



def env_loop(info_pass, prgbar=True):
    tn_l = info_pass["tn_l"]
    reg_reindex = info_pass["reg_reindex"]
    reg_tags = info_pass["reg_tags"]
    chi = info_pass["chi"]
    inds_rho = info_pass["inds_rho"]

    rho_l = []
    rhodata_l = []
    flops_l = []
    peak_l = []
    tn_rho_l = []

    with tqdm(total=len(tn_l),  desc="env", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:

        for count, tn_ in enumerate(tn_l):
            tn_ = tn_.copy()
        
            if reg_reindex:
                tn_bra = tn_.select(["BRA"], which="any")
                tn_bra.reindex_(reg_reindex)
                
                tags_ = list(reg_tags.keys())
                
                tn_left = tn_.select([tags_[0], "KET"], which="all")
                tn_right = tn_.select([tags_[1], "KET"], which="all")
                indx_ = [ i for i in tn_left.outer_inds() if i in tn_right.outer_inds() ][0]
        
                map_ket_l = {indx_:"up_l"}
                map_ket_r = {indx_:"up_r"}
        
                
                tn_left.reindex_(map_ket_l)
                tn_right.reindex_(map_ket_r)
                
                tags_ = list(reg_tags.keys())
        
                
                tn_left = tn_.select([tags_[0], "BRA"], which="all")
                tn_right = tn_.select([tags_[1], "BRA"], which="all")
                
                indx_ = [ i for i in tn_left.outer_inds() if i in tn_right.outer_inds()  ][0]
                
                map_bra_l = {indx_:"low_l"}
                map_bra_r = {indx_:"low_r"}
        
                tn_left.reindex_(map_bra_l)
                tn_right.reindex_(map_bra_r)
        
                
            tn_rho_l.append(tn_)
            tn_.full_simplify_(seq='R', split_method='svd', inplace=True)
        
            if not inds_rho:
                inds_rho = None
        
            
            to_backend, opt_, opt, copt = req_backend(progbar=False, chi=chi, max_repeats=2**8)
            rho, (flops, peak) = apply_hyperoptimized_compressed(tn_, copt, chi, output_inds=inds_rho, 
                                                         tree_gauge_distance=4, progbar=False, 
                                                         equalize_norms=None, f_max = 15 , peak_max=34)
            if flops<1.:
                flops=1.e-4
            
            flops_l.append(np.log10(flops))
            peak_l.append(peak)
    
            pbar.set_postfix({"flops": np.log10(max(flops_l)), 
                              "peak": max(peak_l),
                              })
            pbar.refresh()
            pbar.update(1)
    
            
            if rho:
                rho = rho^all
                inds_l = [inds_rho[0], inds_rho[1]]+["up_l","up_r"]+[inds_rho[2], inds_rho[3]]+["low_l","low_r"]
                rho.transpose_(*inds_l)
        
                rhoh = rho.copy()
                rhoh = rhoh.conj()
        
                inds_l = [inds_rho[2], inds_rho[3]]+["low_l","low_r"]+[inds_rho[0], inds_rho[1]]+["up_l","up_r"]
                rhoh.transpose_(*inds_l)
                rho_data = (rho.data+rhoh.data)*.5
                
                rho.modify(data=rho_data)
                
                rho_l.append(rho)
    
        



    res = {"env":sum(rho_l),"rho_l":rho_l,  "flops_l":flops_l, "peak_l":peak_l, "tn_rho_l":tn_rho_l}
    return res









def bp_info_rho(cor):
    
    
    leftinds_rho = []
    rightinds_rho = []
    reg_tags = []
    reg_reindex = {}
    for cor_ in cor:
        x, y = cor_ 
        reg_tags.append(f"I{x},{y}")
        reg_reindex |= {f"k{x},{y}":f"b{x},{y}"}
        leftinds_rho.append(f"k{x},{y}")
        rightinds_rho.append(f"b{x},{y}")
    inds_rho = leftinds_rho + rightinds_rho
    
    res_cor = {"inds_rho":inds_rho, "reg_reindex":reg_reindex, "reg_tags":reg_tags, "leftinds_rho":leftinds_rho}
    res_cor |= {"rightinds_rho":rightinds_rho}
    return res_cor



def get_pairs_mulitexcit(Lx, Ly, length=2):
    pair_excited = [  ] 
    pair_gs = [   tuple(sorted((f"I{i},{j}", f"I{i+1},{j}")))  for i in range(1,Lx-1, 2) for j in range(Ly)] 

    excited_ = []
    for i in range(length):
        excited_ += list(combinations(pair_gs, i))
    
    gs_ = [ ]
    
    for count in range(len(excited_)):
            gs_.append(list(set(pair_gs) - set(excited_[count])))


    return gs_, excited_

def get_pairs_mulitexcit_(Lx, Ly, length=2):
    pair_excited = [  ] 
    pair_gs = [  tuple(sorted((f"I{i},{j}", f"I{(i+1)%Lx},{j}")))  for i in range(1, Lx, 2) for j in range(Ly)] 

    excited_ = []
    for i in range(length):
        excited_ += list(combinations(pair_gs, i))
    
    gs_ = [ ]
    
    for count in range(len(excited_)):
            gs_.append(list(set(pair_gs) - set(excited_[count])))


    return gs_, excited_


def square_loop_l4_(Lx, Ly):
    loop_0 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (1, 1)),((0, 1), (1, 1))]
    loop_4l = []
    for i in range(Lx-1):
        for j in range(0,Ly-1):
            local_loop = []
            for (x1, y1), (x2, y2) in loop_0:
                local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
            loop_4l.append(local_loop)  
    
    
    return loop_4l



def square_loop_l4(Lx, Ly):
    loop_0 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (1, 1)),((0, 1), (1, 1))]
    loop_4l = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_4l.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_4l.append(local_loop)  
    
    return loop_4l


def loop_line(Lx, Ly):
    loop = []
    for j in range(Ly):
        loop_0 = [ ((i, j), ((i+1)%Lx, j)) for i in range(Lx)  ]
        loop.append(loop_0)

    for i in range(Lx):
        loop_0 = [ ((i, j), (i, (j+1)%Ly)) for j in range(Ly)  ]
        loop.append(loop_0)

    return loop




def square_loop_l8(Lx, Ly):
    loop_0 = [((0, 0), (1, 0)),((1, 0), (2, 0)),((0, 2), (1, 2)),((1, 2), (2, 2)), ]
    loop_0 += [((0, 0), (0, 1)),((0, 1), (0, 2)),((2, 0), (2, 1)),((2, 1), (2, 2)), ]

    loop_4l = []
    for i in range(Lx-1):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_4l.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_4l.append(local_loop)  
    
    return loop_4l

def square_loop_l6(Lx, Ly):
    loop_0 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (1, 1)),((0, 1), (0, 2)),((1, 1), (1, 2)), ((0, 2), (1, 2))]
    loop_6lv = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_6lv.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_6lv.append(local_loop)  
    
    loop_0 = [((0, 0), (1, 0)),((1, 0), (2, 0)),((0, 1), (1, 1)),((1, 1), (2, 1)), ((0, 0), (0, 1)),((2, 0), (2, 1))  ]
    loop_6lh = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_6lh.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_6lh.append(local_loop)  
    
    return loop_6lh, loop_6lv
    

def square_loop_l7(Lx, Ly):
    loop_0 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (1, 1)),((0, 1), (0, 2)),((1, 1), (1, 2)), ((0, 2), (1, 2)), ((0, 1), (1, 1))]
    loop_7lv = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_7lv.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_7lv.append(local_loop)  
    
    loop_0 = [((0, 0), (1, 0)),((1, 0), (2, 0)),((0, 1), (1, 1)),((1, 1), (2, 1)), ((0, 0), (0, 1)),((2, 0), (2, 1))  , ((1, 0), (1, 1))]
    loop_7lh = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_7lh.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_7lh.append(local_loop)  
    
    
    
    return loop_7lh, loop_7lv
 
    
def square_loop_l8_(Lx, Ly):
    loop_0 = [ ((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3)), ((1, 0), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (1, 3)) ]
    loop_0 += [ ((0, 0), (1, 0)), ((0, 3), (1, 3))  ]

    loop_8lv = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_8lv.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_8lv.append(local_loop)  
    
    loop_0 = [ ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)) ]
    loop_0 += [ ((0, 1), (1, 1)), ((1, 1), (2, 1)),((2, 1), (3, 1)) ]
    loop_0 += [ ((0, 0), (0, 1)), ((3, 0), (3, 1)) ]

    loop_8lh = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_8lh.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_8lh.append(local_loop)  
    
    
    
    return loop_8lh, loop_8lv
    

def square_loop_l8_rec(Lx, Ly):
    loop_0 = [ ((0, 0), (0, 1)), ((0, 1), (0, 2))  ]
    loop_0 += [ ((0, 0), (1, 0)), ((1, 0), (2, 0))   ]
    loop_0 += [ ((0, 2), (1, 2)), ((2, 0), (2, 1))   ]
    loop_0 += [ ((1, 1), (2, 1)), ((1, 1), (1, 2))  ]

    loop_8lv = []
    for i in range(Lx):
        if i<Lx-1:
            for j in range(0,Ly):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_8lv.append(local_loop)  
    
        else:
            for j in range(0,Ly-1):
                local_loop = []
                for (x1, y1), (x2, y2) in loop_0:
                    local_loop.append( (((x1+i)%Lx,(y1+j)%Ly),((x2+i)%Lx,(y2+j)%Ly))    )
                loop_8lv.append(local_loop)  
    
    
    return loop_8lv
    





def replace_(pos, count, gate_l_, where_l_, reverse=False):
            
            if pos == count:
                print("warning: X and Z at the same position: pos==count")
            
            X = gate_l_[count] # first gate
            Z = gate_l_[pos] # gate coming after that
            whereX = where_l_[count]
            whereZ = where_l_[pos]        


            if len(whereX) == 1:
                x, = whereX
                
                
                if len(whereZ)==2:
                    x_, y_ = whereZ
                    if reverse:
                        if x_==x:
                            g = np.einsum('pq, qjmn->pjmn', X, Z)
                        if y_==x:
                            g = np.einsum('pq, iqmn ->ipmn', X, Z)
                    else:
                        if x_==x:
                            g = np.einsum('ijmn,mp->ijpn', Z, X)
                        if y_==x:
                            g = np.einsum('ijmn,np->ijmp', Z, X)
                        gate_l_[pos] = g
                        where_l_[pos] = (x_,y_)
                        where_l_.pop(count)
                        gate_l_.pop(count)
                
                
                else:
                    x_, = whereZ
                    if reverse:
                        g = np.einsum('pq, qj->pj', X, Z)
                    else:
                        g = np.einsum('nm,mp->np', Z, X) 
                
                    gate_l_[pos] = g
                    where_l_[pos] = (x_,)
                    where_l_.pop(count)
                    gate_l_.pop(count)
            if len(whereX) == 2:            
                x, y= whereX
                if len(whereZ)==2:
                    x_, y_ = whereZ
                    if x_==x and y_==y:
                        g = np.einsum('pqij, ijmn->pqmn', Z, X)
                    if x_==y and y_==x:
                        g = np.einsum('pqij, jinm->pqmn', Z, X)
                
                    gate_l_[pos] = g
                    where_l_[pos] = (x_,y_)
                    where_l_.pop(count)
                    gate_l_.pop(count)
                

                else:
                    x_, = whereZ
                    if x_ == x:
                        g = np.einsum('pq, qjmn->pjmn', Z, X)
                    if x_ == y:
                        g = np.einsum('pq, iqmn ->ipmn', Z, X)
                
                    gate_l_[count] = g
                    where_l_[count] = (x,y)
                    where_l_.pop(pos)
                    gate_l_.pop(pos)
            

def reduce_(where_l, gate_l, reverse=False):
    for count, where in enumerate(where_l):
            #if len(where) == 1:
                pos = find(where_l, where, count)
                if pos:
                    replace_(pos, count, gate_l, where_l, reverse=reverse)
                    break
    
    return where_l, gate_l



def pepoXmpo_prep(Lx, Ly, to_backend, pauli, where_, cycle_peps=False):
    L = Lx * Ly
    # mpoz_ = mpo_z(Lx*Ly, pauli)
    # mpoz2_ = mpoz_.apply(mpoz_)
    # mpoz2_.compress( "left" )
    # mpoz3 = mpoz_.apply(mpoz2_)
    # mpoz4 = mpoz_.apply(mpoz3)
    # mpoz4.compress( "left" )


    mpoz = mpo_z_prod(Lx*Ly, pauli, where_=where_)
    mpoz2 = mpo_zz_center(Lx*Ly, pauli, where_=where_[0])
    mpozz = mpo_z_prod(Lx*Ly, pauli, where_=[where_[0], where_[0]+1])

    #mpoz4 = mpoz2.apply(mpoz2)
    #mpoz4.compress( "left" )
    
    
    to_backend_ = get_to_backend(to_backend)

    
    # mpoz_prod = mpo_z_prod(Lx*Ly, dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left")
    
    # cord = int(Lx*Ly/2)
    # mpoz_select = mpo_z_select(Lx*Ly,[cord-1, cord, cord+1], dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left")

    mpoz.apply_to_arrays(to_backend_)
    mpoz2.apply_to_arrays(to_backend_)
    mpozz.apply_to_arrays(to_backend_) 

    # mpoz_.apply_to_arrays(to_backend_)
    # mpoz2_.apply_to_arrays(to_backend_)
    # mpoz4.apply_to_arrays(to_backend_)

    mpo_l = [mpoz, mpozz, mpoz2]

    #pepo_z4 = quf.MPO_to_PEPO(mpoz4, lx, ly)
    #pepo_z2 = quf.MPO_to_PEPO(mpoz2, lx, ly)

    
    pepo_z = MPO_to_PEPO(mpoz, Lx, Ly, cycle_peps=cycle_peps)
    pepo_zz = MPO_to_PEPO(mpozz, Lx, Ly, cycle_peps=cycle_peps)
    pepo_z2 = MPO_to_PEPO(mpoz2, Lx, Ly, cycle_peps=cycle_peps)
    
    
    # pepo_z2 = MPO_to_PEPO(mpoz2, Lx, Ly, cycle_peps=cycle_peps)
    # pepo_z4 = MPO_to_PEPO(mpoz4, Lx, Ly, cycle_peps=cycle_peps)

    #print((mpoz.H & mpoz).contract())

    #print((pepo_z.H & pepo_z).contract())

    #print(pepo_z.show())
    #pepo_l = [mpoz, mpoz2, mpoz4]

    return mpo_l, [pepo_z, pepo_zz, pepo_z2]




def mpo_prep(Lx, Ly, to_backend, cycle_peps=False):

    mpoz = mpo_z(Lx*Ly, dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left")

    mpoz2 = mpoz.apply(mpoz)
    mpoz2.compress( "left" )
    mpoz3 = mpoz.apply(mpoz2)
    mpoz4 = mpoz.apply(mpoz3)
    mpoz4.compress( "left" )
    to_backend_ = get_to_backend(to_backend)



    mpoz_prod = mpo_z_prod(Lx*Ly, dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left")

    mpoz.apply_to_arrays(to_backend_)
    mpoz2.apply_to_arrays(to_backend_)
    mpoz4.apply_to_arrays(to_backend_)
    mpoz_prod.apply_to_arrays(to_backend_)



    mpo_l = [mpoz, mpoz2, mpoz4, mpoz_prod]



    #pepo_z4 = quf.MPO_to_PEPO(mpoz4, lx, ly)
    #pepo_z2 = quf.MPO_to_PEPO(mpoz2, lx, ly)

    #pepo_z = MPO_to_PEPO(mpoz2, Lx, Ly, cycle_peps=cycle_peps)
    #pepo_z = MPO_to_PEPO(mpoz, Lx, Ly, cycle_peps=cycle_peps)
    #print((mpoz.H & mpoz).contract())
    #print((pepo_z.H & pepo_z).contract())

    #print(pepo_z.show())
    #pepo_l = [mpoz, mpoz2, mpoz4]

    return mpo_l





def trotter_gates_3dpeps(Lx, Ly, Lz, site, depth_, dt, J, h, 
                        dtype = "complex128", 
                        cycle = False, 
                        max_bond_mpo=320, 
                        triangular=False,
                        style = "left", 
                        cutoff=1.e-12, 
                        basis="0", model="Ising"):
    
    XX = qu.pauli('X', dtype=dtype) & qu.pauli('X', dtype=dtype)
    ZZ = qu.pauli('Z', dtype=dtype) & qu.pauli('Z', dtype=dtype)
    Z = qu.pauli('Z', dtype=dtype) 
    X = qu.pauli('X', dtype=dtype)

    L = Lx * Ly * Lz
    if model == "Ising":

        H_ = qu.expm((ZZ * complex(0, -1.)/2 ) * 2 * dt * J).reshape(2,2,2,2)
        RX = qu.expm( (X * complex(0, -1.)/2.) * 2 * dt * h ) 
        H_.astype(dtype)
        RX.astype(dtype)


        list_basis = [basis] * L
        p_0_ = qtn.MPS_computational_state(list_basis)
        p_0_.astype_(dtype)



    def cor_(i, j, k):
        return (i,j)
    def cor_1d(i, j, k):
        return i*Ly*Lz + j*Lz+k


    # SQ gates
    gate_lSQ = []
    where_lSQ = []

    for i,j,k in  itertools.product(range(Lx), range(Ly), range(Lz)):
        where_lSQ.append((cor_(i, j, k),))
        gate_lSQ.append(RX)

    gate_lNN = []
    where_lNN = []
    if cycle:
        for i,j,k in  itertools.product(range(Lx), range(Ly), range(Lz)):
            where_lNN.append((cor_(i, j, k), cor_(i, (j+1)%Ly, k)))
            gate_lNN.append(H_)
    else:
        for i,j,k in  itertools.product(range(Lx), range(Ly-1), range(Lz)):
            where_lNN.append((cor_(i, j, k), cor_(i, j+1, k)))
            gate_lNN.append(H_)



    gate_lLR = []
    where_lLR = []

    if cycle:
        for i,j,k in  itertools.product(range(Lx), range(Ly), range(Lz)):
            where_lLR.append((cor_(i, j, k), cor_((i+1)%Lx, j, k)))
            gate_lLR.append(H_)
    else:
        for i,j,k in  itertools.product(range(Lx-1), range(Ly), range(Lz)):
            where_lLR.append((cor_(i, j, k), cor_(i+1, j, k)))
            gate_lLR.append(H_)

    where_l = where_lSQ + where_lNN + where_lLR
    gate_l = gate_lSQ + gate_lNN + gate_lLR
    

    # make Circuit class based on given gates:

    circ = qtn.Circuit(N=Lx*Ly*Lz, psi0=p_0_)        
    for count in range(depth_):
        for where in where_l:
            if len(where) == 1:
                x,= where
                x1, y1 = x
                circ.apply_gate('RX', 2. * dt * h, cor_1d(x1, y1,0), gate_round=count, **{"contract":False})
            else:
                x, y = where
                x1, y1 = x
                x2, y2 = y
                circ.apply_gate('RZZ', 2. * dt * J, cor_1d(x1, y1,0), cor_1d(x2, y2,0), gate_round=count, **{"contract":False})
    

    where_l = where_l * depth_
    gate_l = gate_l * depth_
    
    return gate_l, where_l, circ



def trotter_gates_3dpeps_lightcone(Lx, Ly, Lz, site, depth_, dt, J, h, 
                                   dtype = "complex128", 
                                    cycle = "periodic", 
                                    max_bond_mpo=320, 
                                    style = "left",
                                    cutoff=1.e-12,
                                    triangular=False,
                                    basis="0"
                                    ):

    XX = qu.pauli('X', dtype=dtype) & qu.pauli('X', dtype=dtype)
    ZZ = qu.pauli('Z', dtype=dtype) & qu.pauli('Z', dtype=dtype)
    Z = qu.pauli('Z', dtype=dtype) 
    X = qu.pauli('X', dtype=dtype)
    H_ = qu.expm((ZZ * complex(0, -1.) )* dt * J).reshape(2,2,2,2)
    RX = qu.expm( (X * complex(0, -1.)/2.) * dt * h) 

    L = Lx * Ly * Lz
    def cor_(i, j, k):
        return (i, j)
    def cor_1d(i, j, k):
        return i*Ly*Lz + j*Lz + k

    coor_dic = { cor_1d(i, j, k) : (i,j) for i,j,k in itertools.product(range(Lx), range(Ly), range(Lz)) }


    if cycle == "periodic":
        gate_dic = {}
        gate_dic = gate_dic | { ((i,j),): RX  for i,j,k in itertools.product(range(Lx), range(Ly), range(Lz)) }

        if Lx % 2 == 0 and Lx > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_((i+1)%Lx, j, k)) : H_ for i,j,k in itertools.product(range(0, Lx, 2), range(Ly), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_((i+1)%Lx, j, k)) : H_ for i,j,k in itertools.product(range(1, Lx, 2), range(Ly), range(Lz)) }  
        elif Lx > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_((i+1)%Lx, j, k)) : H_ for i,j,k in itertools.product(range(0, Lx, 2), range(Ly), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_((i+1)%Lx, j, k)) : H_ for i,j,k in itertools.product(range(1, Lx-1, 2), range(Ly), range(Lz)) }  

        if Ly % 2 == 0 and Ly > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, (j+1)%Ly, k)) : H_ for i,j,k in  itertools.product(range(Lx), range(0, Ly, 2), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, (j+1)%Ly, k)) : H_ for i,j,k in itertools.product(range(Lx), range(1,Ly,2), range(Lz)) }  
        elif Ly > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, (j+1)%Ly, k)) : H_ for i,j,k in  itertools.product(range(Lx), range(0, Ly, 2), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, (j+1)%Ly, k)) : H_ for i,j,k in itertools.product(range(Lx), range(1,Ly-1,2), range(Lz)) }  

        if Lz % 2 == 0 and Lz > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, (k+1)%Lz)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(0,Lz,2)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, (k+1)%Lz)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(1,Lz,2)) }  
        elif Lz > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, (k+1)%Lz)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(0,Lz,2)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, (k+1)%Lz)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(1,Lz-1,2)) }  

        if triangular:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_((i+1)%Lx, (j+1)%Ly, k)) : H_ for i,j,k in  itertools.product(range(Lx), range(Ly), range(Lz)) }  


    if cycle == "open":
        
        gate_dic = {}
        gate_dic = gate_dic | { ((i,j),): RX  for i,j,k in itertools.product(range(Lx), range(Ly), range(Lz)) }

        if Lx % 2 == 0 and Lx > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i+1, j, k)) : H_ for i,j,k in itertools.product(range(0, Lx-1, 2), range(Ly), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i+1, j, k)) : H_ for i,j,k in itertools.product(range(1, Lx-2, 2), range(Ly), range(Lz)) }  
        elif Lx > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i+1, j, k)) : H_ for i,j,k in itertools.product(range(0, Lx-2, 2),    range(Ly), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i+1, j, k)) : H_ for i,j,k in itertools.product(range(1, Lx-1, 2), range(Ly), range(Lz)) }  
        if Ly % 2 == 0 and Ly > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j+1, k)) : H_ for i,j,k in  itertools.product(range(Lx), range(0, Ly-1, 2), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j+1, k)) : H_ for i,j,k in itertools.product(range(Lx), range(1,Ly-2,2), range(Lz)) }  
        elif Ly > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j+1, k)) : H_ for i,j,k in  itertools.product(range(Lx), range(0, Ly-2, 2), range(Lz)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j+1, k)) : H_ for i,j,k in itertools.product(range(Lx), range(1,Ly-1,2), range(Lz)) }  

        if Lz % 2 == 0 and Lz > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, k+1)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(0,Lz-1,2)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, k+1)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(1,Lz-2,2)) }  
        elif Lz > 1:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, k+1)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(0,Lz-2,2)) }  
            gate_dic = gate_dic | { (cor_(i, j, k), cor_(i, j, k+1)) : H_ for i,j,k in itertools.product(range(Lx), range(Ly), range(1,Lz-1,2)) }  

        if triangular:
            gate_dic = gate_dic | { (cor_(i, j, k), cor_((i+1)%Lx, (j+1)%Ly, k)) : H_ for i,j,k in  itertools.product(range(Lx), range(Ly), range(Lz)) }  


    list_basis = [basis] * L
    p_0_ = qtn.MPS_computational_state(list_basis)


    circ = qtn.Circuit(N=Lx*Ly*Lz, psi0=p_0_)        
    for count in range(depth_):
        for where in gate_dic:
            if len(where) == 1:
                x, = where
                x1, y1 = x
                circ.apply_gate('RX', dt * h, cor_1d(x1, y1,0), gate_round=count, **{"contract":False})
            else:
                x, y = where
                x1, y1 = x
                x2, y2 = y
                circ.apply_gate('RZZ', -2. * dt * J, cor_1d(x1, y1,0), cor_1d(x2, y2,0), gate_round=count, **{"contract":False})




    round_tags = [f'ROUND_{i}' for i in range(depth_)]

    where_ = []
    for site_ in site:
        x, y = site_
        where_.append(cor_1d(x,y,0))
    lc_tags = list(circ.get_reverse_lightcone_tags(where=where_))
    lc_tags.remove("PSI0")

    #psi_lightcone=circ.get_psi_reverse_lightcone(where=where_, keep_psi0=False)
    #print(psi_lightcone)
    
    psi = circ.psi
    tags_cor = []
    #print(psi)

    for j in round_tags:
        tags_l = []
        for i in lc_tags:
            psi_=psi[i]
            if isinstance(psi_, tuple):
                t_ = list(psi_)
                tag_1 = list(t_[0].tags)
                tag_2 = list(t_[1].tags)
                if j in tag_1 and j in tag_2: 
                    tags_local = [ x for x in tag_1 if x.startswith('I')] + [ x for x in tag_2 if x.startswith('I')]
                    tags_l.append(tags_local)
            else:
                tag = list(psi_.tags )
                if j in tag:
                    tags_local = [ x for x in tag if x.startswith('I')] 
                    tags_l.append(tags_local)        
        tags_cor.append(tags_l)


    where_lightcone = []
    gate_lightcone = []
    for count in range(depth_):
        #gate_dic = {}
        for tag in tags_cor[count]:
            if len(tag) == 1:
                int_ = int(re.search(r'\d+', tag[0]).group())
                gate_lightcone.append(RX)
                where_lightcone.append((coor_dic[int_],))
                #gate_dic = gate_dic | { (coor_dic[int_],) : RX }
            if len(tag) == 2:
                int_1 = int(re.search(r'\d+', tag[0]).group())
                int_2 = int(re.search(r'\d+', tag[1]).group())
                gate_lightcone.append(H_)
                where_lightcone.append((coor_dic[int_1], coor_dic[int_2]))
                #gate_dic = gate_dic | { (coor_dic[int_1], coor_dic[int_2]) : H_ }



    
    return gate_lightcone, where_lightcone, circ



def trotter_gates_3dpeps_IBM(Lx, Ly, Lz, site, depth_,dt, J, h, dtype = "complex128", cycle = "periodic", 
                             max_bond_mpo=320, style = "left", cutoff=1.e-12,
                             model = "Ising",
                             ):
    ZZ = qu.pauli('Z', dtype=dtype) & qu.pauli('Z', dtype=dtype)
    X = qu.pauli('X', dtype=dtype) 
    H_ = qu.expm( ZZ * complex(0, -1.) * math.pi/4 ).reshape(2,2,2,2)
    RX = qu.expm(X * complex(0, -1.) * dt/2.)
    L = Lx * Ly * Lz
    def cor_(i, j, k):
        return (i, j)
    def cor_1d(i, j, k):
        return i*Ly*Lz + j*Lz + k


    coor_dic = { cor_1d(i, j, k) : (i,j) for i,j,k in itertools.product(range(Lx), range(Ly), range(Lz)) }


    single_site = [ ((i,j),) for i,j in itertools.product([0,4,8,12],[1,5,9]) ] + [ ((i,j),) for i,j in itertools.product([2,6,10,14],[3,7,11]) ] 

    ZZ_v = [ ((i,j), (i,j+1)) for i,j in itertools.product([0,4,8,12],[0,1]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([0,4,8,12],[4,5]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([0,4,8,12],[8,9]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([2,6,10,14],[2,3]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([2,6,10,14],[6,7]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([2,6,10,14],[10,11]) ] 


    
    gate_dic = { ((i,0),): RX  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { ((i,2),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,4),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,6),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,8),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,10),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,12),) : RX   for i in range(1,Lx,1) }
    gate_dic = gate_dic | { i : RX for i in single_site}

    #<ZZ> Horizontal
    gate_dic = gate_dic | { (cor_(i, 0, 0), cor_(i+1, 0, 0)) : H_  for i in range(0,Lx-2,1) }
    gate_dic = gate_dic | { (cor_(i, 2, 0), cor_(i+1, 2, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 4, 0), cor_(i+1, 4, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 6, 0), cor_(i+1, 6, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 8, 0), cor_(i+1, 8, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 10, 0), cor_(i+1, 10, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 12, 0), cor_(i+1, 12, 0)) : H_  for i in range(1,Lx-1,1) }
    
    #ZZ_vertical
    gate_dic = gate_dic | { i : H_  for i in ZZ_v }


    circ = qtn.Circuit(N=Lx*Ly*Lz)    
    for count in range(depth_):
        for where in gate_dic:
            if len(where) == 1:
                x, = where
                x1, y1 = x
                circ.apply_gate('RX', dt, cor_1d(x1, y1,0), gate_round=count, **{"contract":False})
            else:
                x, y = where
                x1, y1 = x
                x2, y2 = y
                circ.apply_gate('RZZ', -math.pi/4, cor_1d(x1, y1,0), cor_1d(x2, y2,0), gate_round=count, **{"contract":False})



    gate_lightcone = []
    for i in range(depth_):
        gate_lightcone.append(gate_dic)

    return gate_lightcone, circ, coor_dic


def trotter_gates_3dpeps_IBM_lightcone(Lx, Ly, Lz, site,depth_,dt, J, h, dtype = "complex128", cycle = "periodic", max_bond_mpo=320, style = "left", cutoff=1.e-12):
    ZZ = qu.pauli('Z', dtype=dtype) & qu.pauli('Z', dtype=dtype)
    X = qu.pauli('X', dtype=dtype) 
    H_ = qu.expm( ZZ * complex(0, -1.) * math.pi/4 ).reshape(2,2,2,2)
    RX = qu.expm(X * complex(0, -1.) * dt/2.)
    L = Lx * Ly * Lz
    def cor_(i, j, k):
        return (i, j)
    def cor_1d(i, j, k):
        return i*Ly*Lz + j*Lz + k


    coor_dic = { cor_1d(i, j, k) : (i,j) for i,j,k in itertools.product(range(Lx), range(Ly), range(Lz)) }
 
    single_site = [ ((i,j),) for i,j in itertools.product([0,4,8,12],[1,5,9]) ] + [ ((i,j),) for i,j in itertools.product([2,6,10,14],[3,7,11]) ] 

    ZZ_v = [ ((i,j), (i,j+1)) for i,j in itertools.product([0,4,8,12],[0,1]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([0,4,8,12],[4,5]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([0,4,8,12],[8,9]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([2,6,10,14],[2,3]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([2,6,10,14],[6,7]) ] 
    ZZ_v += [ ((i,j), (i,j+1)) for i,j in itertools.product([2,6,10,14],[10,11]) ] 


    gate_dic = {}

    gate_dic = gate_dic | { ((i,0),): RX  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { ((i,2),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,4),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,6),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,8),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,10),) : RX   for i in range(0,Lx,1) }
    gate_dic = gate_dic | { ((i,12),) : RX   for i in range(1,Lx,1) }
    gate_dic = gate_dic | { i : RX for i in single_site}

    #<ZZ> Horizontal
    gate_dic = gate_dic | { (cor_(i, 0, 0), cor_(i+1, 0, 0)) : H_  for i in range(0,Lx-2,1) }
    gate_dic = gate_dic | { (cor_(i, 2, 0), cor_(i+1, 2, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 4, 0), cor_(i+1, 4, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 6, 0), cor_(i+1, 6, 0)) : H_  for i in range(0,Lx-1,1) }


    gate_dic = gate_dic | { (cor_(i, 8, 0), cor_(i+1, 8, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 10, 0), cor_(i+1, 10, 0)) : H_  for i in range(0,Lx-1,1) }
    gate_dic = gate_dic | { (cor_(i, 12, 0), cor_(i+1, 12, 0)) : H_  for i in range(1,Lx-1,1) }


    gate_dic = gate_dic | { i : H_  for i in ZZ_v }

    #ZZ_vertical

    #print(cor_1d(6, 6,0), coor_dic[cor_1d(6, 6,0)])
    #print(cor_1d(5, 6,0), coor_dic[cor_1d(5, 6,0)])
    #print(cor_1d(7, 6,0), coor_dic[cor_1d(7, 6,0)])
    #print(cor_1d(6, 7,0), coor_dic[cor_1d(6, 7,0)])

    # for where in gate_dic:
    #     print(where)
        # if (6, 6) in where:
        #     print(where)


    #print(gate_dic)
    circ = qtn.Circuit(N=Lx*Ly*Lz)    
    for count in range(depth_):
        for where in gate_dic:
            if len(where) == 1:
                x, = where
                x1, y1 = x
                circ.apply_gate('RX', dt, cor_1d(x1, y1,0), gate_round=count, **{"contract":False})
            else:
                x, y = where
                x1, y1 = x
                x2, y2 = y
                circ.apply_gate('RZZ', -math.pi/4, cor_1d(x1, y1,0), cor_1d(x2, y2,0), gate_round=count, **{"contract":False})
    

    round_tags = [f'ROUND_{i}' for i in range(depth_)]

    x, y = site 
    lc_tags = list(circ.get_reverse_lightcone_tags(where=[cor_1d(x,y,0)]))
    lc_tags.remove("PSI0")

    #psi_lightcone=circ.get_psi_reverse_lightcone(where=[cor_1d(x,y,0)], keep_psi0=False)
    #print(psi_lightcone)
    
    psi = circ.psi
    tags_cor = []
    #print(psi)

    for j in round_tags:
        tags_l = []
        for i in lc_tags:
            psi_=psi[i]
            if isinstance(psi_, tuple):
                t_ = list(psi_)
                tag_1 = list(t_[0].tags)
                tag_2 = list(t_[1].tags)
                if j in tag_1 and j in tag_2: 
                    tags_local = [ x for x in tag_1 if x.startswith('I')] + [ x for x in tag_2 if x.startswith('I')]
                    tags_l.append(tags_local)
            else:
                tag = list(psi_.tags )
                if j in tag:
                    tags_local = [ x for x in tag if x.startswith('I')] 
                    tags_l.append(tags_local)        
        tags_cor.append(tags_l)


    where_lightcone = []
    gate_lightcone = []
    for count in range(depth_):
        gate_dic = {}
        for tag in tags_cor[count]:
            if len(tag) == 1:
                int_ = int(re.search(r'\d+', tag[0]).group())
                gate_dic = gate_dic | { (coor_dic[int_],) : RX }
            if len(tag) == 2:
                int_1 = int(re.search(r'\d+', tag[0]).group())
                int_2 = int(re.search(r'\d+', tag[1]).group())
                gate_dic = gate_dic | { (coor_dic[int_1], coor_dic[int_2]) : H_ }
        gate_lightcone.append(gate_dic)


    return gate_lightcone, circ, coor_dic

def to_2d(site, Lx, Ly, Lz):
    dic_ = {}
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                dic_ |= { f"{i*Ly*Lz + j*Lz + k}":(i,j)} 
    site_ = []
    for x in site:
        site_.append(dic_[f"{x}"])
    
    return site_


def trotter_gates(dt=0.5, J=-1, h=-1, Lx=2, Ly=2, Lz=2, hz=-0.5,
                  cycle = True, 
                  depth_= 1, 
                  list_basis = None, 
                  lightcone = None, 
                  swap_gate = False,
                  theta = 0,
                  trotter = 1, 
                  **kwargs
                  ):

    print("J, h, hz", J, h, hz, "dt", dt, "trotter", trotter, "L", Lx * Ly * Lz)
    psi0 = qtn.MPS_computational_state(["0"]*Lx*Ly*Lz)
    for t in psi0:
        vec = np.array([math.cos(theta), math.sin(theta)]) 
        shape = t.shape
        t.modify(data = vec.reshape(shape))


    dic_ = {}
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                dic_ |= { f"{i*Ly*Lz + j*Lz + k}":(i,j)} 

    site_2d = []
    if lightcone:
        for i in lightcone:
            cor = dic_[f"{i}"]
            site_2d.append(cor)
            print(f"1d_2d: ({i}) ---> ({cor})" )
    
    def cor_1d(i, j, k):
        return i*Ly*Lz + j*Lz + k

    def cor_1d_(x):
        i, j = x
        return i*Ly + j

    def cor_2d(i, j, k):
        return (i, j)

    #SQ gates
    where_lSQ_1d = []
    #where_lSQ_2d = []

    for i in range(Lx * Ly * Lz):
        where_lSQ_1d.append((i,))
        #where_lSQ_2d.append((cor_2d(i, j, k),))



    where_lNN_1d = []
    #where_lNN_2d = []
    if cycle:
        for i,j,k in  itertools.product(range(Lx), range(Ly), range(Lz)):
            where_lNN_1d.append((cor_1d(i, j, k), cor_1d(i, (j+1)%Ly, k)))
            #where_lNN_2d.append((cor_2d(i, j, k), cor_2d(i, (j+1)%Ly, k)))
    
    else :
        for i,j,k in  itertools.product(range(Lx), range(Ly-1), range(Lz)):
            where_lNN_1d.append((cor_1d(i, j, k), cor_1d(i, j+1, k)))
            #where_lNN_2d.append((cor_2d(i, j, k), cor_2d(i, j+1, k)))
        

    where_lLR_1d = []
    #where_lLR_2d = []
    if cycle:
        for i,j,k in  itertools.product(range(Lx), range(Ly), range(Lz)):
            where_lLR_1d.append((cor_1d(i, j, k), cor_1d((i+1)%Lx, j, k)))
            #where_lLR_2d.append((cor_2d(i, j, k), cor_2d((i+1)%Lx, j, k)))
            
    else:
        for i,j,k in  itertools.product(range(Lx-1), range(Ly), range(Lz)):
            where_lLR_1d.append((cor_1d(i, j, k), cor_1d(i+1, j, k)))
            #where_lLR_2d.append((cor_2d(i, j, k), cor_2d(i+1, j, k)))

    # where_l_absorbed = where_lNN_1d +  where_lLR_1d  
    if trotter == 1:
        where_l = where_lLR_1d + where_lNN_1d + where_lSQ_1d #+ where_lSQ_1d
    elif trotter ==2:    
        where_l = where_lSQ_1d + where_lLR_1d + where_lNN_1d + where_lSQ_1d
    #where_l2d = where_lSQ_2d + where_lNN_2d + where_lLR_2d

    circ = qtn.Circuit(N=Lx*Ly*Lz, psi0=psi0, 
                       #**{"gate_contract":False}, 
                       **kwargs
                       )

    gates = []
    
    for r in range(depth_):
        gate_round_ = 0
        for count, where in enumerate(where_l):
            if len(where) == 1:
                i, = where
                gates.append(Gate("rz", [ 2 * dt * hz], [i], round=r))
                gate_round_ += 1
                gates.append(Gate("rx", [ 2 * dt * h], [i], round=r))
                gate_round_ += 1
            else:
                if not swap_gate:
                    i, j = where
                    gates.append(Gate("rzz", [2. * dt * J], [i, j], round=r))
                    gate_round_ += 1
                if swap_gate:
                    i, j = where
                    i2d = dic_[f"{i}"]
                    j2d = dic_[f"{j}"]
                    *swaps, final = gen_long_range_swap_path(i2d, j2d, 
                                                            sequence=('av', 'bh', "ah", "bv"),
                                                            )
                    for pair in swaps:
                        x_, y_ = pair
                        gates.append(Gate.from_raw(qu.swap(), [cor_1d_(x_), cor_1d_(y_)], round=r))
                    x_, y_ = final
                    gates.append(Gate("rzz", [2. * dt * J], [cor_1d_(x_), cor_1d_(y_)], round=r))
                    for pair in reversed(swaps):
                        x_, y_ = pair
                        gates.append(Gate.from_raw(qu.swap(), [cor_1d_(x_), cor_1d_(y_)], round=r))

    gate_round = [gate_round_] * depth_
    
    if lightcone:
        lgates = []
        gate_round = [0] * depth_
        lightcone = set(lightcone)
        for g in reversed(gates):
            qs = set(g.qubits)
            if qs & lightcone:
                lgates.append(g)
                lightcone |= qs
                gate_round[g.round] += 1
        gates = lgates[::-1]


    for gate in gates:
        if gate.label in ["RZZ", "RX"]:
            circ.apply_gate(gate)
        else:
            circ.apply_gate(gate, **{"tags":"SWAP"})
    

    gate_l = []
    where_l = []
    for gate in gates:
        if len(gate.qubits) == 2:
            if gate.label in ["RZZ", "RX", "RZ"]:
                gate_l.append(gate.build_array().reshape(2,2,2,2))
            else:
                gate_l.append(qu.swap().reshape(2,2,2,2))
            x, y = gate.qubits
            where_l.append((x, y))
        if len(gate.qubits) == 1:
            gate_l.append(gate.build_array().reshape(2,2))
            x, = gate.qubits
            where_l.append( (x,) )

    # gate_l_absorbed = []
    # gate_round_absorbed = [len(where_l_absorbed)]* depth_    
    # where_l_absorbed = where_l_absorbed * depth_
    # gzz = Gate("rzz", [2. * dt * J], [0, 1]).build_array()
    # gx = Gate("rx", [2. * dt * h/4.], [0]).build_array()
    # gf = np.einsum('pi, qj,ijmn,ms,nt->pqst', gx, gx, gzz, gx,gx)
    # #gf = gf.reshape(2,2,2,2)
    # for i in range(len(where_l_absorbed)): 
    #     #print(i, gf)
    #     gate_l_absorbed.append(gf)


    # where_l2d = []
    # for i in where_l:
    #     if len(i) ==  1:
    #         x, = i
    #         where_l2d.append( (dic_[f"{x}"],) )
    #     if len(i) ==  2:
    #         x, y = i
    #         where_l2d.append( (dic_[f"{x}"], dic_[f"{y}"]) )



    # where_l2d_absorbed = []
    # for i in where_l_absorbed:
    #     x, y = i
    #     where_l2d_absorbed.append( (dic_[f"{x}"], dic_[f"{y}"]) )   

    #print(where_l_absorbed)
    return gate_l, where_l, gate_round, circ
    #return (gate_l_absorbed, where_l_absorbed, where_l2d_absorbed, site_2d), gate_round_absorbed, circ



def rank_simplify_leftinds(
    self,
    output_inds=None,
    equalize_norms=False,
    cache=None,
    max_combinations=500,
    inplace=False,
):
    """Simplify this tensor network by performing contractions that don't
    increase the rank of any tensors.

    Parameters
    ----------
    output_inds : sequence of str, optional
        Explicitly set which indices of the tensor network are output
        indices and thus should not be modified.
    equalize_norms : bool or float
        Actively renormalize the tensors during the simplification process.
        Useful for very large TNs. The scaling factor will be stored as an
        exponent in ``tn.exponent``.
    cache : None or set
        Persistent cache used to mark already checked tensors.
    inplace : bool, optional
        Whether to perform the rand reduction inplace.

    Returns
    -------
    TensorNetwork

    See Also
    --------
    full_simplify, column_reduce, diagonal_reduce
    """
    tn = self if inplace else self.copy()

    if output_inds is None:
        output_inds = tn._outer_inds

    # pairs of tensors we have already checked
    if cache is None:
        cache = set()

    # first parse all tensors
    scalars = []
    count = collections.Counter()
    for tid, t in tuple(tn.tensor_map.items()):
        # remove floating scalar tensors -->
        #     these have no indices so won't be caught otherwise
        if t.ndim == 0:
            tn.pop_tensor(tid)
            scalars.append(t.data)
            continue

        # ... and remove any redundant repeated indices on the same tensor
        t.collapse_repeated_()

        # ... also build the index counter at the same time
        count.update(t.inds)

    # this ensures the output indices are not removed (+1 each)
    count.update(output_inds)

    # special case, everything connected by one index
    trivial = len(count) == 1

    # sorted list of unique indices to check -> start with lowly connected
    def rank_weight(ind):
        return (
            tn.ind_size(ind),
            -sum(tn.tensor_map[tid].ndim for tid in tn.ind_map[ind]),
        )

    queue = qtn.oset(sorted(count, key=rank_weight))

    # number of tensors for which there will be more pairwise combinations
    # than max_combinations
    combi_cutoff = int(0.5 * ((8 * max_combinations + 1) ** 0.5 + 1))

    while queue:
        # get next index
        ind = queue.popright()

        # the tensors it connects
        try:
            tids = tn.ind_map[ind]
        except KeyError:
            # index already contracted alongside another
            continue

        # index only appears on one tensor and not in output -> can sum
        if count[ind] == 1:
            (tid,) = tids
            t = tn.tensor_map[tid]
            t.sum_reduce_(ind)

            # check if we have created a scalar
            if t.ndim == 0:
                tn.pop_tensor(tid)
                scalars.append(t.data)

            continue

        # otherwise check pairwise contractions
        cands = []
        combos_checked = 0

        if len(tids) > combi_cutoff:
            # sort size of the tensors so that when we are limited by
            #     max_combinations we check likely ones first
            tids = sorted(tids, key=lambda tid: tn.tensor_map[tid].ndim)

        for tid_a, tid_b in itertools.combinations(tids, 2):
            ta = tn.tensor_map[tid_a]
            tb = tn.tensor_map[tid_b]

            cache_key = ("rs", tid_a, tid_b, id(ta.data), id(tb.data))
            if cache_key in cache:
                continue

            combos_checked += 1

            # work out the output indices of candidate contraction
            involved = frequencies(itertools.chain(ta.inds, tb.inds))
            out_ab = []
            deincr = []
            for oix, c in involved.items():
                if c != count[oix]:
                    out_ab.append(oix)
                    if c == 2:
                        deincr.append(oix)
                # else this the last occurence of index oix -> remove it

            # check if candidate contraction will reduce rank
            new_ndim = len(out_ab)
            old_ndim = max(ta.ndim, tb.ndim)

            if new_ndim <= old_ndim:
                res = (new_ndim - old_ndim, tid_a, tid_b, out_ab, deincr)
                cands.append(res)
            else:
                cache.add(cache_key)

            if cands and (trivial or combos_checked > max_combinations):
                # can do contractions in any order
                # ... or hyperindex is very large, stop checking
                break

        if not cands:
            # none of the parwise contractions reduce rank
            continue

        _, tid_a, tid_b, out_ab, deincr = min(cands)
        ta = tn.pop_tensor(tid_a)
        tb = tn.pop_tensor(tid_b)
        #print(ta.data, tb.data)

        
        left_inds = []
        dic = {}
        ta_left_inds = ta.left_inds
        ta_right_inds = list(qtn.oset(ta.inds) - qtn.oset(ta_left_inds))

        if len(ta_right_inds) == 2:
            I_tags = [ i for i in ta.tags if i.startswith("I")]
            x = int(re.search(r'\d+', I_tags[0]).group())
            y = int(re.search(r'\d+', I_tags[1]).group())
            dic |= {ta_right_inds[0]:x}
            dic |= {ta_right_inds[1]:y}
        if len(ta_right_inds) == 1:
            I_tags = [ i for i in ta.tags if i.startswith("I")]
            x = int(re.search(r'\d+', I_tags[0]).group())
            dic |= {ta_right_inds[0]:x}


        left_inds_ = []
        tb_left_inds = tb.left_inds
        tb_right_inds = list(qtn.oset(tb.inds) - qtn.oset(tb_left_inds))
        if len(tb_right_inds) == 2:
            I_tags = [ i for i in tb.tags if i.startswith("I")]
            x = int(re.search(r'\d+', I_tags[0]).group())
            y = int(re.search(r'\d+', I_tags[1]).group())
            dic |= {tb_right_inds[0]:x}
            dic |= {tb_right_inds[1]:y}
        if len(tb_right_inds) == 1:
            I_tags = [ i for i in tb.tags if i.startswith("I")]
            x = int(re.search(r'\d+', I_tags[0]).group())
            dic |= {tb_right_inds[0]:x}

        left_inds_f = list(ta_left_inds) + list(tb_left_inds)     
        tab = ta.contract(tb, output_inds=out_ab)
        l_f = []
        r_f = []
        for indx in tab.inds:
            if indx in left_inds_f:
                l_f.append(indx)
            else:
                r_f.append(indx)
        inds = r_f + l_f
        tab.transpose(*inds, inplace=True)
        tab.modify(left_inds=l_f)
        tags = list(tab.tags)
        tags_ = [s for s in tags if not s.startswith("I")]
        tag_add = []
        
        for i in r_f:
            tag_add.append(f"I{dic[i]}")
        tab.modify(tags=tags_ + tag_add)  
        
        for ix in deincr:
            count[ix] -= 1

        if not out_ab:
            # handle scalars produced at the end
            scalars.append(tab)
            continue

        tn |= tab

        if equalize_norms:
            tn.strip_exponent(tab, equalize_norms)

        for ix in out_ab:
            # now we need to check outputs indices again
            queue.add(ix)

    if scalars:
        if equalize_norms:
            signs = []
            for s in scalars:
                signs.append(do("sign", s))
                tn.exponent += do("log10", do("abs", s))
            scalars = signs

        if tn.num_tensors:
            tn *= prod(scalars)
        else:
            # no tensors left! re-add one with all the scalars
            tn |= Tensor(prod(scalars))

    return tn

def extract_gate_pos(circ_):
    
    psi = circ_.uni
    
    place_rzz_ = []
    where_rzz_ = []
    round_info = []
    for t in psi:
        tag_l = list(t.tags)
        gate_tags = [ i for i in tag_l if i.startswith("GATE_")]
        int_ = int(re.search(r'\d+', gate_tags[0]).group())
        if len(t.inds)==4:
            I_tags = [ i for i in tag_l if i.startswith("I")]
            x = int(re.search(r'\d+', I_tags[0]).group())
            y = int(re.search(r'\d+', I_tags[1]).group())
            place_rzz_.append(int_)
            where_rzz_.append((x,y))
            round_tags = [ i for i in tag_l if i.startswith("ROUND_")]
            int_ = int(re.search(r'\d+', round_tags[0]).group())
            round_info.append(int_)

    # print(round_info)
    # Get the maximum value in the list
    max_value = max(round_info)
    
    # Count occurrences of each integer from 0 to max_value
    counts = collections.Counter(round_info)
    
    # Print the counts
    round_ = []
    for i in range(max_value + 1):
        round_.append(counts[i])
    
    sorted_indices = [index for index, _ in sorted(enumerate(place_rzz_), key=lambda x: x[1])]
    where_rzz = [where_rzz_[i] for i in sorted_indices]
    place_rzz = sorted(place_rzz_)
    GATE_rzz = [f"GATE_{i}" for i in place_rzz]
    
    psi_=rank_simplify_leftinds(psi)
    Gate_l =[]
    where_rzz_ = []
    for tag in GATE_rzz:
        G_ = psi_[tag]      #.data.reshape(2,2,2,2)
        tag_l = list(G_.tags)
        I_tags = [ i for i in tag_l if i.startswith("I")]
        x = int(re.search(r'\d+', I_tags[0]).group())
        y = int(re.search(r'\d+', I_tags[1]).group())
        where_rzz_.append((x,y))
        #print(G_.tags)
        #inds = G_.inds
        #left_inds = list(G_.left_inds)
        #right_inds = [ indx for indx in inds if indx not in left_inds] 
        #inds = right_inds + left_inds
        
        #G_.transpose(*inds, inplace=True)
        G_ = G_.data

        
        Gate_l.append(G_)
        # Gdagger = np.conj(G_)
        # result = np.einsum('mnij,mnkp->ijkp', Gdagger, G_)
        # print("----newl-------", result, "----newl-------")
        # result = np.einsum('mnij,mnkp->ijkp', G_, Gdagger)
        # print("----r-------", result, "-----r------")

    if sum(round_) != len(Gate_l):
        print("warning: tqgs get absorbed")
        round_ = [len(Gate_l)]
    #print(where_rzz_, where_rzz)
    return where_rzz_, Gate_l, round_

















def trotter_gates_IBM( site, Lx, Ly, Lz, dt, J, h, dtype = "complex128", cycle = "periodic", 
                      max_bond_mpo = 80, style = "left", cutoff=1.e-12,
                      depth_ = 1
                     ):
    ZZ = qu.pauli('Z', dtype=dtype) & qu.pauli('Z', dtype=dtype)
    X = qu.pauli('X', dtype=dtype) 
    H_ = qu.expm( ZZ * complex(0, -1.) * math.pi/4 ).reshape(2,2,2,2)
    RX = qu.expm(X * complex(0, -1.) * dt/2.) 
    L = Lx * Ly * Lz

    #field
    gate_dic = { (i,): RX  for i in range(Lx * Ly * Lz) }

    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(0, 13, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(18, 32, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(37, 51, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(56, 70, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(75, 89, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(94, 108, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(113, 126, 1) }  

    #red    
    gate_dic = gate_dic | { (1, 2) : H_ } | { (4, 5) : H_ } | { (6, 7) : H_ } | { (8, 9) : H_ } | { (12, 13) : H_ } 
    gate_dic = gate_dic | { (20, 21) : H_ } | { (23, 24) : H_ } | { (25, 26) : H_ } | { (31, 32) : H_ } 
    gate_dic = gate_dic | { (43, 44) : H_ } | { (45, 46) : H_ } | { (47, 48) : H_ } | { (49, 50) : H_ } 
    gate_dic = gate_dic | { (56, 57) : H_ } | { (59, 60) : H_ } | { (61, 62) : H_ } | { (63, 64) : H_ } | { (66, 67) : H_ } 
    gate_dic = gate_dic | { (76, 77) : H_ } | { (78, 79) : H_ } | { (83, 84) : H_ } | { (87, 88) : H_ } 
    gate_dic = gate_dic | { (94, 95) : H_ } | { (96, 97) : H_ } | { (98, 99) : H_ } | { (103, 104) : H_ } | { (105, 106) : H_ }  
    gate_dic = gate_dic | { (107, 108) : H_ } | { (115, 116) : H_ } | { (118, 119) : H_ } | { (120, 121) : H_ } | { (122, 123) : H_ } 
    gate_dic = gate_dic | { (0, 14) : H_ } | { (17, 30) : H_ }| { (15, 22) : H_ } | { (33, 39) : H_ } | { (28, 35) : H_ } | { (36, 51) : H_ } 
    gate_dic = gate_dic | { (37, 52) : H_ } | { (41, 53) : H_ } | { (55, 68) : H_ } | { (58, 71) : H_ } | { (92, 102) : H_ } | { (109, 114) : H_} 
    gate_dic = gate_dic | { (72, 81) : H_ } |  { (73, 85) : H_ } |  { (100, 110) : H_ } |  { (70, 74) : H_ }

    #blue
    gate_dic = gate_dic | { (2, 3) : H_ } | { (5, 6) : H_ } | { (9, 10) : H_ } | { (11, 12) : H_ } | { (19, 20) : H_ } 
    gate_dic = gate_dic | { (21,22) : H_ } | { (26, 27) : H_ } | { (28, 29) : H_ } | { (30, 31) : H_ } | { (38, 39) : H_ } 
    gate_dic = gate_dic | { (40, 41) : H_ } | { (42, 43) : H_ } | { (44, 45) : H_ } | { (48, 49) : H_ } | { (57, 58) : H_ } 
    gate_dic = gate_dic | { (62, 63) : H_ } | { (65, 66) : H_ } | { (67, 68) : H_ } | { (69, 70) : H_ } | { (75, 76) : H_ } 
    gate_dic = gate_dic | { (79, 80) : H_ } | { (81, 82) : H_ } | { (84, 85) : H_ } | { (86, 87) : H_ } | { (99, 100) : H_ } 
    gate_dic = gate_dic | { (101, 102) : H_ } | { (113, 114) : H_ } | { (116, 117) : H_ } | { (121, 122) : H_ } | { (123, 124) : H_ } 
    gate_dic = gate_dic | { (125, 126) : H_ }     
    gate_dic = gate_dic | { (4, 15) : H_ } | { (8, 16) : H_ } | { (14, 18) : H_ } | { (24, 34) : H_ } | { (35, 47) : H_ } 
    gate_dic = gate_dic | { (53, 60) : H_ } | { (54, 64) : H_ } | { (71, 77) : H_ } | { (74, 89) : H_ } | { (83, 92) : H_ } 
    gate_dic = gate_dic | { (90, 94) : H_ }| { (91, 98) : H_ }  | { (93, 106) : H_ } | { (96, 109) : H_ }   
    gate_dic = gate_dic | { (104, 111) : H_ } | { (108, 112) : H_ } | { (110, 118) : H_ } 

                         
    #green
    gate_dic = gate_dic | { (0, 1) : H_ } | { (3, 4) : H_ } | { (7, 8) : H_ } | { (10, 11) : H_ } | { (18, 19) : H_ } 
    gate_dic = gate_dic | { (22, 23) : H_ } | { (24, 25) : H_ } | { (27, 28) : H_ } | { (29, 30) : H_ } | { (37, 38) : H_ } 
    gate_dic = gate_dic | { (39, 40) : H_ } | { (41, 42) : H_ } | { (46, 47) : H_ } | { (50, 51) : H_ } | { (58, 59) : H_ } 
    gate_dic = gate_dic | { (60, 61) : H_ } | { (64, 65) : H_ } | { (68, 69) : H_ } | { (77, 78) : H_ } | { (80, 81) : H_ }| { (82, 83) : H_ } 
    gate_dic = gate_dic | { (85, 86) : H_ } | { (88, 89) : H_ } | { (95, 96) : H_ } | { (97, 98) : H_ } | { (100, 101) : H_ } 
    gate_dic = gate_dic | { (102, 103) : H_ } | { (104, 105) : H_ } | { (106, 107) : H_ } | { (114, 115) : H_ } | { (117, 118) : H_ } 
    gate_dic = gate_dic | { (119, 120) : H_ } | { (124, 125) : H_ } 
    gate_dic = gate_dic | { (12, 17) : H_ } | { (20, 33) : H_ } | { (16, 26) : H_ } | { (34, 43) : H_ } | { (32, 36) : H_ } | { (45, 54) : H_ }
    gate_dic = gate_dic | { (49, 55) : H_ } | { (52, 56) : H_ } | { (62, 72) : H_ } | { (66, 73) : H_ }
    gate_dic = gate_dic | { (75, 90) : H_ } | { (79, 91) : H_ } | { (87, 93) : H_ } | { (111, 122) : H_ } | { (112, 126) : H_ } 


    # where_ = []
    # gate_ = []
    # for _ in range(depth_):
    #     for where in  gate_dic:
    #         where_.append(where)
    #         gate_.append(gate_dic[where])


    mpo_l = []
    count = 0
    for count, where in tqdm(enumerate(gate_dic)):
        mpo = mpo_from_gate(gate_dic[where], where, L,  dtype = dtype, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
        if virtual_bond_max(mpo) == 1 and count > 0:
            mpo_f = mpo.apply(mpo_l[len(mpo_l) - 1])
            mpo_l[len(mpo_l) - 1] = mpo_f 
        else:
            mpo_l.append(mpo)
        count += 1
    #for count, where in enumerate(where_):
    #     mpo = mpo_from_gate(gate_[count], where, L,  dtype = dtype, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
    #     mpo_l.append(mpo)


    circ = qtn.Circuit(N=Lx*Ly*Lz)    
    count = 0
    for _ in range(depth_):
        for where in gate_dic:
            if len(where) == 1:
                x, = where
                circ.apply_gate('RX', dt, x, gate_round=count)
            else:
                x, y = where
                circ.apply_gate('RZZ', -math.pi/4, x, y, gate_round=count)
        count += 1


    mpo_l = mpo_l * depth_
    return mpo_l, circ



def trotter_gates_IBM_lightcone(site,  dt, J, h,  
                                dtype = "complex128", 
                                cycle = "periodic", 
                                max_bond_mpo = 80, 
                                style = "left", 
                                cutoff=1.e-12, 
                                depth_ = 1,
                                triangular = False,
                                cal_mpo = "off",
                                basis=None,
                                alpha_xy = 1.,
                                model = "Ising",

                     ):
    ZZ = qu.pauli('Z', dtype=dtype) & qu.pauli('Z', dtype=dtype)
    X = qu.pauli('X', dtype=dtype) 
    H_ = qu.expm( ZZ * complex(0, -1.) * math.pi/4 ).reshape(2,2,2,2)
    RX = qu.expm(X * complex(0, -1.) * dt/2.) 
    L = 127

    #field
    gate_dic = { (i,): RX  for i in range(L) }

    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(0, 13, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(18, 32, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(37, 51, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(56, 70, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(75, 89, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(94, 108, 1) }  
    # gate_dic = gate_dic | { (i,i+1) : H_ for i in range(113, 126, 1) }  

    #red    
    gate_dic = gate_dic | { (1, 2) : H_ } | { (4, 5) : H_ } | { (6, 7) : H_ } | { (8, 9) : H_ } | { (12, 13) : H_ } 
    gate_dic = gate_dic | { (20, 21) : H_ } | { (23, 24) : H_ } | { (25, 26) : H_ } | { (31, 32) : H_ } 
    gate_dic = gate_dic | { (43, 44) : H_ } | { (45, 46) : H_ } | { (47, 48) : H_ } | { (49, 50) : H_ } 
    gate_dic = gate_dic | { (56, 57) : H_ } | { (59, 60) : H_ } | { (61, 62) : H_ } | { (63, 64) : H_ } | { (66, 67) : H_ } 
    gate_dic = gate_dic | { (76, 77) : H_ } | { (78, 79) : H_ } | { (83, 84) : H_ } | { (87, 88) : H_ } 
    gate_dic = gate_dic | { (94, 95) : H_ } | { (96, 97) : H_ } | { (98, 99) : H_ } | { (103, 104) : H_ } | { (105, 106) : H_ }  
    gate_dic = gate_dic | { (107, 108) : H_ } | { (115, 116) : H_ } | { (118, 119) : H_ } | { (120, 121) : H_ } | { (122, 123) : H_ } 
    gate_dic = gate_dic | { (0, 14) : H_ } | { (17, 30) : H_ }| { (15, 22) : H_ } | { (33, 39) : H_ } | { (28, 35) : H_ } | { (36, 51) : H_ } 
    gate_dic = gate_dic | { (37, 52) : H_ } | { (41, 53) : H_ } | { (55, 68) : H_ } | { (58, 71) : H_ } | { (92, 102) : H_ } | { (109, 114) : H_} 
    gate_dic = gate_dic | { (72, 81) : H_ } |  { (73, 85) : H_ } |  { (100, 110) : H_ } |  { (70, 74) : H_ }

    #blue
    gate_dic = gate_dic | { (2, 3) : H_ } | { (5, 6) : H_ } | { (9, 10) : H_ } | { (11, 12) : H_ } | { (19, 20) : H_ } 
    gate_dic = gate_dic | { (21,22) : H_ } | { (26, 27) : H_ } | { (28, 29) : H_ } | { (30, 31) : H_ } | { (38, 39) : H_ } 
    gate_dic = gate_dic | { (40, 41) : H_ } | { (42, 43) : H_ } | { (44, 45) : H_ } | { (48, 49) : H_ } | { (57, 58) : H_ } 
    gate_dic = gate_dic | { (62, 63) : H_ } | { (65, 66) : H_ } | { (67, 68) : H_ } | { (69, 70) : H_ } | { (75, 76) : H_ } 
    gate_dic = gate_dic | { (79, 80) : H_ } | { (81, 82) : H_ } | { (84, 85) : H_ } | { (86, 87) : H_ } | { (99, 100) : H_ } 
    gate_dic = gate_dic | { (101, 102) : H_ } | { (113, 114) : H_ } | { (116, 117) : H_ } | { (121, 122) : H_ } | { (123, 124) : H_ } 
    gate_dic = gate_dic | { (125, 126) : H_ }     
    gate_dic = gate_dic | { (4, 15) : H_ } | { (8, 16) : H_ } | { (14, 18) : H_ } | { (24, 34) : H_ } | { (35, 47) : H_ } 
    gate_dic = gate_dic | { (53, 60) : H_ } | { (54, 64) : H_ } | { (71, 77) : H_ } | { (74, 89) : H_ } | { (83, 92) : H_ } 
    gate_dic = gate_dic | { (90, 94) : H_ }| { (91, 98) : H_ }  | { (93, 106) : H_ } | { (96, 109) : H_ }   
    gate_dic = gate_dic | { (104, 111) : H_ } | { (108, 112) : H_ } | { (110, 118) : H_ } 

                         
    #green
    gate_dic = gate_dic | { (0, 1) : H_ } | { (3, 4) : H_ } | { (7, 8) : H_ } | { (10, 11) : H_ } | { (18, 19) : H_ } 
    gate_dic = gate_dic | { (22, 23) : H_ } | { (24, 25) : H_ } | { (27, 28) : H_ } | { (29, 30) : H_ } | { (37, 38) : H_ } 
    gate_dic = gate_dic | { (39, 40) : H_ } | { (41, 42) : H_ } | { (46, 47) : H_ } | { (50, 51) : H_ } | { (58, 59) : H_ } 
    gate_dic = gate_dic | { (60, 61) : H_ } | { (64, 65) : H_ } | { (68, 69) : H_ } | { (77, 78) : H_ } | { (80, 81) : H_ }| { (82, 83) : H_ } 
    gate_dic = gate_dic | { (85, 86) : H_ } | { (88, 89) : H_ } | { (95, 96) : H_ } | { (97, 98) : H_ } | { (100, 101) : H_ } 
    gate_dic = gate_dic | { (102, 103) : H_ } | { (104, 105) : H_ } | { (106, 107) : H_ } | { (114, 115) : H_ } | { (117, 118) : H_ } 
    gate_dic = gate_dic | { (119, 120) : H_ } | { (124, 125) : H_ } 
    gate_dic = gate_dic | { (12, 17) : H_ } | { (20, 33) : H_ } | { (16, 26) : H_ } | { (34, 43) : H_ } | { (32, 36) : H_ } | { (45, 54) : H_ }
    gate_dic = gate_dic | { (49, 55) : H_ } | { (52, 56) : H_ } | { (62, 72) : H_ } | { (66, 73) : H_ }
    gate_dic = gate_dic | { (75, 90) : H_ } | { (79, 91) : H_ } | { (87, 93) : H_ } | { (111, 122) : H_ } | { (112, 126) : H_ } 


    circ = qtn.Circuit(N=127)  

    count = 0
    for _ in range(depth_):
        for where in gate_dic:
            if len(where) == 1:
                x, = where
                circ.apply_gate('RX', dt, x, gate_round=count)
            else:
                x, y = where
                circ.apply_gate('RZZ', -math.pi/2, x, y, gate_round=count)
        count += 1


    lc_tags = list(circ.get_reverse_lightcone_tags(where=site))
    lc_tags.remove("PSI0")
    tags_l = []
    psi = circ.psi

    for i in lc_tags:
        if isinstance(psi[i], tuple):
            t_ = list(psi[i])
            tag_1 = list(t_[0].tags)
            tag_2 = list(t_[1].tags)
            tags = [ x for x in tag_1 if x.startswith('I')] + [ x for x in tag_2 if x.startswith('I')]
            tags_l.append(tags)
        else:
            tag = list(psi[i].tags )
            tags = [ x for x in tag if x.startswith('I')]
            tags_l.append(tags) 
        
                
    where_lightcone = []
    gate_lightcone = []
    for tag in tags_l:
        if len(tag) == 1:
            int_ = int(re.search(r'\d+', tag[0]).group())
            where_lightcone.append((int_,))
            gate_lightcone.append(RX)
        if len(tag) == 2:
            int_1 = int(re.search(r'\d+', tag[0]).group())
            int_2 = int(re.search(r'\d+', tag[1]).group())
            where_lightcone.append((int_1,int_2))
            gate_lightcone.append(H_)


    mpo_l_lightcone = []


    # #print(where_lightcone)
    # # count = 0
    # for count, where in tqdm(enumerate(where_lightcone)):
    #     #print(where)
    #     mpo = mpo_from_gate(gate_lightcone[count], where, L,  dtype = dtype, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
    #     #mpo.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )
    #     if virtual_bond_max(mpo) == 1 and count > 0:
    #         mpo_f = mpo_l_lightcone[len(mpo_l_lightcone) - 1].apply(mpo)
    #         mpo_f.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )
    #         mpo_l_lightcone[len(mpo_l_lightcone) - 1] = mpo_f 
    #     else:
    #         mpo_l_lightcone.append(mpo)
    #     count += 1
    

    # for count, where in enumerate(where_lightcone):
    #     mpo = mpo_from_gate(gate_lightcone[count], where, L,  dtype = dtype, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
    #     mpo_l_lightcone.append(mpo)


    return gate_lightcone, where_lightcone, circ











def internal_inds_TN2d(psi):
    open_inds = [ f"k{i},{j}"  for i,j in itertools.product(range(psi.Lx), range(psi.Ly))] + [ f"b{i},{j}"  for i,j in itertools.product(range(psi.Lx), range(psi.Ly))] 
    innre_inds = []
    for t in psi:
        t_list = list(t.inds)
        for j in t_list :
            if j not in open_inds:
                innre_inds.append(j)
    return innre_inds

def inds_new_TN2d(t):
    t_h = t.copy()
    bond_inds = internal_inds_TN2d(t_h)
    t_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds} )
    return t_h


def herm_inds_new_TN2d(t):
    t = t.copy()
    t_h = t.H
    t_h.reindex_( {f"k{i},{j}":f"l{i},{j}" for i,j in itertools.product(range(t.Lx), range(t.Ly))})
    t_h.reindex_( {f"b{i},{j}":f"k{i},{j}" for i,j in itertools.product(range(t.Lx), range(t.Ly))})
    t_h.reindex_( {f"l{i},{j}":f"b{i},{j}" for i,j in itertools.product(range(t.Lx), range(t.Ly))})
    bond_inds = internal_inds_TN2d(t_h)
    t_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    return t_h


def internal_inds_mpo(psi):
    open_inds = [ f"k{i}"  for i in range(psi.L)] + [ f"b{i}"  for i in range(psi.L)] 
    innre_inds = []
    for t in psi:
        t_list = list(t.inds)
        for j in t_list :
            if j not in open_inds:
                innre_inds.append(j)
    return innre_inds

def herm_inds_new(t):
    t_h = t.H
    t_h.reindex_( {f"k{i}":f"l{i}" for i in range(t.L)}  )
    t_h.reindex_( {f"b{i}":f"k{i}" for i in range(t.L)}  )
    t_h.reindex_( {f"l{i}":f"b{i}" for i in range(t.L)}  )
    bond_inds = internal_inds_mpo(t_h)
    t_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    return t_h

def inds_new_mpolist(mpo_list):
    mpo_l = []
    for t in mpo_list:
        t_ = t.copy()
        bond_inds = internal_inds_mpo(t_)
        t_.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
        mpo_l.append(t_)
    return mpo_l

def virtual_bond_max2d(tn):
    l_bnd = []
    for i in range(tn.Lx-1):
        for j in range(tn.Ly):
            l_bnd.append(tn.bond_size((i,j), (i+1,j)))
    for i in range(tn.Lx):
        for j in range(tn.Ly-1):
            l_bnd.append(tn.bond_size((i,j), (i,j+1)))
    return max(l_bnd)

def virtual_bond_max(tn):
    l_bnd = []
    for coor in range(tn.L-1):
        l_bnd.append(tn.bond_size(coor, coor+1))
    return max(l_bnd)

def compress_mpo_list(mpo_l, size_=2, coor=[],  dtype = "complex128", cycle = "periodic", max_bond_mpo=120, style = "left", cutoff=1.e-12):
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    mpo_l_ = chunks(mpo_l, size_)
    mpo_f = []
    for count, mpo_local in tqdm(enumerate(mpo_l_)):
        for count_, t in enumerate(mpo_local):
            if count_ == 0:
                t_ = t
            else:
                t_ = t.apply(t_)
                t_.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )
        mpo_f.append(t_)

    if coor:
        for count, corr_ in enumerate(coor):
            x, y = corr_
            index_set = []
            for count_ in range(x, y+1, 1):
                if count_ == x:
                    t_ = mpo_f[count_]
                else:
                    t_ = mpo_f[count_].apply(t_)
                    t_.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )
                    index_set.append(count_)
            mpo_f[x] = t_
        mpo_f = [x for i, x in enumerate(mpo_f) if i not in index_set]                

    mpo_max_bnd = [i.max_bond() for i in mpo_f]
    return mpo_f, max(mpo_max_bnd)




def compress_mpo_list_operator(mpo_l, size_=2, coor=[],  dtype = "complex128", cycle = "periodic", max_bond_mpo=120, style = "left", cutoff=1.e-12):
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    mpo_l_ = chunks(mpo_l, size_)
    mpo_f = []
    for count, mpo_local in enumerate(mpo_l_):
        for count_, t in enumerate(mpo_local):
            if count_ == 0:
                t_ = t
            else:
                t_ = t_.apply(t)
                t_.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )
        mpo_f.append(t_)

    if coor:
        for count, corr_ in enumerate(coor):
            x, y = corr_
            index_set = []
            for count_ in range(x, y+1, 1):
                if count_ == x:
                    t_ = mpo_f[count_]
                else:
                    t_ = t_.apply(mpo_f[count_])
                    t_.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )
                    index_set.append(count_)
            mpo_f[x] = t_
        mpo_f = [x for i, x in enumerate(mpo_f) if i not in index_set]                

    mpo_max_bnd = [i.max_bond() for i in mpo_f]
    return mpo_f, max(mpo_max_bnd)




def mpo_from_gate(gate_, where, L,  dtype ="complex128", cutoff=1e-12, max_bond_mpo=50, style= "right"):
    swap = qu.swap(dim=2, dtype = dtype)
    swap = get_swap(
        2, dtype=autoray.get_dtype_name(gate_[0]), backend=autoray.infer_backend(gate_[0])
    )
    swap = swap.reshape(2,2,2,2)

    

    mpo_l = []
    if len(where) == 1:
        t_ = mpo_from_gate_({where:gate_}, L, dtype=dtype, cutoff=cutoff) 
    if len(where) == 2:
        start, end = where
        
        count = 0
        if start > end:
            end, start = where
            count += 1
        for i in range(start, end-1):
            mpo_l.append( mpo_from_gate_({(i, i+1):swap}, L, dtype=dtype, cutoff=cutoff) )
        if count == 0:
            mpo_l.append( mpo_from_gate_({(end-1, end):gate_}, L, dtype=dtype, cutoff=cutoff) )
        else:
            mpo_l.append( mpo_from_gate_({(end, end-1):gate_}, L, dtype=dtype, cutoff=cutoff) )

        for i in reversed(range(start+1, end)):
            mpo_l.append( mpo_from_gate_({(i-1, i):swap}, L, dtype=dtype, cutoff=cutoff) )
        t_ = 0
        for count, t in enumerate(mpo_l):
            if count == 0:
                t_ = t
            else:
                t_ = t_.apply(t)
                #t_.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )
    t_.compress( style, max_bond=max_bond_mpo, cutoff=cutoff )

    return t_ 


def mpo_from_gate_(gate_dic, L, dtype="complex128",cutoff=1.0e-20 ):
    
    MPO_I=qtn.MPO_identity(L, phys_dim=2, dtype=dtype)
    
    position_x, poxition_y = 0, 0
    
    for where in gate_dic:
        if len(where) == 2:
            position_x, position_y = where
            U_ = gate_dic[where]
        else:
            position_x,  = where
            U_ = gate_dic[where]
            if position_x == 0 or position_x == L-1:
                W = np.zeros([1, 2, 2], dtype=dtype)
                W[0,:,:] = U_
                MPO_I[position_x].modify(data=W)
            else:
                W = np.zeros([1, 1, 2, 2], dtype=dtype)
                W[0, 0, :, :] = U_
                MPO_I[position_x].modify(data=W)
            return MPO_I
                
    if position_x < 0 or position_y <0:
        print("negetive positoins")
    if position_x == position_y:
        print("U acting on the same positoins")
    elif  abs(position_x - position_y) > 1:
        print("U acting on the long-rnage")
        
        
    T = qtn.Tensor(data=U_, inds=("b0","b1","k0","k1" ), tags=[])

    T_l, T_r = qtn.tensor_split(T, ["b0","k0"], get = "tensors", cutoff=cutoff, bond_ind="x")
    T_l=T_l.transpose("x", "b0", "k0")
    T_r=T_r.transpose("x", "b1", "k1")
    bnd_ = T_l.ind_size("x")
    if position_x < position_y:
        if position_x == 0:
            W = np.zeros([bnd_, 2, 2], dtype=dtype)
            W[:,:,:]=T_l.data
            MPO_I[position_x].modify(data=W)    
        else:
            W = np.zeros([1, bnd_, 2, 2], dtype=dtype)
            W[ 0,:,:,:]=T_l.data
            MPO_I[position_x].modify(data=W)    

        if position_y == L-1:
            W = np.zeros([ bnd_, 2, 2], dtype=dtype)
            W[ :,:,:]=T_r.data
            MPO_I[position_y].modify(data=W)
            #MPO_I.show()    
        else:
            W = np.zeros([ bnd_, 1, 2, 2], dtype=dtype)
            W[ :,0,:,:]=T_r.data
            MPO_I[position_y].modify(data=W)
            #MPO_I.show()
    else:
        if position_x == L-1:
            W = np.zeros([bnd_, 2, 2], dtype=dtype)
            W[:,:,:]=T_l.data
            MPO_I[position_x].modify(data=W)    
        else:
            W = np.zeros([bnd_, 1, 2, 2], dtype=dtype)
            W[ :,0,:,:]=T_l.data
            MPO_I[position_x].modify(data=W)    

        if position_y == 0:
            W = np.zeros([ bnd_, 2, 2], dtype=dtype)
            W[ :,:,:]=T_r.data
            MPO_I[position_y].modify(data=W)
            #MPO_I.show()    
        else:
            W = np.zeros([ 1,bnd_, 2, 2], dtype=dtype)
            W[ 0,:,:,:]=T_r.data
            MPO_I[position_y].modify(data=W)
            #MPO_I.show()
            
    return MPO_I



def split_dictionary(input_dict, chunk_size):
    res = []
    new_dict = {}
    for k, v in input_dict.items():
        if len(new_dict) < chunk_size:
            new_dict[k] = v
        else:
            res.append(new_dict)
            new_dict = {k: v}
    res.append(new_dict)
    return res


def gen_split_dictionary(input_dict, chunk_size):
    new_dict = {}
    for k, v in input_dict.items():
        if len(new_dict) < chunk_size:
            new_dict[k] = v
        else:
            yield new_dict
            new_dict = {k: v}
    yield


def dis_mps(psi, psi_fix, opt, cost_f = "fidelity"): 
    val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
    val_1 = (psi.H & psi_fix).contract(all, optimize=opt)
    val_2 = autoray.do("conj", val_1)
    val_3 = autoray.do("sqrt", val_0)
    if cost_f == "fidelity":  
        return 1- ( abs(val_1)/abs(val_3) )
    elif cost_f == "distance":
        return abs(1.0 + val_0 -  val_1 - val_2)


def fidel_mps(psi, psi_fix):
    to_backend, opt_, opt = req_backend(progbar=False)

    val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))
    val_1 = val_1 ** 2
    return  complex(val_1 / (val_0 * val_) )


def fidel_mps_normalized(psi, psi_fix, opt,  cur_orthog=None):
    tn = psi.H & psi_fix
    val_1 = abs(tn.contract(all, optimize=opt))
    return  val_1 ** 2



def internal_inds(psi):
    open_inds = psi.outer_inds()
    innre_inds = []
    for t in psi:
        t_list = list(t.inds)
        for j in t_list :
            if j not in open_inds:
                innre_inds.append(j)
    return innre_inds


def mpo_auto(psi, psi_fix, opt, optimize=True, n_iter=100, cost_f = "distance", disp = "off", progbar=False, threshold = 1.e-8):
     
    val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt) )
    
    def loss_(psi, psi_fix, opt, cost_f, val_):
        #val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt) ) 
        val_0 = (psi.H & psi).contract(all, optimize=opt) 
        val_1 = (psi.H & psi_fix).contract(all, optimize=opt)
        if cost_f == "fidelity":
            val_0 = autoray.do("abs", val_0)
            val_3 = autoray.do("sqrt", val_0)
            
            val_4 = autoray.do("sqrt", val_)
        
            val_1 = autoray.do("abs", val_1)
            return 1 - ( val_1/( val_3 * val_4 ) )
        elif cost_f == "logfidelity":
            val_0 = autoray.do("abs", val_0)
            val_0 = autoray.do("log", val_0)
            
            val_4 = autoray.do("log", val_)
            
            val_1 = autoray.do("abs", val_1)
            val_1 = autoray.do("log", val_1)
            return -val_1 + (val_0 + val_4) * 0.5

        elif cost_f == "distance":
            val_0 = autoray.do("abs", val_0)
            val_2 = autoray.do("conj", val_1)
            return abs(val_ + val_0 -  val_1 - val_2)


    loss_val = abs(loss_(psi, psi_fix, opt, cost_f, val_))
    if loss_val < 1.e-8:
        return psi, loss_val

    if disp == "on":
        print("init_loss", loss_val)


    tnopt = qtn.TNOptimizer(
        psi,                        # the tensor network we want to optimize
        loss_,
        loss_constants = {'psi_fix': psi_fix},# the function we want to minimize
        loss_kwargs = {'opt': opt, 'cost_f': cost_f, "val_": val_},
        autodiff_backend = "autograd",  # 'autograd',   # use 'autograd' for non-compiled optimization
        optimizer='L-BFGS-B',     # the optimization algorithm
        progbar=progbar,
    )
    if optimize:
        psi_f = tnopt.optimize(n=n_iter, ftol=threshold, maxfun= 10e+10, gtol= 1e-14, eps=1.e-10, maxls=500, iprint = 0, disp=False)
        loss_val = abs(loss_(psi_f, psi_fix, opt, cost_f, val_))
        return psi_f, loss_val
        if disp == "on":
            print("init_final", loss_val)
    else:
        return psi, loss_val








def mps_auto(psi, psi_fix, opt, n_iter=100, cost_f = "distance", disp = "off", progbar=False, threshold = 1.e-8):
     
    def loss_(psi, psi_fix, opt, cost_f): 
        val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
        val_1 = (psi.H & psi_fix).contract(all, optimize=opt)
        val_2 = autoray.do("conj", val_1)
        val_3 = autoray.do("sqrt", val_0)
        if cost_f == "fidelity":  
            return 1. - ( abs(val_1)/abs(val_3) )
        elif cost_f == "distance":
            return abs(1.0 + val_0 -  val_1 - val_2)
    
    if disp == "on":
        print("init_loss", loss_(psi, psi_fix, opt, cost_f))

    tnopt = qtn.TNOptimizer(
        psi,                        # the tensor network we want to optimize
        loss_,
        loss_constants={'psi_fix': psi_fix},# the function we want to minimize
        loss_kwargs={'opt': opt, 'cost_f': cost_f},
        autodiff_backend='autograd',   # use 'autograd' for non-compiled optimization
        optimizer='L-BFGS-B',     # the optimization algorithm
        progbar=progbar,
    )
    psi_f = tnopt.optimize(n=n_iter, ftol=threshold, maxfun= 10e+9, gtol= 1e-12, eps=1.49016e-08, maxls=400, iprint = 0, disp=False)
    
    if disp == "on":
        print("init_final", loss_(psi_f, psi_fix, opt, cost_f))
    psi_f.normalize()
    return psi_f

def mps_dmrg(psi, psi_fix, opt, n_iter=15, cost_f = "fidelity", threshold = 1.e-8):
    
    bond_inds = internal_inds(psi_fix)
    psi_fix.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    E_0 = 1.e+1
    E_1 = 1.e+2
    
    with tqdm(total=n_iter,  desc="dmrg:",  
              leave=False, position=1, 
              colour='red') as pbar:

        for count in range(n_iter):
            
            for site_ in range(psi.L):

                if site_ == 0: 
                    arg = 'calc'
                else: 
                    arg = site_ - 1

                psi.canonize(site_, 
                            cur_orthog=arg, #'calc', 
                            bra=None)
                psi_h=psi.H
                psi_h=psi_h.select( [f"I{j}" for j in range(psi.L) if j not in [site_]], 'any')                        
                f = (psi_h & psi_fix).contract(all, optimize=opt)
                f=f.transpose(*psi[site_].inds)
                norm_f = (f.H @ f)
                f = f * (norm_f**-0.5)        
                psi[site_].modify(data = f.data)
    
            E_0 = E_1 * 1.0
            E_1 = abs(1 - norm_f**0.5)
            
            #print("\n", count, abs(1 - fidel_mps_normalized(psi, psi_fix, opt)), E_1, "\n")

            if abs(E_1) > 1.e-12:
                delta = abs(E_0 - E_1) / max([abs(E_1),abs(E_0)]) 
                if count > 0:
                    if delta < threshold:
                        break
            else:
                break
            
            pbar.set_postfix({'Fidel': E_1, "delta": delta})
            pbar.update(1)



    pbar.close()    
    #e_f = fidel_mps_normalized(psi, psi_fix, opt)
    e_f = norm_f ** 0.5

    return psi, e_f


#@profile
def bmps_dmrg(psi, psi_fix, opt, inds_fuse, n_iter=15, 
              cost_f = "fidelity", threshold = 1.e-8,
              n_tqgate = 2,
              prgbar = False
              ):
    
    from time import sleep
    bond_inds = internal_inds(psi_fix)
    psi_fix.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    L_mps = psi.L
    psi = mps_to_btn(psi, inds_fuse, inplace=True)

    E_0 = 1.e+1
    E_1 = 1.e+2
    Cost_l = [0]
    W_l = [0]
    norm_f = 1
    with tqdm(total=n_iter,  desc="dmrg:",  
              leave=False, position=1, 
              colour='red', disable=not prgbar) as pbar:

        for count in range(n_iter):
            for site_ in range(psi.L):

                if site_ == 0: 
                    arg = 'calc'
                else: 
                    arg = site_ - 1

                psi.canonize(site_, 
                            cur_orthog=arg, #'calc', 
                            bra=None)
                psi_h=psi.H
                psi_h=psi_h.select( [f"I{j}" for j in range(psi.L) if j not in [site_]], 'any')                        
                
                tn = (psi_h & psi_fix)
                tree = tn.contraction_tree(opt)
                Cost_l.append(tree.contraction_cost())
                W_l.append(tree.contraction_width())
                
                f = tn.contract(all, optimize=tree)
                f=f.transpose(*psi[site_].inds)
                norm_f = (f.H & f).contract(all)
                f = f * (norm_f**-0.5)        
                psi[site_].modify(data = f.data)

            E_0 = complex(E_1 * 1.0)
            E_1 = abs(complex(1 - norm_f))
            
            # print("\n", count, abs(1 - fidel_mps_normalized(psi, psi_fix, opt)), E_1, "\n")

            if abs(E_1) > 1.e-12:
                delta = abs(E_0 - E_1) / max([abs(E_1),abs(E_0)]) 
                if count > 0:
                    if delta < threshold:
                        break
            else:
                break
            
            pbar.set_postfix({'infidelity': E_1, "delta": delta, "TQG":n_tqgate})
            pbar.update(1)



    pbar.close()
    #e_f = fidel_mps_normalized(psi, psi_fix, opt)
    e_f = norm_f

    psi, inds_fuse = btn_to_mps(psi, L_mps, inplace=True)
    return psi, e_f, inds_fuse, (max(Cost_l), max(W_l))


#@profile
def bmps_dmrg_eff(psi, psi_fix, opt, inds_fuse, cur_orthog, n_iter=15, 
                  cost_f = "fidelity", threshold = 1.e-8,
                  prgbar = False,
                  n_tqgate = 2,
                  ):
    
    from time import sleep
    
    L_mps = psi.L
    psi = mps_to_btn(psi, inds_fuse, inplace=True)

    E_0 = 1.e+1
    E_1 = 1.e+2
    
    start, stop = cur_orthog 

    # if start > 0:
    #     start = start - 1

    # if stop < L_mps - 1:
    #     stop = stop + 1
    Cost_l = [0]
    W_l = [0]
    norm_f = 1
    with tqdm(total=n_iter,  desc="dmrg:",  
              leave=False, position=1, 
              colour='red', disable=not prgbar) as pbar:

        for count in range(n_iter):
            for i in range(stop, start, -1):
                psi.right_canonize_site(i, bra=None)
            
            #bond_inds = internal_inds(psi_fix)
            #psi_fix.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )

            for site_ in range(start, stop+1):
                psi_h = psi.H
#                psi_h = psi_h.select( [f"I{j}" for j in  range(psi.L) if j not in [site_]], 'any')                        
                psi_h = psi_h.select( [f"I{j}" for j in  range(start, stop + 1) if j not in [site_]], 'any')                        
                psi_fix_cut = psi_fix.select( ["G"]+[f"I{j}" for j in  range(start, stop + 1)], 'any')                        
                #print(site_, psi_h,psi_fix_cut)

                bond_inds = internal_inds(psi_fix_cut)
                psi_fix_cut.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
                
                tn = (psi_h & psi_fix_cut)
                tree = tn.contraction_tree(opt)
                Cost_l.append(tree.contraction_cost())
                W_l.append(tree.contraction_width())
                f = tn.contract(all, optimize=tree)

                f=f.transpose(*psi[site_].inds)
                norm_f = abs((f.H & f).contract(all))
                f = f * (norm_f**-0.5)
                psi[site_].modify(data = f.data)
                if site_ < stop:
                    psi.left_canonize_site(site_, bra=None)

            
            
            
            E_0 = abs(complex(E_1 * 1.0))
            E_1 = abs(complex(1 - norm_f))
            
            #print("\n", count, abs(1 - fidel_mps(psi, psi_fix, opt)), E_1, "\n")
            #print("\n", count, fidel_mps(psi, psi_fix, opt), norm_f, "\n")
    
            if abs(E_1) > 1.e-12:
                delta = abs(E_0 - E_1) / max([abs(E_1),abs(E_0)]) 
                if count > 0:
                    if delta < threshold:
                        break
            else:
                break
            
            pbar.set_postfix({'infidelity': E_1, "delta": delta, "TQG":n_tqgate})
            pbar.update(1)



    pbar.close()
    #e_f = fidel_mps_normalized(psi, psi_fix, opt)
    #print(e_f, norm_f ** 0.5 )
    
    e_f = norm_f 
    psi, inds_fuse = btn_to_mps(psi, L_mps, inplace=True)
    return psi, e_f, inds_fuse, (max(Cost_l), max(W_l))




#@profile
def bmpo_dmrg_eff(psi, psi_fix, opt, inds_fuse, cur_orthog, n_iter=15, 
                  cost_f = "fidelity", threshold = 1.e-8,
                  prgbar = False,
                  ):
    
    L_mps = psi.L
    psi = mpo_to_btn(psi, inds_fuse, inplace=True)

    E_0 = 1.e+1
    E_1 = 1.e+2
    
    start, stop = cur_orthog 

    with tqdm(total=n_iter,  desc="dmrg:",  
              leave=False, position=1, 
              colour='red', disable=not prgbar) as pbar:

        for count in range(n_iter):
            for i in range(stop, start, -1):
                psi.right_canonize_site(i, bra=None)
            

            for site_ in range(start, stop+1):
                psi_h = psi.H                   
                psi_h = psi_h.select( [f"I{j}" for j in  range(start, stop + 1) if j not in [site_]], 'any')                        
                psi_fix_cut = psi_fix.select( ["G"]+[f"I{j}" for j in  range(start, stop + 1)], 'any')                        
                
                bond_inds = internal_inds(psi_fix_cut)
                psi_fix_cut.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
                f = (psi_h & psi_fix_cut).contract(all, optimize=opt)
                f=f.transpose(*psi[site_].inds)
                norm_f = abs((f.H & f).contract(all))
                #f = f * (norm_f**-0.5)
                psi[site_].modify(data = f.data)
                if site_ < stop:
                    psi.left_canonize_site(site_, bra=None)

            
            
            
            E_0 = E_1 * 1.0
            E_1 = abs(1 - abs(norm_f) )
            
            #print("\n", count, abs(1 - fidel_mps_normalized(psi, psi_fix, opt)), E_1, "\n")

            if abs(E_1) > 1.e-12:
                delta = abs(E_0 - E_1) / max([abs(E_1),abs(E_0)]) 
                if count > 0:
                    if delta < threshold:
                        break
            else:
                break
            
            pbar.set_postfix({'Fidel': E_1, "delta": delta})
            pbar.update(1)



    pbar.close()
    #e_f = fidel_mps_normalized(psi, psi_fix, opt)
    #print(e_f, norm_f ** 0.5 )

    e_f = norm_f ** 0.5
    psi, inds_fuse = btn_to_mpo(psi, L_mps, inplace=True)
    return psi, e_f, inds_fuse






#@profile
def bmpo_dmrg(psi, psi_fix, opt, inds_fuse, n_iter=15, 
              cost_f = "fidelity", threshold = 1.e-8,
              progbar = False,
              ):
    
    bond_inds = internal_inds(psi_fix)
    psi_fix.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    L_mps = psi.L
    psi = mpo_to_btn(psi, inds_fuse, inplace=True)

    E_0 = 1.e+1
    E_1 = 1.e+2
    


    with tqdm(total=n_iter,  
              desc="dmrg:",  
              leave=False, 
              position=1, 
              colour='red') as pbar:

        for count in range(n_iter):
            for site_ in range(psi.L):

                if site_ == 0: 
                    arg = 'calc'
                else: 
                    arg = site_ - 1

                psi.canonize(site_, 
                            cur_orthog=arg, #'calc', 
                            bra=None)
                psi_h=psi.H
                psi_h=psi_h.select( [f"I{j}" for j in range(psi.L) if j not in [site_]], 'any')                        
                f = (psi_h & psi_fix).contract(all, optimize=opt)
                f = f.transpose(*psi[site_].inds)
                norm_f = (f.H & f).contract(all)
                #f = f * (norm_f**-0.5)        
                psi[site_].modify(data = f.data)
            
            E_0 = E_1 * 1.0
            E_1 = abs(1 - abs(norm_f) )
            
            #print("\n", count, abs(1 - fidel_mps_normalized(psi, psi_fix, opt)), E_1, "\n")

            if abs(E_1) > 1.e-12:
                delta = abs(E_0 - E_1) / max([abs(E_1),abs(E_0)]) 
                if count > 0:
                    if delta < threshold:
                        break
            else:
                break
            
            pbar.set_postfix({'Fidel': E_1, "delta": delta})
            pbar.update(1)
               

    pbar.close()
    e_f = 1   # abs(norm_f)
    #e_f = fidel_mps(psi, psi_fix, opt)
    psi, inds_fuse = btn_to_mpo(psi, L_mps, inplace=True)
    return psi, e_f, inds_fuse




def swap_method_(psi, central, gate_dic, bond_dim,opt,tags, error_, dtype = "complex128", cutoff=1e-12, verbosity=0):


    swap = qu.swap(dim=2, dtype = dtype)
    swap = swap.reshape(2,2,2,2)
    position = []



    for where in gate_dic:
        if len(where) == 1:
            position.append(*where)
            psi.gate_(gate_dic[where], where, tags=tags, contract=True, max_bond=None)
        elif len(where) == 2:
            U = gate_dic[where]
            start, end = where
            position.append(start)
            position.append(end)
            if start > end:
                U = np.transpose(U, (1,0,3,2))
                end, start = where
            for i in range(start, end-1):
                psi.gate_(swap, (i, i+1), tags=tags, contract='reduce-split', max_bond=None)
            psi.gate_(U, (end-1, end), tags=tags, contract='reduce-split', max_bond=None)
            for i in reversed(range(start+1, end)):
                psi.gate_(swap, (i-1, i), tags=tags, contract='reduce-split', max_bond=None)

    max_ = max(position)
    min_ = min(position)

    psi_fix = psi * 1.

    psi.canonize(min_, cur_orthog=central)
    psi.left_compress(start=min_, stop=max_, max_bond=bond_dim, cutoff=cutoff)
    central = max_
    
    # if len(where) == 2:
    #     error_.append(abs(fidel_mps(psi, psi_fix, opt)))   
        
 
    #psi.compress("right", max_bond=bond_dim, cutoff=cutoff)
    return psi, error_, central




def apply_swap(bpsi, gate_, where_, contract='split', tags=[], 
               cutoff = 1.0e-12,):
    swap = qu.swap(dim=2)
    swap = swap.reshape(2,2,2,2)

    swap = get_swap(
        2, dtype=autoray.get_dtype_name(gate_[0]), backend=autoray.infer_backend(gate_[0])
    )
    # swap_l = gate_to_backend([[swap]], to_backend)
    # swap = swap_l[0][0]
    
    
    for count_, G in enumerate(gate_):
        where = where_[count_]
        
        if len(where) == 2:
            start, end = where
            count = 0
            if start > end:
                end, start = where
                count += 1

            for i in range(start, end-1):
                try:
                    qtn.tensor_network_gate_inds(bpsi, 
                                             swap, 
                                             [f"k{i}", f"k{i+1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff}
                                            )
                except:
                    qtn.tensor_network_gate_inds(bpsi, 
                                             swap, 
                                             [f"k{i}", f"k{i+1}"], 
                                            contract="split", 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            if count == 0:
                try:
                    qtn.tensor_network_gate_inds(bpsi, G, 
                                             [f"k{end-1}", f"k{end}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )
                except:
                    qtn.tensor_network_gate_inds(bpsi, G, 
                                             [f"k{end-1}", f"k{end}"], 
                                            contract="split", 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            else:


                try:
                    qtn.tensor_network_gate_inds(bpsi, G, 
                                             [f"k{end}", f"k{end-1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

                except:
                   qtn.tensor_network_gate_inds(bpsi, G, 
                                             [f"k{end}", f"k{end-1}"], 
                                            contract="split", 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            for i in reversed(range(start+1, end)):
                try:
                    qtn.tensor_network_gate_inds(bpsi, swap, 
                                            [f"k{i-1}", f"k{i}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )
                except:
                    qtn.tensor_network_gate_inds(bpsi, swap, 
                                            [f"k{i-1}", f"k{i}"], 
                                            contract="split", 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

        if len(where) == 1:
            x, = where
            qtn.tensor_network_gate_inds(bpsi, G, [f"k{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True,
                                        **{"cutoff":cutoff}
                                     )






def swap_method(psi, gate_dic, bond_dim,opt,tags, dtype = "complex128", cutoff=1e-12, verbosity=0):
    swap = qu.swap(dim=2, dtype = dtype)
    swap = swap.reshape(2,2,2,2)
    error_ = []
    for where in tqdm(gate_dic):
        psi_fix = psi.gate(gate_dic[where], where, tags=tags)
        if len(where) == 1:
            #print(where)
            psi.gate_(gate_dic[where], where, tags=tags, contract=True, max_bond=None)
        elif len(where) == 2:
            start, end = where
            count = 0
            if start > end:
                end, start = where
                count += 1
            for i in range(start, end-1):
                #print(i, i+1)
                psi.gate_(swap, (i, i+1), tags=tags, contract='split', max_bond=None)
            if count == 0:
                #print("G", end-1, end)
                psi.gate_(gate_dic[where], (end-1, end), tags=tags, contract='split', max_bond=None)
            else:
                #print("G", end, end -1 )
                psi.gate_(gate_dic[where], (end, end-1), tags=tags, contract='split', max_bond=None)

            for i in reversed(range(start+1, end)):
                #print(i, (i-1, i))
                psi.gate_(swap, (i-1, i), tags=tags, contract='split', max_bond=None)

        psi.compress("left", max_bond=bond_dim, cutoff=cutoff)
        if verbosity == 1:
            error_.append( abs(dis_mps(psi, psi_fix, opt)) )
            #print("infidelity", 1-(psi_f.H & psi).contract(all, optimize=opt))

        #infidelity = dis_mps(psi, psi_fix, opt)
    return psi, error_

def site_blocks_(block_l):
    L_mps = len(block_l)
    site_blocks = []
    count = 0
    for i in range(L_mps):
        local_ = []
        for i_ in range(block_l[i]):
            local_.append(count)
            count += 1
        site_blocks.append(local_)    
    return site_blocks

def to_backend(x):
    import torch
    #import tensorflow
    return torch.tensor(x, dtype=torch.complex128, device='cpu')
    #return tensorflow.Tensor(x, dtype=tensorflow.complex128, device='cpu')
#x_mpo.apply_to_arrays(to_backend)

def gate_to_mpo_(gate_, where_, L, dtype="complex128", cutoff = 1.0e-12, max_bond_mpo=256, style="left"):
    for count, where in enumerate(where_):
        if count ==0:
            mpo = mpo_from_gate(gate_[count], where, L, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
        else:
            mpo_ = mpo_from_gate(gate_[count], where, L, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
            mpo = mpo_.apply(mpo)
            mpo.compress(style, max_bond=max_bond_mpo, cutoff=cutoff) 
    return mpo

#@profile
def update_mpo(x_mpo, opt, gate_l, where_l, 
               inds_fuse,
                method = "svd", 
                dtype = "complex128", 
                cutoff=1e-12, 
                rand_strength=0.05,
                form = "left",
                n_iter_dmrg = 15,
                depth_r = 1,
                bnds = [],
                block_l = [],
                threshold = 1.e-9,
                fidel_cal = True,
                smart_canon = True,
                prgbar = False,
                psi = None,
                to_backend= None,
                to_backend_mpo = "numpy-cpu-double",
                tol=1.e-6,
                tol_final=1.e-10,
                damping = 0.01,
                max_iterations = 256,
                opt_ = None,
                prog_compress = True,
                label = None,
                          ):

    MAGENTA = "\033[95m"
    CYAN = '\033[96m'
    RESET = '\033[0m'
    if to_backend:
        to_backend_mpo = get_to_backend(to_backend)
        to_backend = get_to_backend(to_backend)
    else:
        to_backend_mpo = get_to_backend(to_backend_mpo)

    if to_backend:
        psi.apply_to_arrays(to_backend)
        gate_l = gate_to_backend(gate_l, to_backend)
        x_mpo.apply_to_arrays(to_backend)

    fidel_mpo_l = []
    loss_val = 1
    site_blocks = site_blocks_(block_l)
    bond_dim = max(bnds)
    L = sum(block_l)
    n_tqgate_t = 0
    n_tqgate_l = []
    F_avg = 1
    ind_kb = { f"k{i}":f"b{i}" for i in range(L)}
    ind_kl = { f"k{i}":f"l{i}" for i in range(L)}
    ind_kg = { f"k{i}":f"g{i}" for i in range(L)}
    ind_gk = { f"g{i}":f"k{i}" for i in range(L)}
    ind_bl = { f"b{i}":f"l{i}" for i in range(L)}
    ind_bg = { f"b{i}":f"g{i}" for i in range(L)}
    ind_gb = { f"g{i}":f"b{i}" for i in range(L)}
    ind_lb = { f"l{i}":f"b{i}" for i in range(L)}
    ind_bk = { f"b{i}":f"k{i}" for i in range(L)}
    retag_ = {}
    for count, i in enumerate(site_blocks):
        for site_ in i:
            retag_ |= {f"I{site_}":f"I{count}"}

    x_mpo.canonize(x_mpo.L//2, cur_orthog='calc', bra=None)
    cur_orthog = x_mpo.L//2
    

    
    X_l = []
    N_l = []
    if method == "svd":
        print( CYAN+"method:", method, "depth_r:", depth_r, "smart_canon:", f"{smart_canon}"+RESET)



        with tqdm(total=len(gate_l),  desc="svd:",  
                            leave=True, position=0, 
                            colour='GREEN', disable = not prgbar) as pbar:

            for depth in range(len(gate_l)):
                gate_l_ = [gate_l[depth]]
                where_l_ = [where_l[depth]]
                if depth_r:
                    gate_ll = gate_l[depth]
                    where_ll = where_l[depth]                    
                    gate_l_= [gate_ll[i:i + depth_r] for i in range(0, len(gate_ll), depth_r)]
                    where_l_= [where_ll[i:i + depth_r] for i in range(0, len(where_ll), depth_r)]
                
                for count in range(len(gate_l_)):
                    dmrg_run = dmrg_run_(where_l_[count], site_blocks)
                    xmin, xmax = gate_support(where_l_[count], site_blocks)

                    x_mpo = mpo_to_btn(x_mpo, inds_fuse, inplace=True)
                    apply_mpo_swap(x_mpo, gate_l_[count], where_l_[count], 
                                contract = "reduce-split", #, "split", reduce-split 
                                cutoff=cutoff,
                                )
                    x_mpo, inds_fuse = btn_to_mpo(x_mpo, len(block_l), inplace=True)


                    if dmrg_run:
                        if smart_canon:
                            x_mpo_fix = x_mpo * 1.0
                            x_mpo.canonize([xmin, xmax], cur_orthog=cur_orthog, bra=None)
                            cur_orthog = [xmin, xmax]
                            for i in range(xmax, xmin, -1):
                                x_mpo.right_canonize_site(i, bra=None)
                            x_mpo.left_compress(start=xmin, stop=xmax,  max_bond=bond_dim, 
                                            **{"cutoff":cutoff})
                        else:
                            x_mpo_fix = x_mpo * 1.0                        
                            try:
                                x_mpo.compress( form = form, max_bond=bond_dim, cutoff=cutoff, **{"method": "svd"} )
                            except: 
                                print("try-except")
                                x_mpo.compress( form = form, max_bond=bond_dim, cutoff=cutoff, **{"method": "svd"} )

                        if fidel_cal:
                            loss_val = abs(fidel_mps(x_mpo, x_mpo_fix, opt))
                            fidel_mpo_l.append(loss_val)


                if psi:
                    norm = abs((x_mpo & x_mpo.H).contract(all, optimize=opt))
                    norm = abs(complex(norm))
                    norm = (norm/2**L)**0.5
                    psiH = psi.H
                    psiH.reindex_({ f"k{i}":f"b{i}"  for i in range(psi.L) })
                    X = (psiH & x_mpo & psi).contract(all, optimize=opt) 
                    X = abs(complex(X))
                    X_l.append(X)
                    N_l.append(norm)
                    qu.save_to_disk(N_l, f"Store/info_mpo/Norm_{label}_{bond_dim}")
                    qu.save_to_disk(X_l, f"Store/info_mpo/X_{label}_{bond_dim}")

                F_avg = np.prod(fidel_mpo_l)
                pbar.update(1)
                pbar.set_postfix({"L":L, 'bnd': max(bnds), "norm":norm, "X":X})
                pbar.refresh()
    if method == "bp":
        psi = mps_to_btn(psi, inds_fuse)
        print( CYAN+"method:", method, "depth_r", depth_r, "smart_canon:", f"{smart_canon}"+RESET)
        x_mpo = mpo_to_btn(x_mpo, inds_fuse, inplace=True)
        with tqdm(total=len(gate_l),  desc="bp:",  
                            leave=True, position=0, 
                            colour='GREEN', disable = not prgbar) as pbar:

            for depth in range(len(gate_l)):

                x_mpo.equalize_norms_()
                gate_l_ = [gate_l[depth]]
                where_l_ = [where_l[depth]]
                if depth_r:
                    gate_ll = gate_l[depth]
                    where_ll = where_l[depth]                    
                    gate_l_= [gate_ll[i:i + depth_r] for i in range(0, len(gate_ll), depth_r)]
                    where_l_= [where_ll[i:i + depth_r] for i in range(0, len(where_ll), depth_r)]
                
                for count in range(len(gate_l_)):
                    #x_mpo.balance_bonds_()
                    x_mpo.equalize_norms_()
                    svd_run, n_tqgate = dmrg_run_( where_l_[count], site_blocks)
                    xmin, xmax = gate_support(where_l_[count], site_blocks)
                    n_tqgate_t += n_tqgate
                    n_tqgate_l.append(n_tqgate)


                    if svd_run:
                        #x_mpo.equalize_norms_()
                        #x_mpo.balance_bonds_()
                        
                        mpo_ = gate_to_mpo_(gate_l_[count], where_l_[count], L)
                        mpo_.retag_(retag_)
                        mpo_.apply_to_arrays(to_backend_mpo)
                        
                        mpo_.equalize_norms_()
                        #mpo_.balance_bonds_()


                        x_mpo = apply_mpo_sandwich(x_mpo, mpo_)
                        
                        x_mpo_fix = x_mpo.copy()
                        compress_l2bp(x_mpo, max_bond=bond_dim, cutoff=cutoff, 
                                        cutoff_mode='rsum2', 
                                        site_tags=[ f"I{i}" for i in range(len(block_l))],
                                        max_iterations=max_iterations,
                                        tol=tol, 
                                        optimize=opt_,
                                        damping=damping, 
                                        progbar=prog_compress, 
                                        inplace=True,
                                        ) 

                    else:
                        apply_mpo_swap(x_mpo, gate_l_[count], where_l_[count], 
                                        contract = True, #, "split", reduce-split 
                                        cutoff=cutoff, tags=[],
                                        )


    
                if psi:
                    
                    site_tags = [f"I{i}" for i in range(len(block_l))]
                    norm = abs((x_mpo & x_mpo.H).contract(all, optimize=opt))
                    bp = L2BP(x_mpo, optimize=opt, site_tags = site_tags)
                    bp.run(tol=tol_final, max_iterations=max_iterations, progbar=prog_compress)
                    mantissa, norm_exponent = bp.contract(strip_exponent=True)
                    est_norm = float(10 ** ((norm_exponent - (L * log10(2))) / 2))
                    est_norm = (abs(complex(mantissa))**-.5) * est_norm
                    norm = abs(complex(est_norm))


                    psiH = psi.H
                    psiH.reindex_({ f"k{i}":f"b{i}"  for i in range(L) })
                    tn_x = (psiH & x_mpo & psi)

                    X = contract_l1bp(  tn_x,
                                        tol=tol_final,
                                        site_tags = site_tags,
                                        max_iterations=max_iterations,
                                        optimize=opt,
                                        progbar=prog_compress,
                                        damping=0,
                                        strip_exponent=False
                                    )
                                

                    X = abs(complex(X))
                    X_l.append(X)
                    N_l.append(norm)
                    qu.save_to_disk(N_l, f"Store/info_mpo/Norm_{label}_{bond_dim}")
                    qu.save_to_disk(X_l, f"Store/info_mpo/X_{label}_{bond_dim}")
                
                F_avg = np.prod(fidel_mpo_l)
                pbar.update(1)
                pbar.set_postfix({'bnd': max(bnds), "norm":norm, "X":X})
                pbar.refresh()
        x_mpo, inds_fuse = btn_to_mpo(x_mpo, len(block_l), inplace=True)
        x_mpo.view_as_(
                qtn.tensor_1d.MatrixProductOperator,
                L=len(block_l), 
                site_tag_id='I{}',
                upper_ind_id='k{}',
                lower_ind_id='b{}',
                cyclic = False,
                    )

    pbar.close()
    return (x_mpo, inds_fuse), fidel_mpo_l, X_l, N_l



#@profile
def Mps_bp(psi, opt, gate_l, where_l, inds_fuse,
            cutoff=1e-12, 
            form = "left",
            depth_r = 1,
            bnds = [],
            block_l = [],
            fidel_cal = False,
            smart_canon = True,
            prgbar = False,
            to_backend= None,
            cur_orthog = 'calc',
            svd_init = None,
            threshold = None,
            n_iter_dmrg = None,
            fidel_exact = False,
            store_state= False,
            label = "rand",
            tol=1.e-6,
            damping = 0.01,
            max_iterations = 256,
            opt_ = None,
            to_backend_mpo = "numpy-cpu-double",
            prog_bp = False,
            O_label = None,
            site = None,
            ):
    
    if to_backend:
        to_backend_mpo = get_to_backend(to_backend)
        to_backend = get_to_backend(to_backend)
        psi.apply_to_arrays(to_backend)
        gate_l = gate_to_backend(gate_l, to_backend)
        
    else:
        to_backend_mpo = get_to_backend(to_backend_mpo)




    site_blocks = site_blocks_(block_l)
    bond_dim = max(bnds)
    block_size = max(block_l)

    error_ = [1]
    N_TQG_l = []
    n_tqgate_l = []

    e_l = []
    MAGENTA = "\033[95m"
    CYAN = '\033[96m'
    RESET = '\033[0m'
    cost = 1
    F_t = 1
    F_avg = 1
    n_tqgate_t = 0
    n_tqgate_l = []
    fidel_svd_l = []
    e_svd_l = []
    f_svd_l = []
    e_wrt_exact_l = []
    f_wrt_exact_l = []
    n_tqgate_l.append(0)
    L = sum(block_l)
    error_tqg = 0
    Cotengra_W = [0]
    Cotengra_C = [0]
    X_l = []
    X_ = 0
    ind_kb = { f"k{i}":f"b{i}" for i in range(L)}
    ind_bl = { f"b{i}":f"l{i}" for i in range(L)}
    ind_lb = { f"l{i}":f"b{i}" for i in range(L)}
    ind_bk = { f"b{i}":f"k{i}" for i in range(L)}

    retag_ = {}
    for count, i in enumerate(site_blocks):
        for site_ in i:
            retag_ |= {f"I{site_}":f"I{count}"}


    psi = mps_to_btn(psi, inds_fuse, inplace=True)

    with tqdm(total=len(gate_l),  desc="mps_bp:", leave=True, position=0, 
                colour='MAGENTA', disable = not prgbar) as pbar:
        for depth in range(len(gate_l)):
            gate_l_ = [gate_l[depth]]
            where_l_ = [where_l[depth]]
            if depth_r:
                gate_ll = gate_l[depth]
                where_ll = where_l[depth]                    
                gate_l_= [gate_ll[i:i + depth_r] for i in range(0, len(gate_ll), depth_r)]
                where_l_= [where_ll[i:i + depth_r] for i in range(0, len(where_ll), depth_r)]
            
            for count in range(len(gate_l_)):
                psi.balance_bonds_()
                svd_run, n_tqgate = dmrg_run_( where_l_[count], site_blocks)
                xmin, xmax = gate_support(where_l_[count], site_blocks)
                n_tqgate_t += n_tqgate
                n_tqgate_l.append(n_tqgate)


                if svd_run:
                    norm = (psi.H & psi).contract(all, optimize=opt)
                    psi = psi * (norm**(-0.5))
                    # psi = apply_long_mps(psi, gate_l_[count], where_l_[count], 
                    #         site_blocks, inplace=True
                    #         )
                    
                    mpo_ = gate_to_mpo_(gate_l_[count], where_l_[count], L)
                    mpo_.retag_(retag_)
                    mpo_.apply_to_arrays(to_backend_mpo)
                    
                    
                    psi.reindex_(ind_kb)
                    psi = psi & mpo_ 
                    psi_fix = psi.copy()

                    compress_l2bp(psi, max_bond=bond_dim, cutoff=cutoff, 
                                    cutoff_mode='rsum2', 
                                    site_tags=[ f"I{i}" for i in range(len(block_l))],
                                    max_iterations=max_iterations,
                                    tol=tol, 
                                    optimize=opt_,
                                    damping=damping, 
                                    progbar=prog_bp, 
                                    inplace=True,
                                    ) 
                    
                    if fidel_cal:
                        psi_, inds_fuse = btn_to_mps(psi, len(block_l), inplace=True)
                        psi_.normalize()
                        psi = mps_to_btn(psi_, inds_fuse, inplace=True)
                        cost = abs(fidel_mps_normalized(psi_, psi_fix, opt))
                        error_.append(cost)         
                else:
                    apply_swap(psi, gate_l_[count], where_l_[count], 
                           contract = True, #, "split"
                           cutoff=cutoff,
                        )
                    cost = 1
                    error_.append(cost)


            N_TQG_l.append(n_tqgate_t)
            if fidel_cal and (n_tqgate_t != 0):
                error_tqg = 0
                for f_ in error_: 
                    error_tqg += -np.log(f_) 
                error_tqg = abs(complex(error_tqg * (1/n_tqgate_t)))
                e_l.append(abs(error_tqg))

                qu.save_to_disk(np.prod(error_), f"Store/info_mps/Fidel_bp_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(N_TQG_l, f"Store/info_mps/N_bp_{label}_TQG_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(e_l, f"Store/info_mps/e_bp_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(Cotengra_C, f"Store/info_mps/C_bp_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(Cotengra_W, f"Store/info_mps/W_bp_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(n_tqgate_l, f"Store/info_mps/K_bp_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
            
            if O_label:
                psi_ = psi.copy()
                psi_gate = qtn.tensor_network_gate_inds(psi_, 
                                                        O_label[0] , 
                                                        [f"k{site[0]}"], 
                                                        contract=False,  
                                                        inplace=False
                                        )

                X_ = (psi_gate & psi_.H).contract(all, optimize=opt)
                X_ = abs(complex(X_))
                X_l.append(X_)
                qu.save_to_disk(X_l, f"Store/info_mps/X_bp_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")


 
            
            pbar.set_postfix({"N" : sum(block_l),'F': abs(complex(error_[-1])), 
                              "e": error_tqg, "tqg": n_tqgate_t,
                              "bnd": psi.max_bond(),"K": max(n_tqgate_l),
                              "<X>":X_
                            }
                            )
            pbar.refresh()
            pbar.update(1)


    psi, inds_fuse = btn_to_mps(psi, len(block_l), inplace=True)
    psi.normalize()


    return (psi, inds_fuse), (error_, N_TQG_l, e_l), X_l,



def Mps_mpo(bnd = 128, lx=4, ly=4, depth_=4, theta = -2*math.pi/48, cutoff = 1.e-12):


    L = lx * ly
    to_backend, opt, opt_ = req_backend(chi=None, threads=None)
    to_backend_ = get_to_backend(to_backend) #"numpy-single"
    
    circ, gate_l, where_l, info = prepare_gates( 
                                              depth_=depth_, 
                                              lx=lx, 
                                              ly=ly,
                                              label = "t2",
                                              theta = theta, 
                                              )
    
    peps, mpo_l, pepo_l, x_pepo, label = info
    def ps(theta, phi, L):
    
        vec = np.array([math.cos(theta), np.exp(-complex(0,1)*phi)*math.sin(theta)]) 
    
        p = qtn.MPS_computational_state([0]*L)
        for t in p:
            vec = np.array([math.cos(theta), np.exp(-complex(0,1)*phi)*math.sin(theta)]) 
            shape = t.shape
            t.modify(data = vec.reshape(shape))
        return p 
    p = ps(theta, 0, L)
    p.apply_to_arrays(to_backend_)

    
    info_c={"cur_orthog":"calc"}
    p.canonicalize_([0], cur_orthog='calc', info=info_c)
    print(info_c)
    z_l = []
    zz_l = []
    z2_l = []
    s_l = []
    for cycle_, where in tqdm(enumerate(where_l)):
        entropy = p.entropy(L//2, info=info_c, method='svd')
        z = complex(energy_global(mpo_l[0], p, opt)).real
        zz = complex(energy_global(mpo_l[1], p, opt)).real
        z2 = complex(energy_global(mpo_l[2], p, opt)).real
        print(entropy, z, zz, z2)
        z_l.append(z)
        zz_l.append(zz)
        z2_l.append(z2)
        s_l.append(entropy)
        for count_, where_ in enumerate(where):
            if len(where_) == 1:
                x, = where_
                p.gate_(gate_l[cycle_][count_], x, tags=["U1"], contract=True)
                p.canonicalize_([x], cur_orthog=info_c["cur_orthog"], info=info_c)
            if len(where_) == 2:
                x, y = where_
                p.gate_nonlocal_(gate_l[cycle_][count_], [x,y], max_bond = bnd , info=info_c ,dims=None, method='direct', **{"cutoff":cutoff})


    return p, s_l, z_l, zz_l, z2_l



#@profile
def Mps_svd(psi, gate_l, where_l, inds_fuse,
            cutoff=1e-12, 
            form = "left",
            depth_r = 1,
            bnds = [],
            block_l = [],
            fidel_cal = False,
            smart_canon = True,
            prgbar = False,
            to_backend= None,
            cur_orthog = 'calc',
            svd_init = None,
            threshold = None,
            n_iter_dmrg = None,
            fidel_exact = False,
            store_state= False,
            label = "rand",
            tol=1.e-6,
            damping = 0.01,
            max_iterations = 256,
            opt_ = None,
            to_backend_mpo = None,
            O_label = None,
            site = None,
            prog_bp = False,
            rand_strength = 0,
            mpo_l = None,
            ):
    
    if to_backend:
        to_backend = get_to_backend(to_backend)

    # if to_backend:
    #     psi.apply_to_arrays(to_backend)
    #     gate_l = gate_to_backend(gate_l, to_backend)

    site_blocks = site_blocks_(block_l)
    bond_dim = max(bnds)
    block_size = max(block_l)

    error_ = [1]
    N_TQG_l = []
    n_tqgate_l = []
    mpo_res = []
    e_l = []
    MAGENTA = "\033[95m"
    CYAN = '\033[96m'
    RESET = '\033[0m'
    cost = 1
    F_t = 1
    F_avg = 1
    n_tqgate_t = 0
    n_tqgate_l = []
    fidel_svd_l = []
    e_svd_l = []
    f_svd_l = []
    e_wrt_exact_l = []
    f_wrt_exact_l = []
    n_tqgate_l.append(0)
    L = sum(block_l)
    error_tqg = 0

    X_l = []
    X_ = 0

    if cur_orthog == 'calc':
        psi.canonize(psi.L//2, cur_orthog='calc', bra=None)
        cur_orthog = [psi.L//2]
    else:
        psi.canonize(cur_orthog, cur_orthog='calc', bra=None)
        cur_orthog = cur_orthog
    Cotengra_W = [0]
    Cotengra_C = [0]
    if prgbar:
        print( CYAN+ f"{label}: depth_r:", depth_r, "smart_canon:", f"{smart_canon}"+RESET)


    with tqdm(total=len(gate_l),  desc="mps_svd:", leave=True, position=0, 
                colour='MAGENTA', disable = not prgbar) as pbar:
        for depth in range(len(gate_l)):
            gate_l_ = [gate_l[depth]]
            where_l_ = [where_l[depth]]
            if depth_r:
                gate_ll = gate_l[depth]
                where_ll = where_l[depth]                    
                gate_l_= [gate_ll[i:i + depth_r] for i in range(0, len(gate_ll), depth_r)]
                where_l_= [where_ll[i:i + depth_r] for i in range(0, len(where_ll), depth_r)]
            
            for count in range(len(gate_l_)):
                
                psi.normalize(insert=cur_orthog[0])

                svd_run, n_tqgate = dmrg_run_( where_l_[count], site_blocks)
                xmin, xmax = gate_support(where_l_[count], site_blocks)
                n_tqgate_t += n_tqgate
                n_tqgate_l.append(n_tqgate)

                psi_fix = mps_to_btn(psi, inds_fuse, inplace=True)
                apply_swap(psi_fix, gate_l_[count], where_l_[count], 
                           contract = "reduce-split", #, "split"
                           cutoff=cutoff,
                        )
                psi_fix, inds_fuse = btn_to_mps(psi_fix, len(block_l), inplace=True)
                psi = psi_fix.copy()

                if svd_run:
                    if smart_canon:

                        psi.canonize([xmin, xmax], cur_orthog=cur_orthog, bra=None)
                        cur_orthog = [xmin, xmax]
                        
                        for i in range(xmax, xmin, -1):
                            psi.right_canonize_site(i, bra=None)

                        psi.left_compress(start=xmin, stop=xmax,  max_bond=bond_dim, 
                                          **{"cutoff":cutoff}
                                          )
                    else:
                        psi.compress(form, max_bond=bond_dim, cutoff=cutoff)
                

                    if fidel_cal:
                        psi.normalize(insert=cur_orthog[0])
                        cost = abs(fidel_mps_normalized(psi, psi_fix, opt))
                        #cost_ = abs(fidel_mps(psi, psi_fix, opt))
                        error_.append(cost)         
                else:
                    cost = 1
                    error_.append(cost)


            N_TQG_l.append(n_tqgate_t)
            if fidel_cal and (n_tqgate_t != 0):
                error_tqg = 0
                for f_ in error_: 
                    error_tqg += -np.log(f_) 
                error_tqg = abs(complex(error_tqg * (1/n_tqgate_t)))
                e_l.append(abs(error_tqg))

                qu.save_to_disk(np.prod(error_), f"Store/info_mps/Fidel_svd_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(N_TQG_l, f"Store/info_mps/N_svd_{label}_TQG_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(e_l, f"Store/info_mps/e_svd_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(Cotengra_C, f"Store/info_mps/C_svd_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(Cotengra_W, f"Store/info_mps/W_svd_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(n_tqgate_l, f"Store/info_mps/K_svd_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")


            if O_label:
                psi_ = mps_to_btn(psi, inds_fuse, inplace=False)

                # psi_gate = qtn.tensor_network_gate_inds(psi_, 
                #                                         to_backend(O_label[0]) , 
                #                                         [f"k{site[0]}"], 
                #                                         contract=False,  
                #                                         inplace=False
                #                         )

                # X_ = (psi_gate & psi_.H).contract(all, optimize=opt)
                # X_ = abs(complex(X_))
                # X_l.append(X_)
                # qu.save_to_disk(X_l, f"Store/info_mps/X_svd_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                if mpo_l:
                    psi_h = psi_.H
                    psi_h.reindex_({f"k{i}":f"b{i}" for i in range(sum(block_l))})
                    x_local = []
                    for mpo in mpo_l:
                        x = (psi_h & mpo & psi_).contract(all, optimize=opt_) 
                        x_local.append(complex(x).real)
                    mpo_res.append(x_local)
                    print("x_local", x_local)
                    qu.save_to_disk(mpo_res, f"Store/info_mps/mpo_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
            
            pbar.set_postfix({"N" : sum(block_l),'F': abs(complex(error_[-1])), 
                              "e": error_tqg, "cur": cur_orthog, "tqg": n_tqgate_t,
                              "bnd": psi.max_bond(),"K": max(n_tqgate_l),
                              "<X>": X_,
                            }
                            )
            pbar.refresh()
            pbar.update(1)
            

    return (psi, inds_fuse), (error_, N_TQG_l, e_l), X_l

def rand_to_tn(psi, rand = 0.02, cur_orthog=None):

    if cur_orthog:
        start, stop = cur_orthog
    
    for count in range(psi.L):
        t = psi[f"I{count}"]
        t_ = t.copy()

        if "complex128" in str(t.dtype):
            t_.randomize(dtype="complex128", inplace=True)
        if "complex64" in str(t.dtype):
            t_.randomize(dtype="complex64", inplace=True)
        if  'float64' in  str(t.dtype):
            t_.randomize(dtype="float64", inplace=True)
        if  'float32' in  str(t.dtype):
            t_.randomize(dtype="float32", inplace=True)
        
        if cur_orthog:
            if count>=start and count<=stop:
                t = t + t_ * rand
                psi[f"I{count}"] = t
        else:
            t = t + t_ * rand
            psi[f"I{count}"] = t
    

    if cur_orthog:
        psi.normalize(insert=start)
    else: 
        psi.normalize()
    
    return psi



def Mps_dmrg(psi, gate_l, where_l, inds_fuse,
                n_iter_dmrg = 6,
                depth_r = 1,
                bnds = [],
                block_l = [],
                threshold = 8.e-8,
                smart_canon = True,
                prgbar = True,
                to_backend= "numpy-double",
                svd_init = False,
                fidel_exact = False,
                store_state = False,
                form = "left",
                cutoff=1e-12,
                label = "rand",
                tol=1.e-6,
                damping = 0.01,
                max_iterations = 256,
                fidel_cal = True,
                opt_ = None,
                O_label = [qu.pauli("Z")],
                site = None,
                prog_bp = False,
                rand_strength = 0.02,
                cur_orthog = [0, 1],
                mpo_l = None,
            ):



    #psi.normalize()
    psi = iregular_bnd(psi, bnds, rand_strength=0.0)
    backend, opt, opt_ = req_backend()
    to_backend_ = get_to_backend(backend)

    site_blocks = site_blocks_(block_l)
    bond_dim = max(bnds)
    X_l = []
    Z_l = []
    block_size = max(block_l)
    
    error_ = [1]
    N_TQG_l = []
    F_avg_l = []
    F_t_l = []
    e_l = []
    e_density_l = []
    error_density = []
    mpo_res = []
    MAGENTA = "\033[95m"
    CYAN = '\033[96m'
    RESET = '\033[0m'
    cost = 1
    F_t = 1
    F_avg = 1
    n_tqgate_t = 0
    n_tqgate_l = []
    error_tqg = 0
    psi.canonize(cur_orthog, cur_orthog='calc', bra=None)
    cur_orthog = cur_orthog
    cost_local = 0
    Cotengra_W = []
    Cotengra_C = []
    L = sum(block_l)
    e_svd_l = []
    f_svd_l = []
    e_wrt_exact_l = []
    f_wrt_exact_l = []
    f_avg_depth = []
    error_depth = []
    print( CYAN+ f"{label}: depth_r", depth_r, "smart_canon:", f"{smart_canon}"+RESET)
    print( MAGENTA+ "svd_init:", svd_init, ""+RESET)
    print("n_iter_dmrg", n_iter_dmrg)
    with tqdm(total=len(gate_l),  desc="mps_dmrg:",  
                leave=True, position=0, 
                colour='MAGENTA', disable = not prgbar) as pbar:

        for depth in range(len(gate_l)):
            t_dmrg_l = []
            t_svd_l = []

            Cotengra_W_local = []
            Cotengra_C_local = []
            gate_l_ = [gate_l[depth]]
            where_l_ = [where_l[depth]]
            if depth_r:
                gate_ll = gate_l[depth]
                where_ll = where_l[depth]
                gate_l_= [gate_ll[i:i + depth_r] for i in range(0, len(gate_ll), depth_r)]
                where_l_= [where_ll[i:i + depth_r] for i in range(0, len(where_ll), depth_r)]

            for count in range(len(gate_l_)):
                error_local = []
                start, stop = cur_orthog
                psi.normalize(insert=start)

                
                dmrg_run, n_tqgate = dmrg_run_(where_l_[count], site_blocks)
                n_tqgate_t += n_tqgate
                n_tqgate_l.append(n_tqgate)
                xmin, xmax = gate_support(where_l_[count], site_blocks)

                if dmrg_run:

                    if smart_canon:
                        psi.canonize([xmin, xmax], cur_orthog=cur_orthog, bra=None)
                        cur_orthog = [xmin, xmax]

                        psi_fix = psi.copy()
                        psi_fix = mps_to_btn(psi_fix, inds_fuse, inplace=True)
                        #
                        apply_(psi_fix, gate_l_[count], where_l_[count], 
                                contract = "auto-split-gate", tags=["G"])
                        #psi_fix_ = psi_fix.copy()
                        #print(psi_fix_, psi_fix_.outer_inds())
                        if svd_init:
                            
                            start_time = time.time()
                            (psi, inds_fuse), info, l_ = Mps_svd(psi, opt, 
                                                                [gate_l_[count]], 
                                                                [where_l_[count]], 
                                                                inds_fuse,
                                                                bnds = bnds,
                                                                block_l = block_l,
                                                                fidel_cal = False,
                                                                smart_canon = True,
                                                                cur_orthog = cur_orthog.copy(),
                                                                cutoff = cutoff,
                                                                tol=tol,
                                                                damping = damping,
                                                                max_iterations = max_iterations,
                                                                opt_ = opt_,
                                                                to_backend_mpo = to_backend,
                                                                )  
                            

                            t_svd_l.append((time.time() - start_time))
                            # currunt_bnd = virtual_bond_max(psi)
                            # if currunt_bnd < max(bnds):
                            #     print("increase bnd", currunt_bnd, max(bnds))
                            #     psi = iregular_bnd(psi, bnds, rand_strength=0.0)
                            #     psi.canonize(cur_orthog, cur_orthog="calc", bra=None)
                            #     print(psi, psi.show())

                            # inds_cor_ = []
                            # for cor in range(psi.L - 1):
                            #     inds_cor_.append(psi.bond(cor,cor+1))
                            # dic_ = {}
                            # for counter_ in range(len(inds_cor_)):
                            #     dic_ |= {inds_cor_[counter_]:inds_cor[counter_]}
                            # psi.reindex_(dic_)
                            # psi = iregular_bnd(psi, bnds_, rand_strength=0.0)
                        
                        start_time = time.time()
                        psi = rand_to_tn(psi, rand = rand_strength, cur_orthog=cur_orthog)

                        psi, cost, inds_fuse, cotengra_cost = bmps_dmrg_eff(psi, psi_fix, opt, 
                                                inds_fuse, cur_orthog,
                                                n_iter=n_iter_dmrg,  
                                                threshold = threshold,
                                                prgbar = prgbar,
                                                n_tqgate = n_tqgate,
                                                )

                        #psi_tn = mps_to_btn(psi, inds_fuse, inplace=False)
                        #print(cur_orthog, psi_tn, psi_fix)
                        #cost_local = error_density_local(psi_tn, psi_fix, sum(block_l), opt)
                        #cost_local = sum(cost_local)/len(cost_local)
                        #print(complex(cost).real, )

                        t_dmrg_l.append((time.time() - start_time))
                        C, W = cotengra_cost
                        Cotengra_W_local.append(W)
                        Cotengra_C_local.append(C)

                    else:
                        psi_fix = psi.copy()
                        psi_fix = mps_to_btn(psi_fix, inds_fuse, inplace=True)
                        apply_(psi_fix, gate_l_[count], where_l_[count], 
                                contract = "auto-split-gate", tags=[]
                                )
                        inds_phy = psi_fix.outer_inds()
                        psi_fix.rank_simplify(output_inds=inds_phy, inplace = True)

                        if svd_init:

                            start_time = time.time()
                            (psi, inds_fuse), info, l_ = Mps_bp(psi, opt, 
                                                                [gate_l_[count]], 
                                                                [where_l_[count]], 
                                                                inds_fuse,
                                                                bnds = bnds,
                                                                block_l = block_l,
                                                                fidel_cal = False,
                                                                smart_canon = False,
                                                                prgbar = False,
                                                                cur_orthog = cur_orthog.copy(),
                                                                cutoff = cutoff,
                                                                tol=tol,
                                                                damping = damping,
                                                                max_iterations = max_iterations,
                                                                opt_ = opt_,
                                                                to_backend_mpo = to_backend,
                                                            )
                            t_svd_l.append((time.time() - start_time))
                        
                        start_time = time.time()
                        psi = rand_to_tn(psi, rand = rand_strength)
                        psi, cost, inds_fuse, cotengra_cost = bmps_dmrg(psi, psi_fix, opt, 
                                                                        inds_fuse, 
                                                                        n_iter=n_iter_dmrg,  
                                                                        threshold = threshold,
                                                                        prgbar = prgbar,
                                                                        n_tqgate = n_tqgate,
                                                                        )                                      
                        t_dmrg_l.append((time.time() - start_time))
                        C, W = cotengra_cost
                        Cotengra_W_local.append(W)
                        Cotengra_C_local.append(C)

                    
                    
                    error_local.append(cost)
                    error_.append(cost)
                    #error_density.append(cost_local)
                else:
                    psi = mps_to_btn(psi, inds_fuse, inplace=True)
                    apply_(psi, gate_l_[count], where_l_[count], 
                            contract = True, tags=[])
                    psi, inds_fuse = btn_to_mps(psi, len(block_l), cycle=False,inplace=True)
        
            N_TQG_l.append(n_tqgate_t)
            error_depth.append(error_local)
            if n_tqgate_t != 0:
                error_tqg = 0
                for f_ in error_:
                    error_tqg += -np.log(f_)

                # error_density_tqg = 0
                # for f_ in error_density:
                #     error_density_tqg += -np.log(f_)

                error_tqg = complex(error_tqg * (1/n_tqgate_t))
                # error_density_tqg = error_density_tqg * (1/n_tqgate_t)
                # print(error_density_tqg)
                
                e_l.append(abs(error_tqg))
#                e_density_l.append(abs(error_density_tqg))



                #psi_store = mps_to_btn(psi, inds_fuse)
                f_avg_depth.append(abs(complex(np.prod(error_))))
                Cotengra_W.append(max(Cotengra_W_local))
                Cotengra_C.append(max(Cotengra_C_local))


                qu.save_to_disk(f_avg_depth, f"Store/info_mps/F_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(N_TQG_l, f"Store/info_mps/Tqg_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(e_l, f"Store/info_mps/E_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                #qu.save_to_disk(e_density_l, f"Store/info_mps/ed_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(Cotengra_C, f"Store/info_mps/C_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(Cotengra_W, f"Store/info_mps/W_{label}_l_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(n_tqgate_l, f"Store/info_mps/K_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                #qu.save_to_disk(psi_store, f"Store/psi-bnd{bond_dim}D{depth}d{depth_r}")
        
            if site:
                psi.normalize()
                print(psi.show(), len(block_l)//2)
                start_time = time.time()
                entropy = psi.entropy(len(block_l)//2, cur_orthog=cur_orthog, method='svd')
                entropy_time = (time.time() - start_time)
                print("entropy_time", entropy_time)
                psi_ = mps_to_btn(psi, inds_fuse, inplace=False)

                start_time = time.time()
                zz_l = []
                for count, cor_ in enumerate(site):
                    x, y = cor_
                    Z, X = O_label[count]
                    psi_gate = qtn.tensor_network_gate_inds(psi_, 
                                                            to_backend_(Z), 
                                                            [f"k{x}"], 
                                                            contract=False,  
                                                            inplace=False
                                                            )
                    psi_gate = qtn.tensor_network_gate_inds(psi_gate, 
                                                            to_backend_(X), 
                                                            [f"k{y}"], 
                                                            contract=False,  
                                                            inplace=False
                                                            )
                    
                    X_ = (psi_gate & psi_.H).contract(all, optimize=opt)
                    X_ = complex(X_)
                    zz_l.append(round(complex(X_).real,7))
                Z_l.append(zz_l)
                print("site", site)
                print("zz_l", zz_l)
                zz_time = (time.time() - start_time)
                print("zz_time", zz_time)
                
                qu.save_to_disk(Z_l, f"Store/info_mps/X_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")
                qu.save_to_disk(site, f"Store/info_mps/site_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")


                if store_state:
                    psi.right_canonize()
                    psi_ = mps_to_btn(psi, inds_fuse, inplace=False)
                    qu.save_to_disk(psi_, f"Store/info_mps/state_mps/mps_bnd{max(bnds)}_{depth}")
                    qu.save_to_disk(inds_fuse, f"Store/info_mps/state_mps/inds_fuse_bnd{max(bnds)}_{depth}")

                if mpo_l:
                    psi_h = psi_.H
                    psi_h.reindex_({f"k{i}":f"b{i}" for i in range(sum(block_l))})
                    x_local = [entropy]
                    for mpo in mpo_l:
                        x = (psi_h & mpo & psi_).contract(all, optimize=opt) 
                        x_local.append(complex(x).real)
                    mpo_res.append(x_local)
                    print("mpo_res", x_local)
                    qu.save_to_disk(mpo_res, f"Store/info_mps/mpo_{label}_L{L}bnd{bond_dim}b{block_size}d{depth_r}")


            pbar.set_postfix({"L" : L,
                              'F': abs(complex(np.prod(error_))), "e": abs(error_tqg), 
                              "td":sum(t_dmrg_l),
                                "ts":sum(t_svd_l),
                                "K": max(n_tqgate_l),
                                "bnd": max(bnds),
                                "W":max(Cotengra_W), 
                                "C":max(Cotengra_C),
                                "<X>":X_,}
                                )


            pbar.refresh()
            pbar.update(1)
            
        
    pbar.close()
    return (psi, inds_fuse), (error_, N_TQG_l, e_l), X_l


def  mpo_ITF(L_L, data_type="complex64", chi=200, cutoff_val=1.0e-12, field=1.0,sign="+"):

    Z = qu.pauli('Z',dtype=data_type) 
    X = qu.pauli('X',dtype=data_type) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=data_type)
    #  Y = Y.astype(data_type)
    #  X = X.astype(data_type)
    #  Z = Z.astype(data_type)

    Ham = [X]
    MPO_I=qtn.MPO_identity(L_L, phys_dim=2)
    MPO_result=qtn.MPO_identity(L_L, phys_dim=2)
    MPO_result=MPO_result*0.0
    MPO_f=MPO_result*0.0

    max_bond_val=chi
    cutoff_val=cutoff_val
    for count, elem in enumerate (Ham):
        for i in range(L_L):
            ii = i
            ii_ = (i+1)%L_L
        #   print("mpo_info", ii,ii_,MPO_result.max_bond())
            Wl = np.zeros([ 1, 2, 2], dtype=data_type)
            W = np.zeros([1, 1, 2, 2], dtype=data_type)
            Wr = np.zeros([ 1, 2, 2], dtype=data_type)

            Wl[ 0,:,:]=elem
            W[ 0,0,:,:]=elem
            Wr[ 0,:,:]=elem
            W_list=[Wl]+[W]*(L_L-2)+[Wr]
            MPO_I=qtn.MPO_identity(L_L, phys_dim=2 )
            MPO_I[ii].modify(data=W_list[ii])
            MPO_I[ii_].modify(data=W_list[ii_])
            if sign=="+":
                MPO_result=MPO_result+MPO_I
            if sign=="-":
                MPO_result=MPO_result+MPO_I*-1.

            MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

    Ham = [Z*field]
    for count, elem in enumerate (Ham):
        for i in range(L_L):
            # print("mpo_info", i, MPO_result.max_bond())
            Wl = np.zeros([ 1, 2, 2], dtype=data_type)
            W = np.zeros([1, 1, 2, 2], dtype=data_type)
            Wr = np.zeros([ 1, 2, 2], dtype=data_type)

            Wl[ 0,:,:]=elem
            W[ 0,0,:,:]=elem
            Wr[ 0,:,:]=elem
            W_list=[Wl]+[W]*(L_L-2)+[Wr]
            MPO_I=qtn.MPO_identity(L_L, phys_dim=2 )
            MPO_I[i].modify(data=W_list[i])
            if sign=="+":
                MPO_result=MPO_result+MPO_I
            if sign=="-":
                MPO_result=MPO_result+MPO_I*-1.

            #MPO_result += MPO_I
            MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

    return  MPO_result 



    #print(loss_(psi, psi_fix, opt))




    # if method == "swap":
    #     print("method=", method)
    #     for gate_dic in gate_dic_l:
    #         for count, gate_dic_local in tqdm(enumerate(split_dictionary(gate_dic, chunk_size))):
    #             psi, error_ , central= swap_method_(psi, central, gate_dic_local, bond_dim, opt,tags,  error_, dtype=dtype, cutoff=cutoff, verbosity=verbosity)
    

    # elif method == "autodiff":
    #     print("method", method)
    #     for gate_dic in gate_dic_l:
    #         for count, gate_dic_local in enumerate(split_dictionary(gate_dic, chunk_size)):
    #             print("count", count, len(gate_dic), len(gate_dic_local))
    #             psi_guess, error_ = swap_method(psi*1.0, gate_dic_local, bond_dim, opt,tags, dtype=dtype, cutoff=cutoff, verbosity=verbosity)
    #             psi_guess.expand_bond_dimension_(bond_dim, rand_strength=rand_strength)
    #             psi_guess.normalize()
    #             psi_fix = psi.copy()
    #             for where in gate_dic_local:
    #                 psi_fix.gate_(gate_dic_local[where], where, tags=tags)
    
    #             psi = mps_auto(psi_guess, psi_fix, opt, n_iter=n_iter_auto, cost_f = "fidelity", disp = "off", progbar=False, threshold = 1.e-9)
    # elif method == "dmrg":
    #     print("method=", method)
    #     for gate_dic in gate_dic_l:
    #         for count, gate_dic_local in enumerate(split_dictionary(gate_dic, chunk_size)):
    #             print("count", count, len(gate_dic), len(gate_dic_local))
    #             psi_guess, error_ = swap_method(psi*1.0, gate_dic_local, bond_dim, opt,tags, dtype=dtype, cutoff=cutoff, verbosity=verbosity)
    #             psi_guess.expand_bond_dimension_(bond_dim, rand_strength=rand_strength)
    #             psi_guess.normalize()
    #             psi_fix = psi.copy()
    #             for where in gate_dic_local:
    #                 psi_fix.gate_(gate_dic_local[where], where, tags=tags)
    #             psi = mps_dmrg(psi_guess, psi_fix, opt, n_iter=n_iter_dmrg, cost_f = "fidelity", disp = "on", threshold = 1.e-9)
    # elif method == "autodiff":
    #     print("method=", method, "depth_r", depth_r)
    #     mpo_l_= [mpo_l[i:i + depth_r] for i in range(0, len(mpo_l), depth_r)]
    #     for count, mpo_ in tqdm(enumerate(mpo_l_)):
    #         psi_fix = psi*1.
    #         for count, mpo_local in enumerate(mpo_):
    #             mpo_local = tn_T(mpo_local, mpo_local.L)
    #             psi_fix = mpo_local.apply(psi_fix, 
    #                                     compress=True, 
    #                                     form=form, 
    #                                     max_bond=max_bond_dmrg, 
    #                                     cutoff=cutoff,
    #                                     )
    #         #print("psi_fix.norm()", psi_fix.norm())
    #         #print("psi_fix: max_bond()", psi_fix.max_bond())
    #         psi_guess = psi_fix * 1.
    #         psi_guess.compress("left", max_bond=bond_dim//2, cutoff=cutoff)
    #         psi_guess.expand_bond_dimension_(bond_dim, rand_strength=rand_strength)
    #         psi_guess.normalize()
            
    #         psi = mps_auto(psi_guess, psi_fix, opt, n_iter=n_iter_auto, cost_f = "fidelity", disp = "off", progbar=True, threshold = 1.e-12)
    #         #print("psi", psi)

    #     if verbosity == 1:
    #         cost = abs(fidel_mps(psi, psi_fix, opt))
    #         error_.append( cost )

    # elif method == "dmrg-mpo":
    #     print( CYAN+"method=", method, "depth_r", depth_r, )
    #     print(MAGENTA +"bnds", f"{bnds}"+RESET)
    #     # shrink MPO as taking larger circuit depth 
    #     gate_l_= [gate_l[i:i + depth_r] for i in range(0, len(gate_l), depth_r)]
    #     where_l_= [where_l[i:i + depth_r] for i in range(0, len(where_l), depth_r)]


    #     site_blocks = site_blocks_(block_l)

    #     with tqdm(total=len(gate_l_),  desc="gate:",  
    #                 leave=True, position=0, 
    #                 colour='MAGENTA') as pbar:

    #         for count in range(len(gate_l_)):
    #             psi.normalize()
    #             dmrg_run = dmrg_run_(where_l_[count], site_blocks)
    #             for count_, G_ in enumerate(gate_l_[count]):
    #                 mpo_local = mpo_from_gate(G_, where_l_[count][count_], sum(block_l),  
    #                                         dtype = dtype, cutoff=cutoff, 
    #                                         max_bond_mpo = max_bond_dmrg, 
    #                                         style = form,
    #                                           ) 
    #                 mpo_local = mpo_to_bmpo(mpo_local, block_l, inds_fuse, len(block_l), opt)
    #                 mpo_local = tn_T(mpo_local, mpo_local.L)

    #                 psi = mpo_local.apply(psi, 
    #                                     compress=False, 
    #                                     form=form, 
    #                                     max_bond=None,  #max_bond_dmrg, 
    #                                     cutoff=cutoff,
    #                                      )
                    
                        
    #             if dmrg_run:
    #                 psi_fix = psi.copy()
    #                 psi.compress(form, max_bond=bond_dim//2, cutoff=cutoff)
    #                 psi = iregular_bnd(psi, bnds, rand_strength=rand_strength)
    #                 psi.normalize()
    #                 psi, cost = mps_dmrg(psi, psi_fix, opt, 
    #                                     n_iter=n_iter_dmrg,  
    #                                     threshold = threshold)

    #                 error_.append( cost )
    #             else:
    #                 error_.append( 1. )

                
    #             avg_F = np.prod(error_) ** (1/len(error_))
    #             pbar.n = count + 1
    #             #pbar.set_postfix({'avg_F':avg_F})
    #             pbar.refresh()
    #             pbar.set_description(f"avg_F: {avg_F}")

def rho_site(p, site, opt, rescale_sites=True):
    ph = p.H
    ph.reindex_({f"k{site}":f"b{site}"})
    rho = (p & ph).contract(all, optimize=opt)
    if rescale_sites:
        rho.reindex_({f"k{site}":f"k{0}"})
        rho.reindex_({f"b{site}":f"b{0}"})
    return rho

def error_density_local(p, p_, L, opt):

    error_ = []
    for i in range(L):
        rho = rho_site(p, i, opt)
        rho_ = rho_site(p_, i, opt)
        error = qu.trace_distance(rho.data, rho_.data)
        error_.append(1-error)
    return error_

def cal_time( H, t, p_0, x_mpo):
    U = qu.expm(H * t * complex(0.,-1.) , herm=False)
    x_mpo_ = x_mpo.to_dense()
    p_0_ = p_0.to_dense()
    gs = U @ p_0_
    z_0_val = gs.conj().T @ x_mpo_ @ gs
    return z_0_val[0,0]



def energy_local(MPO_origin, mps_a, L, opt):
    mps_a_ = mps_a.copy()
    mps_a_.normalize()
    p_h=mps_a_.H 
    MPO_t = MPO_origin *1.0
    mps_a_.align_(MPO_t, p_h)
    E_dmrg = (p_h & MPO_t & mps_a_).contract(all,optimize=opt)
    return E_dmrg / L

def energy_global(MPO_origin, mps_a):
    _, opt, opt_= req_backend()
    mps_a_ = mps_a.copy()
    mps_a_.normalize()
    p_h=mps_a_.H 
    MPO_t = MPO_origin *1.0
    mps_a_.align_(MPO_t, p_h)
    E_dmrg = (p_h & MPO_t & mps_a_).contract(all,optimize=opt)
    return E_dmrg 

def auto_diff_tangentmps(tangent_mps, L, MPO_origin, tags_, opt, type = "time"):
    
    mps_a, mps_b_l = tangent_mps_list(tangent_mps, L)
    
    if type == "time":
        print(type)
        MPO_origin_ = MPO_origin * complex(0.,1.)
    if type == "energy":
        print(type)
        MPO_origin_ = MPO_origin * complex(-1.,0.)

    mps_a_H = MPO_origin_.apply(mps_a, form="left", compress=True,  max_bond=128, cutoff=1e-16)
    value_0 = mps_a_H.norm()
    optimizer = qtn.TNOptimizer(
        tangent_mps,                                # our initial input, the tensors of which to optimize
        loss_fn=loss_,
        loss_constants={'mps_a_H': mps_a_H},  # additional tensor/tn kwargs
        loss_kwargs={'opt': opt, "L":L, "value_0": value_0},    
        autodiff_backend = "autograd", #tensorflow,"torch", #'autograd',      # {'jax', 'tensorflow', 'autograd'}
        optimizer = "L-BFGS-B",  #'L-BFGS-B',               # supplied to scipy.minimize
        tags=tags_,
        shared_tags=[],
        device = "cpu"
    )
    return optimizer


def norm_tangent_mps_eff(tangent_mps, opt, L):
    mps_a, mps_b_l = tangent_mps_list(tangent_mps, L)
    value = 0
    for i in range(len(mps_b_l)):
        for j in range(i+1, len(mps_b_l), 1):
            val = (mps_b_l[i].H & mps_b_l[j]).contract(all, optimize=opt)
            value += val 
    value = value + autoray.do('conj', value) 
    for i in range(len(mps_b_l)):
            value += (mps_b_l[i].H & mps_b_l[i]).contract(all, optimize=opt)
    return value


def norm_tangent_mps(tangent_mps, opt, L):
    mps_a, mps_b_l = tangent_mps_list(tangent_mps, L)
    value = 0
    for i in range(len(mps_b_l)):
        for j in range(len(mps_b_l)):
            val = (mps_b_l[i].H & mps_b_l[j]).contract(all, optimize=opt)
            value += val 
    return value


def overlap_tangent_mps_mps(tangent_mps, mps, opt, L):
        
    mps_a, mps_b_l = tangent_mps_list(tangent_mps, L)
    value = 0
    for i in range(len(mps_b_l)):
            value += (mps.H & mps_b_l[i]).contract(all, optimize=opt)
    return value

def normalize_tangent_mps(tangent_mps, opt, L):
    norm = norm_tangent_mps(tangent_mps, opt, L)
    mps_a, mps_b_l = tangent_mps_list(tangent_mps, L)
    tangent_mps.equalize_norms_(1.0)

def loss_(tangent_mps, mps_a_H, L, opt, value_0):
    
    value_1 = norm_tangent_mps(tangent_mps, opt, L)
    value_2 = overlap_tangent_mps_mps(tangent_mps, mps_a_H, opt, L)
    value_2_ = autoray.do('conj',  value_2)
    
    #print(value_0 , value_1 , value_2 , value_2_)
    cost =  abs( value_0 + value_1 - value_2 - value_2_ )
    #cost = 1 - ( abs(value_2) / ((abs(value_1) * abs(value_0))**(0.5)) )
    return cost



def tangent_mps_init(mps_a, chi, rand_strength_ = 0.01):
    mps_a.expand_bond_dimension_(chi, rand_strength=0.0)
    mps_a = mps_a.astype("complex128")
    b_ten_ = []
    a_ten_ = [ t for t in mps_a]
    count = 0
    for t in a_ten_:
        t = t.copy()
        #t = t.randomize()
        t = t + t.randomize() * rand_strength_
        t.add_tag(f"b{count}")
        b_ten_.append(t)
        count += 1
    tangent_mps = qtn.TensorNetwork(a_ten_ + b_ten_)
    return tangent_mps



def update_linear(tangent_mps, delta_t, L):
    mps_a, mps_b_l = tangent_mps_list(tangent_mps, L)
    a_ten_ = []
    for count, t in enumerate(mps_a):
        t_ = t.copy()
        data_new = t.data + mps_b_l[count][count].data * delta_t
        t_.modify( data = data_new )
        a_ten_.append(t_)

    l_t = []
    for count, t in enumerate(a_ten_): 
            l_t.append(t)
    
    mps = qtn.TensorNetwork(l_t[:L])
    mps.view_as_(qtn.MatrixProductState, L = L, site_tag_id='I{}',site_ind_id='k{}',cyclic=False)
    mps.normalize()
    
    a_ten_ = []
    for count, t in enumerate(mps):
        a_ten_.append(t)
    
    b_ten_ = []
    count = 0
    for t in a_ten_:
        # t = t.randomize()
        t_ = t.copy()
        t_.add_tag(f"b{count}")
        b_ten_.append(t_)
        count += 1
    
    tangent_mps_ = qtn.TensorNetwork(a_ten_ + b_ten_)
    return tangent_mps_



def update_Heun(mps_a, mps_b_l, mps_b_l_, delta_t, L):
    a_ten_ = []
    b_ten_ = []
    for count, t in enumerate(mps_a):
        t_ = t.copy()
        #print(count, t.shape, mps_b_l[count][count].shape, )
        data_new = t.data + ( mps_b_l[count][count].data + mps_b_l_[count][count].data) * (delta_t/2)
        t_.modify( data = data_new)
        a_ten_.append(t_)

    l_t = []
    for count, t in enumerate(a_ten_): 
            l_t.append(t)
    
    mps = qtn.TensorNetwork(l_t[:L])
    mps.view_as_(qtn.MatrixProductState, L = L, site_tag_id='I{}',site_ind_id='k{}',cyclic=False)
    mps.normalize()
    
    a_ten_ = []
    for count, t in enumerate(mps):
        a_ten_.append(t)
    
    count = 0
    for t in a_ten_:
        # t = t.randomize()
        t_ = t.copy()
        t_.add_tag(f"b{count}")
        b_ten_.append(t_)
        count += 1
    
    tangent_mps_ = qtn.TensorNetwork(a_ten_ + b_ten_)
    return tangent_mps_


def tangent_mps_list(tangent_mps, L):
    mps_b_l = []
    l_t = []
    tags_ = [ f"b{i}" for i in range(L) ] 

    for count, t in enumerate(tangent_mps): 
        l_t.append(t)
    
    mps_a = qtn.TensorNetwork(l_t[:L])
    mps_a.view_as_(qtn.MatrixProductState, L = L, site_tag_id='I{}',site_ind_id='k{}',cyclic=False)
    
    # print("mps_a", mps_a.tags)
    
    mps_b_l = []
    for i in range(L, 2*L, 1):
        l_t_ = l_t[:L]
        l_t_[i-L] = l_t[i]  
        mps_b = qtn.TensorNetwork(l_t_[:L])
        mps_b.view_as_(qtn.MatrixProductState,L=L, site_tag_id='I{}',site_ind_id='k{}',cyclic=False)
        mps_b_l.append(mps_b)
        # print("mps_b", mps_b.tags)

    return mps_a, mps_b_l

def get_optimizer_exact(
    target_size=2**27,
    minimize="combo",
    max_time="rate:1e8",
    directory=True,
    progbar=False,
    max_repeats = 2**9,
    **kwargs,
):
    import cotengra as ctg

    if "parallel" not in kwargs:
        if "OMP_NUM_THREADS" in os.environ:
            parallel = int(os.environ["OMP_NUM_THREADS"])
        else:
            import multiprocessing as mp

            parallel = mp.cpu_count()

        if parallel == 1:
            parallel = False

        kwargs["parallel"] = parallel

    if target_size is not None:
        kwargs["slicing_reconf_opts"] = dict(target_size=target_size)
    else:
        kwargs["reconf_opts"] = {}


    return ctg.ReusableHyperOptimizer(
        progbar=progbar,
        minimize=minimize,
        max_time=max_time,
        #max_repeats = max_repeats,
        directory="../cash/",
        **kwargs,
    )


def opt_contraction_path(progbar=False, 
                        max_repeats=2**8,
                        max_repeats_=2**8,
                        parallel=True, 
                        target_size = 2**33, 
                        alpha = "combo",  
                        optlib="optuna", 
                        auto_=False,
                        max_time = "rate:1e8",
                        subtree_size = 4, bnd = 12,
                        ):
    
    #print("target_size", target_size)
    opt = ctg.ReusableHyperOptimizer(
        # do extra runs
        max_repeats=max_repeats,
        minimize=alpha,
        #minimize='flops',
        # use dynamic slicing to target a width of 30
        slicing_reconf_opts={'target_size': target_size},  # then advanced slicing with reconfiguring
        reconf_opts={'subtree_size': subtree_size},            # then finally just higher quality reconfiguring
        # use the nevergrad space searcher - good with large trial budget
        #optlib=optlib,
        # terminate search if no change for 128 trials
        max_time=max_time,
        parallel=parallel,
        # show live progress
        progbar=progbar,
        directory="cash/",
    )
    
    if auto_:
        opt = 'auto-hq'
    
    
    # copt = ctg.ReusableHyperCompressedOptimizer(
    #     bnd,
    #     max_repeats=max_repeats_,
    #     methods=  ('greedy-compressed', 'greedy-span', 'kahypar-agglom'),
    #     minimize='combo-compressed', 
    #     progbar=progbar,
    #     parallel=parallel,
    #     directory="cash/",
    # )

    
    return opt




def  Exact_time(L, MPO_origin, time_trotter, p_ex0, MPO_Z, MPO_X):

    H_ITF=(qtn.TensorNetwork(MPO_origin))^all
    listk=[]
    listb=[]
    for i_ite in range(L):
        listk.append(f'k{i_ite}')
        listb.append(f'b{i_ite}')


    list_f=listb+listk
    H_ITF=H_ITF.transpose( *listb+listk)


    A=qu.expm(complex(0.,-1.)*time_trotter*H_ITF.data.reshape(2**L,2**L), herm=False)
    #print ( "\n", "new", A, "\n", "new" )

    A_tensor=qtn.Tensor(A.reshape((2,)*(2*L)),list_f)
    p_ex=(A_tensor & qtn.TensorNetwork(p_ex0))^all


    p_ex.transpose(*listb)
    p_ex.modify(inds=listk)
    p_ex_H=p_ex.H
    p_ex_H.modify(inds=listb)

    E_0=(( p_ex & MPO_origin & p_ex_H )^all)


    #p_ex=p_ex*(N_0**(-0.5))

    #print( "ED=", E_0 , ( p_ex & MPO_Z & p_ex_H)^all, ( p_ex & MPO_X & p_ex_H)^all )
    return ( p_ex & MPO_Z & p_ex_H)^all


def exact_evolution(num_of_qubits: int, Time: float, J:float, h:float):
    N=num_of_qubits
    Jt=J*Time
    gf=h/J
    small = 1.e-8
    def epsilon(g: float, k:float):
        return 2*(g-np.cos(k))
    def gamma(k:float):
        return 2*np.sin(k)
    def En(g:float, k:float):
        return np.sqrt(epsilon(g,k)**2+gamma(k)**2)
    return sum(list((epsilon(gf,k)**2 + gamma(k)**2 * np.cos(2*En(gf,k)*Jt))/En(gf,k)**2 for k in np.linspace(-np.pi+(np.pi/N), np.pi-(np.pi/N), num=N)+small))/N




def  mpo_x(L, dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left"):

    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=dtype)
    #  Y = Y.astype(dtype)
    #  X = X.astype(dtype)
    #  Z = Z.astype(dtype)

    MPO_I=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=MPO_result*0.0
    MPO_f=MPO_result*0.0

    max_bond_val=chi
    cutoff=cutoff

    Ham = [X]
    for count, elem in enumerate (Ham):
        for i in range(L):
            #print("mpo_info_X", i, MPO_result.max_bond())
            Wl = np.zeros([ 1, 2, 2], dtype=dtype)
            W = np.zeros([1, 1, 2, 2], dtype=dtype)
            Wr = np.zeros([ 1, 2, 2], dtype=dtype)

            Wl[ 0,:,:]=elem
            W[ 0,0,:,:]=elem
            Wr[ 0,:,:]=elem
            W_list=[Wl]+[W]*(L-2)+[Wr]
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[i].modify(data=W_list[i])
            MPO_result=MPO_result+MPO_I

            #MPO_result += MPO_I
            MPO_result.compress(  style_1, max_bond=max_bond_val, cutoff=cutoff )

    MPO_result.compress(  style_1,max_bond=max_bond_val, cutoff=cutoff )

    return  MPO_result 



    

def mpo_z_IBM(L, Lx, Ly, dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left"):


    single_site = [ ((i,j)) for i,j in itertools.product([0,4,8,12],[1,5,9]) ] + [ ((i,j)) for i,j in itertools.product([2,6,10,14],[3,7,11]) ] 
    single_site += [ (i,0)  for i in range(0,Lx-1,1)]
    single_site += [ (i,2)    for i in range(0,Lx,1)]
    single_site += [ (i,4)    for i in range(0,Lx,1)]
    single_site += [ (i,6)    for i in range(0,Lx,1)]
    single_site += [ (i,8)   for i in range(0,Lx,1)]
    single_site += [ (i,10)    for i in range(0,Lx,1)]
    single_site += [ (i,12)   for i in range(1,Lx,1)]
    
    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=dtype)
    #  Y = Y.astype(dtype)
    #  X = X.astype(dtype)
    #  Z = Z.astype(dtype)

    MPO_I=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=MPO_result*0.0
    MPO_f=MPO_result*0.0

    max_bond_val=chi
    cutoff=cutoff

    Ham = [Z]
    for count, elem in enumerate(Ham):
        for where in tqdm(single_site):
            x, y = where
            i = x*Ly +y
            #print("mpo_info_X", i, MPO_result.max_bond())
            Wl = np.zeros([ 1, 2, 2], dtype=dtype)
            W = np.zeros([1, 1, 2, 2], dtype=dtype)
            Wr = np.zeros([ 1, 2, 2], dtype=dtype)

            Wl[ 0,:,:]=elem
            W[ 0,0,:,:]=elem
            Wr[ 0,:,:]=elem
            W_list=[Wl]+[W]*(L-2)+[Wr]
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[i].modify(data=W_list[i])
            MPO_result=MPO_result+MPO_I

            #MPO_result += MPO_I
            MPO_result.compress( style_1, max_bond=max_bond_val, cutoff=cutoff )

    MPO_result.compress(  style_1,max_bond=max_bond_val, cutoff=cutoff )
    MPO_result.add_tag("mpo_tag", where=None, which='all')


    return  MPO_result 


def mpo_z_IBM_rotated(L, Lx, Ly, dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left"):


    single_site = [ ((i,j)) for i,j in itertools.product([0,4,8,12],[1,5,9]) ] + [ ((i,j)) for i,j in itertools.product([2,6,10,14],[3,7,11]) ] 
    single_site += [ (i,0)  for i in range(0,Lx-1,1)]
    single_site += [ (i,2)    for i in range(0,Lx,1)]
    single_site += [ (i,4)    for i in range(0,Lx,1)]
    single_site += [ (i,6)    for i in range(0,Lx,1)]
    single_site += [ (i,8)   for i in range(0,Lx,1)]
    single_site += [ (i,10)    for i in range(0,Lx,1)]
    single_site += [ (i,12)   for i in range(1,Lx,1)]
    
    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=dtype)
    #  Y = Y.astype(dtype)
    #  X = X.astype(dtype)
    #  Z = Z.astype(dtype)

    MPO_I=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=MPO_result*0.0
    MPO_f=MPO_result*0.0

    max_bond_val=chi
    cutoff=cutoff

    Ham = [Z]
    for count, elem in enumerate(Ham):
        for where in tqdm(single_site):
            x, y = where
            i = y*Lx +x
            #print("mpo_info_X", i, MPO_result.max_bond())
            Wl = np.zeros([ 1, 2, 2], dtype=dtype)
            W = np.zeros([1, 1, 2, 2], dtype=dtype)
            Wr = np.zeros([ 1, 2, 2], dtype=dtype)

            Wl[ 0,:,:]=elem
            W[ 0,0,:,:]=elem
            Wr[ 0,:,:]=elem
            W_list=[Wl]+[W]*(L-2)+[Wr]
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[i].modify(data=W_list[i])
            MPO_result=MPO_result+MPO_I

            #MPO_result += MPO_I
            MPO_result.compress( style_1, max_bond=max_bond_val, cutoff=cutoff )

    MPO_result.compress(  style_1,max_bond=max_bond_val, cutoff=cutoff )
    MPO_result.add_tag("mpo_tag", where=None, which='all')

    return  MPO_result 

def mpo_z(L, pauli, dtype="float64", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left"):

    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=dtype)
    #  Y = Y.astype(dtype)
    #  X = X.astype(dtype)
    #  Z = Z.astype(dtype)

    if pauli=="X":
        Z = X

    MPO_I=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=MPO_result*0.0
    MPO_f=MPO_result*0.0

    max_bond_val=chi
    cutoff=cutoff

    Ham = [Z]
    for count, elem in enumerate (Ham):
        for i in range(L):
            #print("mpo_info_X", i, MPO_result.max_bond())
            Wl = np.zeros([ 1, 2, 2], dtype=dtype)
            W = np.zeros([1, 1, 2, 2], dtype=dtype)
            Wr = np.zeros([ 1, 2, 2], dtype=dtype)

            Wl[ 0,:,:]=elem
            W[ 0,0,:,:]=elem
            Wr[ 0,:,:]=elem
            W_list=[Wl]+[W]*(L-2)+[Wr]
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[i].modify(data=W_list[i])
            MPO_result=MPO_result+MPO_I

            #MPO_result += MPO_I
            MPO_result.compress(  style_1, max_bond=max_bond_val, cutoff=cutoff )

    MPO_result.compress(  style_1,max_bond=max_bond_val, cutoff=cutoff )
    MPO_result.add_tag("mpo_tag", where=None, which='all')
    
    for i in range(L):
        t = MPO_result[f"I{i}"]
        t.add_tag(f"O{i}")
    return  MPO_result * (1/L) 




def mpo_z_select(L,l_, pauli,dtype="complex128", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left"):

    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=dtype)
    #  Y = Y.astype(dtype)
    #  X = X.astype(dtype)
    #  Z = Z.astype(dtype)
    if pauli=="X":
        Z = X
    if pauli=="Y":
        Z = Y
    if pauli=="Z":
        Z = Z
    
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    MPO_result = MPO_result * 0.0
    for i in l_:
        MPO_I=qtn.MPO_identity(L, phys_dim=2)
        Wl = np.zeros([ 1, 2, 2], dtype=dtype)
        W = np.zeros([1, 1, 2, 2], dtype=dtype)
        Wr = np.zeros([ 1, 2, 2], dtype=dtype)

        Wl[ 0,:,:] = Z
        W[ 0,0,:,:] = Z
        Wr[ 0,:,:] = Z
        W_list=[Wl]+[W]*(L-2)+[Wr]
        
        MPO_I[i].modify(data=W_list[i])
        MPO_result= MPO_I + MPO_result
        MPO_result.compress(  style_1, max_bond=chi, cutoff=cutoff )

    return  MPO_result/len(l_)


def mpo_zz_center(L, pauli, where_=0, dtype="complex128", chi=200, cutoff=1.0e-12, style_1= "left", style_2= "left"):

    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=dtype)
    #  Y = Y.astype(dtype)
    #  X = X.astype(dtype)
    #  Z = Z.astype(dtype)
    if pauli=="X":
        Z = X
    if pauli=="Y":
        Z = Y
    if pauli=="Z":
        Z = Z
    
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    for i in range(L):
        MPO_I=qtn.MPO_identity(L, phys_dim=2)
        Wl = np.zeros([ 1, 2, 2], dtype=dtype)
        W = np.zeros([1, 1, 2, 2], dtype=dtype)
        Wr = np.zeros([ 1, 2, 2], dtype=dtype)

        Wl[ 0,:,:] = Z
        W[ 0,0,:,:] = Z
        Wr[ 0,:,:] = Z
        W_list=[Wl]+[W]*(L-2)+[Wr]
        if i != where_:
            MPO_I[i].modify(data=W_list[i])
            MPO_I[where_].modify(data=W_list[where_])
            MPO_result= MPO_I + MPO_result
        MPO_result.compress(  style_1, max_bond=chi, cutoff=cutoff )

    return  MPO_result/L


def mpo_z_prod(L, pauli, where_=None, dtype="complex128", chi=200, 
                 cutoff=1.0e-12, style_1= "left", style_2= "left"):

    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = qu.pauli('Y',dtype=dtype) 
    I = qu.pauli('I',dtype=dtype)
    #  Y = Y.astype(dtype)
    #  X = X.astype(dtype)
    #  Z = Z.astype(dtype)

    if pauli=="X":
        Z = X
    elif pauli=="Y":
        Z = Y


    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    
    Wl = np.zeros([ 1, 2, 2], dtype=dtype)
    W = np.zeros([1, 1, 2, 2], dtype=dtype)
    Wr = np.zeros([ 1, 2, 2], dtype=dtype)
    Wl[ 0,:,:] = Z
    W[ 0,0,:,:] = Z
    Wr[ 0,:,:] = Z
    W_list=[Wl]+[W]*(L-2)+[Wr]

    for cor in where_:

        MPO_result[cor].modify(data=W_list[cor])

    return  MPO_result




def mpo_local(L, site_, pauli=None, dtype="complex128", style_1= "left", style_2= "left"):
     
    MPO_I=qtn.MPO_identity(L, phys_dim=2)
    for count, site in enumerate(site_):
        Wl = np.zeros([ 1, 2, 2], dtype=dtype)
        W = np.zeros([1, 1, 2, 2], dtype=dtype)
        Wr = np.zeros([ 1, 2, 2], dtype=dtype)
        Wl[ 0,:,:] = pauli[count]
        W[ 0,0,:,:] = pauli[count]
        Wr[ 0,:,:] = pauli[count]
        W_list=[Wl]+[W]*(L-2)+[Wr]
        MPO_I[site].modify(data=W_list[site])
        MPO_I[site].add_tag(f"O{site}")
        MPO_I.add_tag("mpo_tag", where=None, which='all')
    MPO_I.astype_(dtype)

    return  MPO_I  

#@profile
def pepo_local(Lx, Ly, site_, Z_, dtype="complex128"):
     
    pepo_I = pepo_identity(Lx, Ly, dtype=dtype)
    
    for count, site in enumerate(site_):
        Z = Z_[count]
        t = pepo_I[site] 
        shape = t.shape
        t.modify(data = Z.reshape(shape))
        t.add_tag(f"O_")
        pepo_I.add_tag("pepo_tag", where=None, which='all')
    pepo_I.astype_(dtype)

    return  pepo_I  

















def mpo_2d_ITF(Lx, Ly, Lz, L, J=1., h=1., chi=300, dtype="float64", cutoff=1.0e-12,style_1= "left", style_2= "left", cycle = "open"):
    Wl = np.zeros([ 1, 2, 2], dtype=dtype)
    W = np.zeros([1, 1, 2, 2], dtype=dtype)
    Wr = np.zeros([ 1, 2, 2], dtype=dtype)

    Z=qu.pauli('Z',dtype=dtype) 
    X=qu.pauli('X',dtype=dtype) 
    Y=qu.pauli('Y',dtype=dtype)
    I=qu.pauli('I',dtype=dtype)
    Y=Y.astype(dtype)
    X=X.astype(dtype)
    Z=Z.astype(dtype)
    terms_x = {}
    terms_y = {}
    if cycle == "open":
        terms_x = {  (i*Ly*Lz + j*Lz + k, (i+1)*Ly*Lz + j*Lz + k) : Z  for i, j, k in itertools.product(range(Lx-1), range(Ly), range(Lz))}
        terms_y = {  (i*Ly*Lz + j*Lz + k, i*Ly*Lz + (j+1)*Lz + k) : Z  for i, j, k in itertools.product(range(Lx), range(Ly-1), range(Lz))}
        terms_z = {  (i*Ly*Lz + j*Lz + k, i*Ly*Lz + j*Lz + k+1) : Z  for i, j, k in itertools.product(range(Lx), range(Ly), range(Lz-1))}
        terms_h = {    (i, ) : X             for i in range(L) }
        terms = terms_x |  terms_y | terms_z | terms_h
    elif cycle == "periodic":
        terms_x = {}
        terms_z = {}
        terms_y = {}        
        if Lx > 1:
            terms_x = { (i*Ly*Lz + j*Lz + k, ((i+1)%Lx)*Ly*Lz + j*Lz + k) : Z  for i, j, k in itertools.product(range(Lx), range(Ly), range(Lz)) }
        if Ly > 1:
            terms_y = {  (i*Ly*Lz + j*Lz + k, i*Ly*Lz + ((j+1)%Ly)*Lz + k) : Z  for i, j, k in itertools.product(range(Lx), range(Ly), range(Lz))}
        if Lz > 1:
            terms_z = {  (i*Ly*Lz + j*Lz + k, i*Ly*Lz + j*Lz + (k+1)%Lz ) : Z  for i, j, k in itertools.product(range(Lx), range(Ly), range(Lz))}
        
        terms_h = {    (i, ) : X             for i in range(L) }
        
        terms = terms_x |  terms_y | terms_z | terms_h
    

    count_ = 0
    for i in tqdm(terms):        
        
        elem = terms[i]
        Wl[ 0, :, :]  = elem
        W[ 0,0, :, :] = elem
        Wr[ 0, :, :]  = elem
        W_list = [Wl] + [W]*(L-2)+ [Wr]

        if len(i) == 2:
            ii, ii_ = i
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[ii].modify(data=W_list[ii])
            MPO_I[ii_].modify(data=W_list[ii_] * J )
        else :
            ii, = i
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[ii].modify(data=W_list[ii] * h)

        if count_ == 0:
            MPO_H = MPO_I
        else:
            MPO_H = MPO_H + MPO_I
            MPO_H.compress( style_1, max_bond=chi, cutoff=cutoff )
        count_ += 1
    return  MPO_H , terms




def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}
        
        

def  Exact_time(L, MPO_origin, time_trotter, p_ex0, MPO_Z, MPO_X):

    H_ITF=(qtn.TensorNetwork(MPO_origin))^all
    listk=[]
    listb=[]
    for i_ite in range(L):
        listk.append(f'k{i_ite}')
        listb.append(f'b{i_ite}')


    list_f=listb+listk
    H_ITF=H_ITF.transpose( *listb+listk)


    A=qu.expm(complex(0.,-1.)*time_trotter*H_ITF.data.reshape(2**L,2**L), herm=False)
    #print ( "\n", "new", A, "\n", "new" )

    A_tensor=qtn.Tensor(A.reshape((2,)*(2*L)),list_f)
    p_ex=(A_tensor & qtn.TensorNetwork(p_ex0))^all


    p_ex.transpose(*listb)
    p_ex.modify(inds=listk)
    p_ex_H=p_ex.H
    p_ex_H.modify(inds=listb)

    E_0=(( p_ex & MPO_origin & p_ex_H )^all)


    #p_ex=p_ex*(N_0**(-0.5))

    #print( "ED=", E_0 , ( p_ex & MPO_Z & p_ex_H)^all, ( p_ex & MPO_X & p_ex_H)^all )
    return ( p_ex & MPO_Z & p_ex_H)^all


def exact_evolution(num_of_qubits: int, Time: float, J:float, h:float):
    N=num_of_qubits
    Jt=J*Time
    gf=h/J
    small = 1.e-8
    def epsilon(g: float, k:float):
        return 2*(g-np.cos(k))
    def gamma(k:float):
        return 2*np.sin(k)
    def En(g:float, k:float):
        return np.sqrt(epsilon(g,k)**2+gamma(k)**2)
    return sum(list((epsilon(gf,k)**2 + gamma(k)**2 * np.cos(2*En(gf,k)*Jt))/En(gf,k)**2 for k in np.linspace(-np.pi+(np.pi/N), np.pi-(np.pi/N), num=N)+small))/N






def MPO_site(J_, h, L_inter, dtype="float64"):
    Z=qu.pauli('Z',dtype=dtype) 
    X=qu.pauli('X',dtype=dtype) 
    Y= np.array([[0, -1],[1,0]]) 
    I=qu.pauli('I',dtype=dtype)
    Y=Y.astype(dtype)
    X=X.astype(dtype)
    Z=Z.astype(dtype)
    

    inter_range = max(L_inter)
    
    wl_mpo_l = np.zeros([ inter_range+2, 2, 2], dtype = dtype) 
    wl_mpo_l[ 0, :, :] = I
    wl_mpo_r = np.zeros([ inter_range+2, 2, 2], dtype = dtype) 
    wl_mpo_r[ inter_range+1, :, :] = I

    A_wl = np.zeros([ inter_range, inter_range, 2, 2], dtype = dtype)
    for i in range(0, inter_range-1, 1):
        A_wl[ i, i+1,:, :] = I
        
    C_wl = np.zeros([ inter_range, 2, 2], dtype = dtype)
    C_wl[ 0,:, :] = +X
    D_wl = np.zeros([ 2, 2], dtype = dtype)
    D_wl[:, :] = h*Z
    B_wl = np.zeros([ inter_range, 2, 2], dtype = dtype)
    
    
    for count, item in enumerate(L_inter):
        B_wl[ item-1,:, :] = J_[count] * X


    wl_mpo = np.zeros([ inter_range+2, inter_range+2, 2, 2], dtype = dtype)
    wl_mpo[ 0, 0, :, :] = I
    wl_mpo[ inter_range+1, inter_range+1, :, :] = I    
    wl_mpo[1:inter_range+1, 1:inter_range+1, :, :] = A_wl
    wl_mpo[0, 1:inter_range+1,  :, :] = C_wl
    wl_mpo[ 1:inter_range+1, inter_range+1,  :, :] = B_wl
    wl_mpo[ 0, inter_range+1,  :, :] = D_wl    
    
    wl_mpo_lf = np.einsum('ilp, ijpk->jlk', wl_mpo_l, wl_mpo)
    wl_mpo_rf = np.einsum('ijpk, jkl->ipl', wl_mpo, wl_mpo_r)

    return wl_mpo, wl_mpo_lf, wl_mpo_rf


def linear_(p, Ly):
    value = 0 
    for count, i in enumerate(p):
        value += i*Ly**(len(p)-count-1)
    return value

def mpo_higher_order(wl_mpo, N, Ly):
    wl_mpo_ = wl_mpo * 1.0
    for i in range(1, N, 1):
        wl_mpo_ = np.einsum('imps, jnsq->ijmnpq', wl_mpo_, wl_mpo)
        #print("wl_mpo_.shape", wl_mpo_.shape)
        wl_mpo_ = wl_mpo_.reshape(Ly**(i+1),Ly**(i+1), 2, 2)
    
    return wl_mpo_



def MPO_site_t(J_, h, L_inter, N_order, delta, dtype="float64", order_type="f"):
    Z=qu.pauli('Z',dtype=dtype) 
    X=qu.pauli('X',dtype=dtype) 
    Y= np.array([[0, -1],[1,0]]) 
    I=qu.pauli('I',dtype=dtype)
    Y=Y.astype(dtype)
    X=X.astype(dtype)
    Z=Z.astype(dtype)

    Z=qu.pauli('Z',dtype=dtype) 
    X=qu.pauli('X',dtype=dtype) 
    Y= np.array([[0, -1],[1,0]]) 
    I=qu.pauli('I',dtype=dtype)
    Y=Y.astype(dtype)
    X=X.astype(dtype)
    Z=Z.astype(dtype)
    
    inter_range = max(L_inter)
    bond_mpo = inter_range + 2
    wl_mpo = np.zeros([ inter_range+2, inter_range+2, 2, 2], dtype = dtype)
    

    A_wl = np.zeros([ inter_range, inter_range, 2, 2], dtype = dtype)
    for i in range(0, inter_range-1, 1):
        A_wl[ i, i+1,:, :] = I
        
    C_wl = np.zeros([ inter_range, 2, 2], dtype = dtype)
    C_wl[ 0,:, :] = +X
    D_wl = np.zeros([ 2, 2], dtype = dtype)
    D_wl[:, :] = h*Z
    B_wl = np.zeros([ inter_range, 2, 2], dtype = dtype)
    for count, item in enumerate(L_inter):
        B_wl[ item-1,:, :] = J_[count] * X


    wl_mpo[ 0, 0, :, :] = I
    wl_mpo[ inter_range+1, inter_range+1, :, :] = I    
    wl_mpo[1:inter_range+1, 1:inter_range+1, :, :] = A_wl
    wl_mpo[0, 1:inter_range+1,  :, :] = C_wl
    wl_mpo[ 1:inter_range+1, inter_range+1,  :, :] = B_wl
    wl_mpo[ 0, inter_range+1,  :, :] = D_wl    
    
    #print(bond_mpo, wl_mpo)
    if order_type == "order" : 
        #wl_mpo_2 = np.einsum('imps, jnsq->ijmnpq', wl_mpo, wl_mpo)
        N = N_order
        wl_mpo_h = mpo_higher_order(wl_mpo, N, bond_mpo)
        #print("wl_mpo_h", wl_mpo_h[0,:,:,:])

        #print("wl_mpo_h, (0,0)", wl_mpo_h[0,0,:,:])
        #print("wl_mpo_h, (0,4)", wl_mpo_h[0,4,:,:])
        #print("wl_mpo_h, (0,8)", wl_mpo_h[0,8,:,:])
        #print("wl_mpo_h, (3,8)", wl_mpo_h[3,8,:,:])
        eliminate_list = []
        for a in range(1, N+1, 1):
            p = [0] * N
            for replace_ in range(0, a):
                p[replace_] = bond_mpo -1
            #print("a, p", a, p)
            #for i_ in permutations( p ):
            for i_ in distinct_permutations( p ):    
                #print("permutation_p", i_, "linear_(p, bond_mpo)", linear_(i_, bond_mpo), "tau", (delta**(a)) * math.factorial(N-a) * (1/ math.factorial(N)))
                #print( wl_mpo_h[0,2,:,:], (delta**(a)) * math.factorial(N-a) * (1/ math.factorial(N)) )
                wl_mpo_h[:,0,:,:] = wl_mpo_h[:,0,:,:] + wl_mpo_h[:,linear_(i_, bond_mpo),:,:] * (delta**(a)) * math.factorial(N-a) * (1/ math.factorial(N))
                #print(i_, "dynamicwl_mpo_h[0][0]", wl_mpo_h[0,0,:,:])
                eliminate_list.append(linear_(i_, bond_mpo))
        
        wl_mpo_h = np.delete(wl_mpo_h, eliminate_list, axis=0)
        wl_mpo_h = np.delete(wl_mpo_h, eliminate_list, axis=1)
        
        
        #print("wl_mpo_h, final", wl_mpo_h.shape)
        #print("wl_mpo_h", wl_mpo_h)
        #print("wl_mpo_h, (1,0)", wl_mpo_h[0,1,:,:])
        #print("wl_mpo_h, (2,0)", wl_mpo_h[1,0,:,:])
        #print("wl_mpo_h, (1,1)", wl_mpo_h[1,1,:,:])

        wl_mpo_t = wl_mpo_h * 1.0
        x, y, p, q= wl_mpo_t.shape
        #print("x,y", x, y)
        
        wl_mpo_l = np.zeros([ x, 2, 2], dtype = dtype) 
        wl_mpo_l[ 0, :, :] = I
        wl_mpo_r = np.zeros([ y, 2, 2], dtype = dtype) 
        wl_mpo_r[ 0, :, :] = I
        
        wl_mpo_lf_t = np.einsum('ilp, ijpk->jlk', wl_mpo_l, wl_mpo_t)
        wl_mpo_rf_t = np.einsum('ijpk, jkl->ipl', wl_mpo_t, wl_mpo_r)

            
    if order_type=="fo" : 
        wl_mpo_l = np.zeros([ inter_range+1, 2, 2], dtype = dtype) 
        wl_mpo_l[ 0, :, :] = I
        wl_mpo_r = np.zeros([ inter_range+1, 2, 2], dtype = dtype) 
        wl_mpo_r[ 0, :, :] = I

        wl_mpo_t = np.zeros([ inter_range+1,inter_range+1, 2, 2], dtype = dtype)
        DD_wl = np.einsum('fp, pq->fq', D_wl, D_wl)
        wl_mpo_t[0, 0, :, :] =  I + D_wl * delta + DD_wl * ( ((delta)**2) / 2. )
        #print("0,0", I + D_wl * delta + DD_wl * ( ((delta)**2) / 2. ))
        
        
        CD_wl = np.einsum('jpq, qf->jpf', C_wl, D_wl)
        DC_wl = np.einsum('qf, jfp->jqp', D_wl, C_wl)
        #print("0,1", C_wl + ( CD_wl + DC_wl ) * (delta/2.))
        wl_mpo_t[0, 1:inter_range+1,  :, :] = C_wl + ( CD_wl + DC_wl ) * (delta/2.)  
        
        
        BD_wl = np.einsum('jpq, qf->jpf', B_wl, D_wl)
        DB_wl = np.einsum('qf, jfp->jqp', D_wl, B_wl)
    
        AD_wl = np.einsum('ijpq, qf->ijpf', A_wl, D_wl)
        DA_wl = np.einsum('qf, ijfp->ijqp', D_wl, A_wl)
        
        BC_wl = np.einsum('ipq, jqf->ijpf', B_wl, C_wl)
        CB_wl = np.einsum('ipq, jqf->ijpf', C_wl, B_wl)
        
        #print("1,0", B_wl * delta + ( BD_wl + DB_wl ) *  ( (delta**(2))/2. ) )
        #print("1,1", A_wl + ( BC_wl + CB_wl + DA_wl + AD_wl ) * (delta/2.)  )

        wl_mpo_t[ 1:inter_range+1, 0,  :, :] = B_wl * delta + ( BD_wl + DB_wl ) *  ( (delta**(2))/2. )
        wl_mpo_t[ 1:inter_range+1, 1:inter_range+1,  :, :] = A_wl + ( BC_wl + CB_wl + DA_wl + AD_wl ) * (delta/2.)   

        wl_mpo_lf_t = np.einsum('ilp, ijpk->jlk', wl_mpo_l, wl_mpo_t)
        wl_mpo_rf_t = np.einsum('ijpk, jkl->ipl', wl_mpo_t, wl_mpo_r)
        # print("wl_mpo_t[0,0,:,:]", wl_mpo_t[0,0,:,:])
        # print("wl_mpo_t[0,1,:,:]", wl_mpo_t[0,1,:,:])
        # print("wl_mpo_t[1,0,:,:]", wl_mpo_t[1,0,:,:])
        # print("wl_mpo_t[1,1,:,:]", wl_mpo_t[1,1,:,:])
        # print("wl_mpo_l", wl_mpo_l)
        # print("wl_mpo_r", wl_mpo_r)

    elif order_type=="f" : 

        wl_mpo_l = np.zeros([ inter_range+1, 2, 2], dtype = dtype) 
        wl_mpo_l[ 0, :, :] = I
        wl_mpo_r = np.zeros([ inter_range+1, 2, 2], dtype = dtype) 
        wl_mpo_r[ 0, :, :] = I

        wl_mpo_t = np.zeros([ inter_range+1,inter_range+1, 2, 2], dtype = dtype)
        
        wl_mpo_t[0, 0, :, :] =  I + D_wl * delta 
        wl_mpo_t[0, 1:inter_range +1,  :, :] = C_wl 
        wl_mpo_t[ 1:inter_range +1, 0,  :, :] = B_wl * delta 
        wl_mpo_t[ 1:inter_range+1, 1:inter_range+1:,  :, :] = A_wl
        wl_mpo_lf_t = np.einsum('ilp, ijpk->jlk', wl_mpo_l, wl_mpo_t)
        wl_mpo_rf_t = np.einsum('ijpk, jkl->ipl', wl_mpo_t, wl_mpo_r)
        # print("wl_mpo_t[0,0,:,:]", wl_mpo_t[0,0,:,:])
        # print("wl_mpo_t[0,1,:,:]", wl_mpo_t[0,1,:,:])
        # print("wl_mpo_t[1,0,:,:]", wl_mpo_t[1,0,:,:])
        # print("wl_mpo_t[1,1,:,:]", wl_mpo_t[1,1,:,:])
        # print("wl_mpo_l", wl_mpo_l)
        # print("wl_mpo_r", wl_mpo_r)

        
    elif order_type=="so":
        wl_mpo_l = np.zeros([ 1+ inter_range + inter_range*inter_range, 2, 2], dtype = dtype) 
        wl_mpo_l[ 0, :, :] = I
        wl_mpo_r = np.zeros([ 1+ inter_range + inter_range*inter_range, 2, 2], dtype = dtype) 
        wl_mpo_r[ 0, :, :] = I

        wl_mpo_t = np.zeros([ 1+ inter_range + inter_range*inter_range, 1+ inter_range + inter_range*inter_range, 2, 2], dtype = dtype)
        DD_wl = np.einsum('fp, pq->fq', D_wl, D_wl)
        DDD_wl = np.einsum('fp, pq->fq', DD_wl, D_wl)
        #print("DDD_wl", DDD_wl, "DD_wl", DD_wl, "D_wl", D_wl)
        #M_t = I + D_wl * delta + DD_wl * ( ((delta)**2) / 2. ) + DDD_wl * ( ((delta)**3) / 6. )
        #print("0,0", M_t.shape, M_t)
        
        wl_mpo_t[0, 0, :, :] =  I + D_wl * delta + DD_wl * ( ((delta)**2) / 2. ) + DDD_wl * ( ((delta)**3) / 6. )
        
        CD_wl = np.einsum('jpq, qf->jpf', C_wl, D_wl)
        DC_wl = np.einsum('qf, jfp->jqp', D_wl, C_wl)
        CDD_wl = np.einsum('jpq, qf->jpf', CD_wl, D_wl)
        DCD_wl = np.einsum('jpq, qf->jpf', DC_wl, D_wl)
        DDC_wl = np.einsum('qf, jfp ->jqp', DD_wl, C_wl)
        
        #print("CDD_wl", CDD_wl, "DCD_wl", DCD_wl, "DDC_wl", DDC_wl)
        #print("CD_wl", CD_wl, "DC_wl", DC_wl)
        #print("0,1", C_wl + ( CD_wl + DC_wl ) * (delta/2.)  + (CDD_wl + DCD_wl + DDC_wl) *  ( ((delta)**2) / 6. ))
        
        wl_mpo_t[0, 1:inter_range+1,  :, :] = C_wl + ( CD_wl + DC_wl ) * (delta/2.)  + (CDD_wl + DCD_wl + DDC_wl) *  ( ((delta)**2) / 6. )
        

        CC_wl = np.einsum('jpq, mqf->jmpf', C_wl, C_wl)
        DCC_wl = np.einsum('qf, jmfp->jmqp', D_wl, CC_wl)
        CDC_wl = np.einsum('jpq, mqf->jmpf', CD_wl, C_wl)
        CCD_wl = np.einsum('jmpq, qf->jmpf', CC_wl, D_wl)
        #print("inter_range", inter_range)
        #print("CC_wl_shape",CC_wl.shape, "CC_wl", CC_wl)
        #print("DCC_wl_shape", DCC_wl.shape, "DCC_wl", DCC_wl, "CDC_wl", CDC_wl, "CCD_wl", CCD_wl)
        
        CC_wl = CC_wl.reshape(inter_range*inter_range,2, 2)
        DCC_wl = DCC_wl.reshape(inter_range*inter_range,2, 2)
        CDC_wl = CDC_wl.reshape(inter_range*inter_range,2, 2)
        CCD_wl = CCD_wl.reshape(inter_range*inter_range,2, 2)
        #print("CC_wl_shape",CC_wl.shape, "CC_wl", CC_wl)
        #print("DCC_wl_shape", DCC_wl.shape, "DCC_wl", DCC_wl, "CDC_wl", CDC_wl, "CCD_wl", CCD_wl)
        #print("0,2", CC_wl + (DCC_wl + CDC_wl + CCD_wl) * ( delta / 3. ))

        wl_mpo_t[0, inter_range+1 : inter_range*inter_range + inter_range + 2,  :, :] = CC_wl + (DCC_wl + CDC_wl + CCD_wl) * ( delta / 3. )

        
        BD_wl = np.einsum('jpq, qf->jpf', B_wl, D_wl)
        DB_wl = np.einsum('qf, jfp->jqp', D_wl, B_wl)
        DDB_wl = np.einsum('qf, jfp->jqp', D_wl, DB_wl)
        BDD_wl = np.einsum('jpq, qf->jpf', B_wl, DD_wl)
        DBD_wl = np.einsum('jpq, qf->jpf', DB_wl, D_wl)

        #print("DDB_wl", DDB_wl, "BDD_wl", BDD_wl, "DBD_wl", DBD_wl)
        #print( "DB_wl", DB_wl, "BD_wl", BD_wl)
        #print( "B_wl * delta", B_wl * delta, B_wl)
        #print("1,0",B_wl * delta + ( BD_wl + DB_wl ) *  ( (delta**(2))/2. ) + ( DBD_wl + DDB_wl + BDD_wl ) * ( ((delta)**3) / 6. ))

        wl_mpo_t[ 1:inter_range+1, 0,  :, :] = B_wl * delta + ( BD_wl + DB_wl ) *  ( (delta**(2))/2. ) + ( DBD_wl + DDB_wl + BDD_wl ) * ( ((delta)**3) / 6. )
        
        
        AD_wl = np.einsum('ijpq, qf->ijpf', A_wl, D_wl)
        DA_wl = np.einsum('qf, ijfp->ijqp', D_wl, A_wl)
        BC_wl = np.einsum('ipq, jqf->ijpf', B_wl, C_wl)
        CB_wl = np.einsum('ipq, jqf->ijpf', C_wl, B_wl)

#        print("AD_wl", AD_wl, "BC_wl", BC_wl, "CB_wl", CB_wl)

        ADD_wl = np.einsum('ijpq, qf->ijpf', AD_wl, D_wl)
        DAD_wl = np.einsum('qf, ijfp->ijqp', D_wl, AD_wl)
        DDA_wl = np.einsum('qf, ijfp->ijqp', D_wl, DA_wl)

#        print("ADD_wl", ADD_wl, DAD_wl, DDA_wl)

        
        CBD_wl = np.einsum('ijpq, qf->ijpf', CB_wl, D_wl)
        CDB_wl = np.einsum('ipq, jqf->ijpf', CD_wl, B_wl)
        DBC_wl = np.einsum('qf, ijfp->ijqp', D_wl, BC_wl)
        DCB_wl = np.einsum('qf, ijfp->ijqp', D_wl, CB_wl)
        BCD_wl = np.einsum('ijpq, qf->ijpf', BC_wl, D_wl)
        BDC_wl = np.einsum('ipq, jqf->ijpf', BD_wl, C_wl)
  
        # print("CBD_wl", CBD_wl,"CDB_wl", CDB_wl )
        # print("DBC_wl", DBC_wl,"DCB_wl", DCB_wl )
        # print("BDC_wl", BDC_wl,"BCD_wl", BCD_wl )

        #print("1,1",A_wl + ( BC_wl + CB_wl + DA_wl + AD_wl ) * (delta/2.) +   (CBD_wl + CDB_wl + DBC_wl + DCB_wl + BCD_wl + BDC_wl) * ( ((delta)**2) / 6. ))

        wl_mpo_t[ 1:inter_range+1, 1:inter_range+1,  :, :] = A_wl + ( BC_wl + CB_wl + DA_wl + AD_wl ) * (delta/2.) +   (CBD_wl + CDB_wl + DBC_wl + DCB_wl + BCD_wl + BDC_wl) * ( ((delta)**2) / 6. )

        AC_wl = np.einsum('ijpq, mqf->ijmpf', A_wl, C_wl)        
        CA_wl = np.einsum('mqf, ijfp ->imjqp', C_wl, A_wl)
        AC_wl = AC_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        CA_wl = CA_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        
        ACD_wl = np.einsum('ijpq, mqf ->ijmpf', A_wl, CD_wl)
        ACD_wl=ACD_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        ADC_wl = np.einsum('ijpq, mqf ->ijmpf', A_wl, DC_wl)
        ADC_wl=ADC_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        CAD_wl = np.einsum('mqf, ijfp  ->imjqp', C_wl, AD_wl)
        CAD_wl=CAD_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        CDA_wl = np.einsum('mqf, ijfp  ->imjqp', C_wl, DA_wl)
        CDA_wl=CDA_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        
        DAC_wl = np.einsum('qf, ijfp  ->ijqp', D_wl, AC_wl)
        DCA_wl = np.einsum('qf, ijfp  ->ijqp', D_wl, CA_wl)

        BCC_wl = np.einsum('mfp, ipq->mifq', B_wl, CC_wl)
        CBC_wl = np.einsum('mfp, ijpq->imjfq', C_wl, BC_wl)
        CBC_wl=CBC_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        CCB_wl = np.einsum('mfp, ijpq->imjfq', C_wl, CB_wl)
        CCB_wl = CCB_wl.reshape(inter_range, inter_range*inter_range,2, 2)
        #print("A_wl", A_wl)
        #print("BCC_wl=Z", BCC_wl, CBC_wl, CCB_wl )

        #print("1,2", AC_wl + CA_wl + (ACD_wl + ADC_wl + CAD_wl + CDA_wl + DAC_wl + DCA_wl + BCC_wl + CBC_wl + CCB_wl) * (delta / 3.))
        

        wl_mpo_t[ 1:inter_range+1, inter_range+1 : inter_range*inter_range + inter_range + 2,  :, :] = AC_wl + CA_wl + (ACD_wl + ADC_wl + CAD_wl + CDA_wl + DAC_wl + DCA_wl + BCC_wl + CBC_wl + CCB_wl) * (delta / 3.)
        

        BB_wl = np.einsum('jpq, mqf->jmpf', B_wl, B_wl)
        DBB_wl = np.einsum('qf, jmfp->jmqp', D_wl, BB_wl)
        BDB_wl = np.einsum('jpq, mqf->jmpf', BD_wl, B_wl)
        BBD_wl = np.einsum('jmpq, qf->jmpf', BB_wl, D_wl)
        DBB_wl = DBB_wl.reshape(inter_range*inter_range,2, 2)
        BDB_wl = BDB_wl.reshape(inter_range*inter_range,2, 2)
        BBD_wl = BBD_wl.reshape(inter_range*inter_range,2, 2)
        BB_wl = BB_wl.reshape(inter_range*inter_range,2, 2)
        
        #print("BB_wl=I", BB_wl, "DBB_wl=X", DBB_wl, "BDB_wl=-X", BDB_wl, "BBD_wl=X", BBD_wl)
        #print("2,0", BB_wl * ( ((delta)**2) / 2. ) + (DBB_wl + BDB_wl + BBD_wl) * ( ((delta)**3) / 6. ))

        wl_mpo_t[1+inter_range: inter_range*inter_range + inter_range + 2, 0,  :, :] = BB_wl * ( ((delta)**2) / 2. ) + (DBB_wl + BDB_wl + BBD_wl) * ( ((delta)**3) / 6. )
        

        AB_wl = np.einsum('ijpq, mqf->imjpf', A_wl, B_wl)
        AB_wl=AB_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        BA_wl = np.einsum('mqf, ijfp ->mijqp', B_wl, A_wl)
        BA_wl=BA_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        
        ABD_wl = np.einsum('ijpq, mqf ->imjpf', A_wl, BD_wl)
        ABD_wl=ABD_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        ADB_wl = np.einsum('ijpq, mqf ->imjpf', A_wl, DB_wl)
        ADB_wl=ADB_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        BAD_wl = np.einsum('mqf, ijfp  ->mijqp', B_wl, AD_wl)
        BAD_wl=BAD_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        BDA_wl = np.einsum('mqf, ijfp  ->mijqp', B_wl, DA_wl)
        BDA_wl=BDA_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        
        DAB_wl = np.einsum('qf, ijfp  ->ijqp', D_wl, AB_wl)
        DBA_wl = np.einsum('qf, ijfp  ->ijqp', D_wl, BA_wl)
  
        CBB_wl = np.einsum('mfp, ipq->imfq', C_wl, BB_wl)
        BCB_wl = np.einsum('mfp, ijpq->mijfq', B_wl, CB_wl)
        BCB_wl=BCB_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        BBC_wl = np.einsum('mfp, ijpq->mijfq', B_wl, BC_wl)
        BBC_wl=BBC_wl.reshape(inter_range*inter_range, inter_range,2, 2)
        
        #print("CBB_wl=Z", CBB_wl, "BCB_wl=Z", BCB_wl, "BBC_wl=Z", BBC_wl)
        #print("2,1",  (AB_wl + BA_wl) * (delta/2.) + (ABD_wl + ADB_wl + BAD_wl + BDA_wl + DAB_wl + DBA_wl + CBB_wl + BCB_wl + BBC_wl) * ( ((delta)**2) / 6. ))

        wl_mpo_t[ 1+inter_range: inter_range*inter_range + inter_range + 2, 1:inter_range+1,  :, :] = (AB_wl + BA_wl) * (delta/2.) + (ABD_wl + ADB_wl + BAD_wl + BDA_wl + DAB_wl + DBA_wl + CBB_wl + BCB_wl + BBC_wl) * ( ((delta)**2) / 6. )
        
        AA_wl = np.einsum('ijpq, mnqf->imjnpf', A_wl, A_wl)
        AA_wl=AA_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)
        
        ABC_wl = np.einsum('ijpq, mnqf ->imjnpf', A_wl, BC_wl)
        ABC_wl=ABC_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)
        ACB_wl = np.einsum('ijpq, mnqf ->imjnpf', A_wl, CB_wl)
        ACB_wl=ACB_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)
        
        BAC_wl = np.einsum('ipq, mnqf ->imnpf', B_wl, AC_wl)
        BAC_wl=BAC_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)
        BCA_wl = np.einsum('ipq, mnqf ->imnpf', B_wl, CA_wl)
        BCA_wl=BAC_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)

        CAB_wl = np.einsum('ipq, mnqf ->minpf', C_wl, AB_wl)
        CAB_wl=CAB_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)
        CBA_wl = np.einsum('ipq, mnqf ->minpf', C_wl, BA_wl)
        CBA_wl=CBA_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)

        AAD_wl = np.einsum('ijpq, qf->ijpf', AA_wl, D_wl)
        DAA_wl = np.einsum('qf, ijfp->ijqp', D_wl, AA_wl)
        ADA_wl = np.einsum('mnqf, ijfp->minjqp', AD_wl, A_wl)
        ADA_wl=ADA_wl.reshape(inter_range*inter_range, inter_range*inter_range,2, 2)
        #print("2,2",  AA_wl+ (ABC_wl + ACB_wl+ BAC_wl+ BCA_wl+ CAB_wl+CBA_wl + AAD_wl +ADA_wl+DAA_wl) * (delta/3.))
  
        wl_mpo_t[ 1+inter_range: inter_range*inter_range + inter_range + 2, 1+inter_range: inter_range*inter_range + inter_range + 2,  :, :] = AA_wl+ (ABC_wl + ACB_wl+ BAC_wl+ BCA_wl+ CAB_wl+CBA_wl + AAD_wl +ADA_wl+DAA_wl) * (delta/3.)
    
    
        # print("wl_mpo_t[0,0,:,:]", wl_mpo_t[0,0,:,:])
        # print("wl_mpo_t[0,1,:,:]", wl_mpo_t[0,1,:,:])
        # print("wl_mpo_t[0,2,:,:]", wl_mpo_t[0,2,:,:])
        # print("wl_mpo_t[1,0,:,:]", wl_mpo_t[1,0,:,:])
        # print("wl_mpo_t[1,1,:,:]", wl_mpo_t[1,1,:,:])
        # print("wl_mpo_t[1,2,:,:]", wl_mpo_t[1,2,:,:])
        # print("wl_mpo_t[2,0,:,:]", wl_mpo_t[2,0,:,:])
        # print("wl_mpo_t[2,1,:,:]", wl_mpo_t[2,1,:,:])
        # print("wl_mpo_t[2,2,:,:]", wl_mpo_t[2,2,:,:])
        # print("wl_mpo_l", wl_mpo_l)
        # print("wl_mpo_r", wl_mpo_r)

        wl_mpo_lf_t = np.einsum('ilp, ijpk->jlk', wl_mpo_l, wl_mpo_t)
        wl_mpo_rf_t = np.einsum('ijpk, jkl->ipl', wl_mpo_t, wl_mpo_r)
        


    return wl_mpo_t, wl_mpo_lf_t, wl_mpo_rf_t


def length_inter_coming_to_site_(l_inter, i):
    local_inter = [ item for item in l_inter if i in item ]
    local_inter_ = []
    for item in local_inter:
        x, y = item
        if y == i :
            local_inter_.append(item)
    local_inter_length = [ abs( item[0] - item[1] ) for item in local_inter_]
    return local_inter_length

def mpo_2d_ITF_analytic(L, terms, J=1., h=1., chi=300, dtype="float64", cutoff=1.0e-12,style_1= "left", style_2= "left"):

    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    l_inter = []
    for i in terms:
        #ignore uniform magnetic field
        if len(i) >= 2:
            l_inter.append(i)
    #print(l_inter)
    
    l_ = [] 
    for i in range(L):
        local_inter_length = length_inter_coming_to_site_(l_inter, i)
        if local_inter_length:
            l_ = [ item for item in  local_inter_length ] 
    
    l_ = sorted(list(set(l_)), key=abs)
    print("interaction_length", l_)

    for i in range(L):
        if i ==0 :
            J_ = [ 0 for item in l_]
            wl_mpo, wl_mpo_lf, wl_mpo_rf = MPO_site(J_, h, l_,dtype=dtype)
            MPO_result[i].modify(data=wl_mpo_lf)            
        elif i == L - 1:
            J_ = [ J for item in l_]
            wl_mpo, wl_mpo_lf, wl_mpo_rf = MPO_site(J_, h, l_, dtype=dtype)
            MPO_result[i].modify(data=wl_mpo_rf)
        else:
            local_inter_length = length_inter_coming_to_site_(l_inter, i)
            J_ = []
            for item in l_:
                if item in local_inter_length:
                    J_.append(J)
                else:
                    J_.append(0)
            #print("J_", J_)
            wl_mpo, wl_mpo_lf, wl_mpo_rf = MPO_site(J_, h, l_, dtype=dtype)
            MPO_result[i].modify(data=wl_mpo)


    #MPO_f=MPO_result
    #MPO_f.compress( style_1, max_bond=chi, cutoff=cutoff )
    #print ( "MPO_ITF_O",  MPO_result.show() )
    return  MPO_result 

def mpo_2d_ITF_analytic_t(L, terms, N_order = 1, delta=0.01, J=1., h=1., chi=300, dtype="complex128", order_type="f",cutoff=1.0e-12,style_1= "left", style_2= "left"):

    
    print("delta", delta)
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    l_inter = []
    for i in terms:
        # "ignore uniform magnetic field"
        if len(i) >= 2:
            l_inter.append(i)
    
    l_ = [] 
    for i in range(L):
        local_inter_length = length_inter_coming_to_site_(l_inter, i)
        if local_inter_length:
            l_ = [ item for item in  local_inter_length ] 
    
    l_ = sorted(list(set(l_)), key=abs)
    for i in tqdm(range(L)):
        if i ==0 :
            J_ = [ 0 for item in l_]
            wl_mpo, wl_mpo_lf, wl_mpo_rf = MPO_site_t(J_, h, l_, N_order, delta, order_type=order_type,dtype=dtype)
            MPO_result[i].modify(data=wl_mpo_lf)            
        elif i == L - 1:
            J_ = [ J for item in l_]
            wl_mpo, wl_mpo_lf, wl_mpo_rf = MPO_site_t(J_, h, l_, N_order, delta, order_type=order_type,dtype=dtype)
            MPO_result[i].modify(data=wl_mpo_rf)
        else:
            local_inter_length = length_inter_coming_to_site_(l_inter, i)
            J_ = []
            for item in l_:
                if item in local_inter_length:
                    J_.append(J)
                else:
                    J_.append(0)
            wl_mpo, wl_mpo_lf, wl_mpo_rf = MPO_site_t(J_, h, l_,N_order, delta, order_type=order_type, dtype=dtype)
            MPO_result[i].modify(data=wl_mpo)


    return  MPO_result 




def mpo_2d_ITF_t( Lx, Ly, L, delta = 0.1, J=1., h=1., chi=300, dtype="float64", cutoff=1.0e-12,style_1= "left", style_2= "left", order=1, keep_term=100):
    
    Wl = np.zeros([ 1, 2, 2], dtype=dtype)
    W = np.zeros([1, 1, 2, 2], dtype=dtype)
    Wr = np.zeros([ 1, 2, 2], dtype=dtype)
    Wll = np.zeros([ 1, 2, 2], dtype=dtype)
    WW = np.zeros([1, 1, 2, 2], dtype=dtype)
    Wrr = np.zeros([ 1, 2, 2], dtype=dtype)
    Wlll = np.zeros([ 1, 2, 2], dtype=dtype)
    WWW = np.zeros([1, 1, 2, 2], dtype=dtype)
    Wrrr = np.zeros([ 1, 2, 2], dtype=dtype)
    Wl4 = np.zeros([ 1, 2, 2], dtype=dtype)
    W4 = np.zeros([1, 1, 2, 2], dtype=dtype)
    Wr4 = np.zeros([ 1, 2, 2], dtype=dtype)

    if order >= 1:
        mpo_dt = mpo_2d_ITF_analytic_t( Lx, Ly, L, delta = delta, J=J, h=h, order_type="f", dtype="complex128")

    Z=qu.pauli('Z',dtype=dtype) 
    X=qu.pauli('X',dtype=dtype) 
    Y= np.array([[0, -1],[1,0]]) 
    I=qu.pauli('I',dtype=dtype)
    Y=Y.astype(dtype)
    X=X.astype(dtype)
    Z=Z.astype(dtype)

    MPO_I=qtn.MPO_identity(L, phys_dim=2)
    MPO_result=qtn.MPO_identity(L, phys_dim=2)
    MPO_result = mpo_dt
    
    MPO_H = mpo_dt * 1.e-12
    MPO_2 = mpo_dt * 1.e-12
    MPO_3 = mpo_dt * 1.e-12
    MPO_33 = mpo_dt * 1.e-12
    MPO_333 = mpo_dt * 1.e-12
    
    max_bond_val = chi
    
    terms_x = {    (i*Ly + j, (  i+1  )*Ly+j) : Z             for i, j in itertools.product(range(Lx-1), range(Ly)) }
    terms_y = {    (i*Ly + j, i*Ly+j+1) : Z             for i, j in itertools.product(range(Lx), range(Ly-1)) }
    terms_z = {    (i, ) : X             for i in range(L) }
    terms = terms_x |  terms_y | terms_z
   
    count_ = 0
    for i in tqdm(terms):
        elem = terms[i]  
        Wl[ 0,:,:] = elem
        W[ 0,0,:,:] = elem
        Wr[ 0,:,:] = elem
        W_list = [Wl] + [W]*(L-2)+ [Wr]

        if len(i) == 2:
            ii, ii_ = i
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[ii].modify(data=W_list[ii])
            MPO_I[ii_].modify(data=W_list[ii_] * J )
        else :
            ii, = i
            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
            MPO_I[ii].modify(data=W_list[ii] * h)

        if count_ == 0:
            MPO_H = MPO_I
        else:
            MPO_H = MPO_H + MPO_I
            MPO_H.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
        count_ += 1


    if order >= 2:
        print("order", 2)
        count_ = 0
        for i,j in product(terms, terms):
                if count_>keep_term:
                        #print("count_>", count_)
                        break;

                if set(i) & set(j):
                    #print("2", i, j, set(i) & set(j) )

                    elem = terms[i]  
                    Wl[ 0,:,:] = elem
                    W[ 0,0,:,:] = elem
                    Wr[ 0,:,:] = elem

                    elem = terms[j]  
                    Wll[ 0,:,:] = elem
                    WW[ 0,0,:,:] = elem
                    Wrr[ 0,:,:] = elem

                    W_list = [Wl] + [W]*(L-2)+ [Wr]
                    WW_list = [Wll] + [WW]*(L-2)+ [Wrr]
                    if len(i) == 2:
                        ii, ii_ = i
                        MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                        MPO_I[ii].modify(data=W_list[ii])
                        MPO_I[ii_].modify(data=W_list[ii_] * J )
                    else :
                        ii, = i
                        MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                        MPO_I[ii].modify(data=W_list[ii] * h)

                    if len(j) == 2:
                        jj, jj_ = j
                        MPO_II=qtn.MPO_identity(L, phys_dim=2 )
                        MPO_II[jj].modify(data=WW_list[jj])
                        MPO_II[jj_].modify(data=WW_list[jj_] * J)
                    else:
                        jj, = j
                        MPO_II=qtn.MPO_identity(L, phys_dim=2 )
                        MPO_II[jj].modify(data=WW_list[jj] * h)
            
                    MPO_I = MPO_I.apply(MPO_II)
                    if count_ == 0:
                        MPO_2 = MPO_I
                    else:
                        MPO_2 = MPO_2 + MPO_I
                        MPO_2.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                    count_ += 1
                    
        MPO_result = MPO_result + MPO_2 * ((delta**2) /2.)
        MPO_result.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
    

    if order >= 3:
        print("order", 3)
        count_ = 0
        for i,j,k in product(terms, terms, terms):
                    if count_>keep_term:
                        #print("count_>", count_)
                        break;

                    #print("original", i, j, k)
                    if set(i) & set(j) or set(i) & set(k) or set(j) & set(k):
                        #print(i, j,k, set(i) & set(j) ,  set(i) & set(k) , set(k) & set(j) )
                        elem = terms[i]  
                        Wl[ 0,:,:] = elem
                        W[ 0,0,:,:] = elem
                        Wr[ 0,:,:] = elem

                        elem = terms[j]  
                        Wll[ 0,:,:] = elem
                        WW[ 0,0,:,:] = elem
                        Wrr[ 0,:,:] = elem

                        elem = terms[k]  
                        Wlll[ 0,:,:] = elem
                        WWW[ 0,0,:,:] = elem
                        Wrrr[ 0,:,:] = elem


                        W_list = [Wl] + [W]*(L-2)+ [Wr]
                        WW_list = [Wll] + [WW]*(L-2)+ [Wrr]
                        WWW_list = [Wlll] + [WWW]*(L-2)+ [Wrrr]

                        if len(i) == 2:
                            ii, ii_ = i
                            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_I[ii].modify(data=W_list[ii])
                            MPO_I[ii_].modify(data=W_list[ii_] * J )
                        else :
                            ii, = i
                            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_I[ii].modify(data=W_list[ii] * h)

                        if len(j) == 2:
                            jj, jj_ = j
                            MPO_II=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_II[jj].modify(data=WW_list[jj])
                            MPO_II[jj_].modify(data=WW_list[jj_] * J)
                        else:
                            jj, = j
                            MPO_II=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_II[jj].modify(data=WW_list[jj] * h)

                        if len(k) == 2:
                            kk, kk_ = k
                            MPO_III=qtn.MPO_identity(L, phys_dim=2)
                            MPO_III[kk].modify(data=WWW_list[kk])
                            MPO_III[kk_].modify(data=WWW_list[kk_] * J)
                        else:
                            kk, = k
                            MPO_III=qtn.MPO_identity(L, phys_dim=2)
                            MPO_III[kk].modify(data=WWW_list[kk] * h)



                        MPO_I = MPO_II.apply(MPO_I)
                        MPO_I.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                        MPO_I = MPO_III.apply(MPO_I)
                        MPO_I.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                        
                        if count_ == 0:
                            MPO_3f =  MPO_I 
                        else:
                            MPO_3f = MPO_3f +  MPO_I 
                            MPO_3f.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                        count_ += 1
        MPO_result = MPO_result + MPO_3f * ((delta**3) /6.)
        MPO_result.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
        print("count_", count_)

    if order >= 4:
        print("order", 4)
        count_ = 0
        for i,j,k, m in product(terms, terms, terms, terms):
                    if count_>keep_term:
                        print("count_>", count_)
                        break;
                    #print("original", i, j, k)
                    if set(i) & set(j) or set(i) & set(k) or set(i) & set(m) or set(j) & set(k)  or set(j) & set(m) or set(k) & set(m):
                        #print(i, j,k, set(i) & set(j) ,  set(i) & set(k) , set(k) & set(j) )
                        elem = terms[i]  
                        Wl[ 0,:,:] = elem
                        W[ 0,0,:,:] = elem
                        Wr[ 0,:,:] = elem

                        elem = terms[j]  
                        Wll[ 0,:,:] = elem
                        WW[ 0,0,:,:] = elem
                        Wrr[ 0,:,:] = elem

                        elem = terms[k]  
                        Wlll[ 0,:,:] = elem
                        WWW[ 0,0,:,:] = elem
                        Wrrr[ 0,:,:] = elem

                        elem = terms[m]  
                        Wl4[ 0,:,:] = elem
                        W4[ 0,0,:,:] = elem
                        Wr4[ 0,:,:] = elem

                        
                        W_list = [Wl] + [W]*(L-2)+ [Wr]
                        WW_list = [Wll] + [WW]*(L-2)+ [Wrr]
                        WWW_list = [Wlll] + [WWW]*(L-2)+ [Wrrr]
                        W4_list = [Wl4] + [W4]*(L-2)+ [Wr4]

                        if len(i) == 2:
                            ii, ii_ = i
                            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_I[ii].modify(data=W_list[ii])
                            MPO_I[ii_].modify(data=W_list[ii_] * J )
                        else :
                            ii, = i
                            MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_I[ii].modify(data=W_list[ii] * h)

                        if len(j) == 2:
                            jj, jj_ = j
                            MPO_II=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_II[jj].modify(data=WW_list[jj])
                            MPO_II[jj_].modify(data=WW_list[jj_] * J)
                        else:
                            jj, = j
                            MPO_II=qtn.MPO_identity(L, phys_dim=2 )
                            MPO_II[jj].modify(data=WW_list[jj] * h)

                        if len(k) == 2:
                            kk, kk_ = k
                            MPO_III=qtn.MPO_identity(L, phys_dim=2)
                            MPO_III[kk].modify(data=WWW_list[kk])
                            MPO_III[kk_].modify(data=WWW_list[kk_] * J)
                        else:
                            kk, = k
                            MPO_III=qtn.MPO_identity(L, phys_dim=2)
                            MPO_III[kk].modify(data=WWW_list[kk] * h)

                        if len(m) == 2:
                            mm, mm_ = m
                            MPO_I4=qtn.MPO_identity(L, phys_dim=2)
                            MPO_I4[mm].modify(data=W4_list[mm])
                            MPO_I4[mm_].modify(data=W4_list[mm_] * J)
                        else:
                            mm, = m
                            MPO_I4=qtn.MPO_identity(L, phys_dim=2)
                            MPO_I4[mm].modify(data=W4_list[mm] * h)


                        MPO_I = MPO_II.apply(MPO_I)
                        MPO_I.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                        MPO_I = MPO_III.apply(MPO_I)
                        MPO_I.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                        MPO_I = MPO_I4.apply(MPO_I)
                        MPO_I.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                        
                        if count_ == 0:
                            MPO_4f =  MPO_I 
                        else:
                            MPO_4f = MPO_4f +  MPO_I 
                            MPO_4f.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
                        count_ += 1
                        
        MPO_result = MPO_result + MPO_4f * ((delta**4) /math.factorial(4))
        MPO_result.compress( style_2, max_bond=max_bond_val, cutoff=cutoff )
        
        

    print ( "MPO_ITF_O",  MPO_result.show() )
    return  MPO_result 

def exp_ham_Tylor_eff(L, MPO, delta = 0.01, order_expan=2, cutoff=1e-20, max_bond_val=50, style= "left", style_= "left"):

    
    MPO_list = [] 
    MPO_list.append(qtn.MPO_identity(L, phys_dim=2, cyclic=False ))
    MPO_list.append(MPO*1.0)
    l_maxbond = []
    
    if order_expan>=2:
        print("2")
        MPO_2 = MPO.apply(MPO)
        MPO_2.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_2)
        l_maxbond.append(MPO_2.max_bond())
    if order_expan>=3:
        print("3")
        MPO_3 = MPO_2.apply(MPO)
        MPO_3.compress( style_, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_3)
        l_maxbond.append(MPO_3.max_bond())
    if order_expan>=4:
        print("4")
        MPO_4 = MPO_2.apply(MPO_2)
        MPO_4.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_4)
        l_maxbond.append(MPO_4.max_bond())
    if order_expan>=5:
        print("5")
        MPO_5 = MPO_3.apply(MPO_2)
        MPO_5.compress( style_, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_5)
        l_maxbond.append(MPO_5.max_bond())
    if order_expan>=6:
        print("6")
        MPO_6 = MPO_3.apply(MPO_3)
        MPO_6.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_6)
        l_maxbond.append(MPO_6.max_bond())
    if order_expan>=7:
        print("7")
        MPO_7 = MPO_4.apply(MPO_3)
        MPO_7.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_7)
        l_maxbond.append(MPO_7.max_bond())
    if order_expan>=8:
        print("8")
        MPO_8 = MPO_4.apply(MPO_4)
        MPO_8.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_8)
        l_maxbond.append(MPO_8.max_bond())
    if order_expan>=9:
        print("9")
        MPO_9 = MPO_5.apply(MPO_4)
        MPO_9.compress( style_, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_9)
        l_maxbond.append(MPO_9.max_bond())
    if order_expan>=10:
        print("10")
        MPO_10 = MPO_5.apply(MPO_5)
        MPO_10.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_10)
        l_maxbond.append(MPO_10.max_bond())
    if order_expan>=11:
        print("11")
        MPO_11 = MPO_6.apply(MPO_5)
        MPO_11.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_11)
        l_maxbond.append(MPO_11.max_bond())
    if order_expan>=12:
        print("12")
        MPO_12 = MPO_6.apply(MPO_6)
        MPO_12.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_12)
        l_maxbond.append(MPO_12.max_bond())
    if order_expan>=13:
        print("13")
        MPO_13 = MPO_7.apply(MPO_6)
        MPO_13.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_13)
        l_maxbond.append(MPO_13.max_bond())
    if order_expan>=14:
        print("14")
        MPO_14 = MPO_8.apply(MPO_6)
        MPO_14.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_14)
        l_maxbond.append(MPO_14.max_bond())
    if order_expan>=15:
        print("15")
        MPO_15 = MPO_8.apply(MPO_7)
        MPO_15.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_15)
        l_maxbond.append(MPO_15.max_bond())
    if order_expan>=16:
        print("16")
        MPO_16 = MPO_8.apply(MPO_8)
        MPO_16.compress( style, max_bond=max_bond_val, cutoff=cutoff )
        MPO_list.append(MPO_16)
        l_maxbond.append(MPO_16.max_bond())
        
    
    MPO_i_STEP = qtn.MPO_identity(L, phys_dim=2, cyclic=False ) 
    for i in range(len(MPO_list)):
            #print(i, (delta**(i))/math.factorial(i))
            if i == 0:
                MPO_i_STEP = MPO_list[i]
            else:
                MPO_i_STEP = MPO_i_STEP + MPO_list[i]*((delta**(i))/math.factorial(i))            
                MPO_i_STEP.compress( style, max_bond=max_bond_val, cutoff=cutoff )
            

    print("largest_max_bond_during_Taylor", l_maxbond, max(l_maxbond))
    #MPO_result.show()
    return   MPO_i_STEP, max(l_maxbond)



def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))
    
    
def exp_ham_Tylor(L, MPO, delta = 0.01, order_expan=2, cutoff=1e-20, max_bond_val=50, style= "left"):

    l_maxbond = []
    for i in range(order_expan):
        #print(i, l_maxbond)
        if i==0:
            MPO_result = qtn.MPO_identity(L, phys_dim=2, cyclic=False )
            l_maxbond.append(MPO_result.max_bond())
        elif i==1:
            MPO_result = MPO_result+MPO*((delta**(i))/math.factorial(i))   
            MPO_result.compress( style, max_bond=max_bond_val, cutoff=cutoff )
            l_maxbond.append(MPO_result.max_bond())
        else:
            MPO_i=MPO*1.0
            for j in range(i-1):
                #print(i, "j", j)
                MPO_i = MPO_i.apply(MPO)
                MPO_i.compress( style, max_bond=max_bond_val, cutoff=cutoff )
                l_maxbond.append(MPO_i.max_bond())
    
            MPO_i_STEP = MPO_i*((delta**(i))/math.factorial(i))            
            MPO_result =  MPO_result + MPO_i_STEP  
            MPO_result.compress( style, max_bond=max_bond_val, cutoff=cutoff )
            l_maxbond.append(MPO_result.max_bond())
            

    #print("largest_max_bond_during_Taylor", max(l_maxbond))
    #MPO_result.show()
    return   MPO_result, max(l_maxbond)



def exp_ham_Tylor_local(L, MPO, delta = 0.01, order_expan=2, cutoff=1e-20, max_bond_val=50, style= "left"):

    l_maxbond = []
    for i in range(order_expan):
        #print("i",i)
        if i==1:
            #print("i")
            MPO_result = MPO*((delta**(i))/math.factorial(i))   
            MPO_result.compress( style, max_bond=max_bond_val, cutoff=cutoff )
            l_maxbond.append(MPO_result.max_bond())
        elif i>1:
            MPO_i=MPO*1.0
            for j in range(i-1):
                #print(i, "j", j)
                MPO_i = MPO_i.apply(MPO)
                MPO_i.compress( style, max_bond=max_bond_val, cutoff=cutoff )
                l_maxbond.append(MPO_i.max_bond())
    
            MPO_i_STEP = MPO_i*((delta**(i))/math.factorial(i))            
            MPO_result =  MPO_result + MPO_i_STEP  
            MPO_result.compress( style, max_bond=max_bond_val, cutoff=cutoff )
            l_maxbond.append(MPO_result.max_bond())
            

    #print("largest_max_bond_during_Taylor", max(l_maxbond))
    #MPO_result.show()
    return   MPO_result, max(l_maxbond)


def mpo_t_compress(Lx,Ly,L, terms_t, J=1., h=1., delta=0.01, SIZE=4 ,order_expan_local = 6, chi=300, dtype="float64", 
                   cutoff=1.0e-12,style_1= "left", style_2= "left", 
                   cycle = "open",
                   keep_term = math.inf
                  ):
    Wl = np.zeros([ 1, 2, 2], dtype=dtype)
    W = np.zeros([1, 1, 2, 2], dtype=dtype)
    Wr = np.zeros([ 1, 2, 2], dtype=dtype)

    Z=qu.pauli('Z',dtype=dtype) 
    X=qu.pauli('X',dtype=dtype) 
    Y=qu.pauli('Y',dtype=dtype)
    I=qu.pauli('I',dtype=dtype)
    Y=Y.astype(dtype)
    X=X.astype(dtype)
    Z=Z.astype(dtype)
    mpo_l = []
    for terms in chunks(terms_t, SIZE):
        count_ = 0
        # print(terms)
        for i in tqdm(terms, colour="green"):        
            elem = terms[i]
            Wl[ 0, :, :]  = elem
            W[ 0,0, :, :] = elem
            Wr[ 0, :, :]  = elem
            W_list = [Wl] + [W]*(L-2)+ [Wr]
            if len(i) == 2:
                ii, ii_ = i
                MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                MPO_I[ii].modify(data=W_list[ii])
                MPO_I[ii_].modify(data=W_list[ii_] * J )
            else:
                ii, = i
                MPO_I=qtn.MPO_identity(L, phys_dim=2 )
                MPO_I[ii].modify(data=W_list[ii] * h)

            if count_ == 0:
                MPO_H = MPO_I
            else:
                MPO_H = MPO_H + MPO_I
                MPO_H.compress( style_1, max_bond=chi, cutoff=cutoff )
            count_ += 1
        mpo_l.append(MPO_H)
        

    mpo_t_f = qtn.MPO_identity(L, phys_dim=2 )
    
    mpo_t_l = []
    for count, mpo in enumerate(tqdm(mpo_l, colour="CYAN")):
        mpo, max_bond = exp_ham_Tylor_local(L, mpo, delta = delta, order_expan=order_expan_local, cutoff=1e-16, max_bond_val=chi, style=style_1)
        mpo_t_l.append( mpo )
    
    
    for count, value in enumerate(tqdm(mpo_t_l, colour="YELLOW")):
            mpo_t_f  = mpo_t_f + value
            mpo_t_f.compress( style_1, max_bond=chi, cutoff=cutoff )

            
            
    count_ = 0
    print("order_2")
    for idx, pair in enumerated_product(mpo_l, mpo_l):
        if count_>keep_term:
            print("count_>", count_)
            break;

        #print(idx, pair)
        i, j = idx
        MPO_I, MPO_II = pair
        
        if i != j:
            #print(i,j, i != j)
            MPO_2 = MPO_I.apply(MPO_II)
            MPO_2.compress( style_2, max_bond=chi, cutoff=cutoff )
            mpo_t_f = mpo_t_f + MPO_2 *  ( (delta**2) / (math.factorial(2)) )
            mpo_t_f.compress( style_1, max_bond=chi, cutoff=cutoff )
            count_ += 1

    count_ = 0
    print("order_3")
    for idx, pair in enumerated_product(mpo_l, mpo_l, mpo_l):
        if count_>keep_term:
            print("count_>", count_)
            break;
        
        #print(idx, pair)
        i, j, k = idx
        MPO_I, MPO_II, MPO_III = pair
        
        if i != j or i != k or j != k:
            #print(i,j,k, i != j or i != k or j != k)
            MPO_2 = MPO_I.apply(MPO_II)
            MPO_2.compress( style_2, max_bond=chi, cutoff=cutoff )
            MPO_3 = MPO_2.apply(MPO_III)
            MPO_3.compress( style_2, max_bond=chi, cutoff=cutoff )
            mpo_t_f = mpo_t_f + MPO_3 *  ((delta**3)/math.factorial(3))
            mpo_t_f.compress( style_1, max_bond=chi, cutoff=cutoff )
            count_ += 1
    
    count_ = 0
    print("order_4")
    for idx, pair in enumerated_product(mpo_l, mpo_l, mpo_l, mpo_l):
        if count_>keep_term:
            print("count_>", count_)
            break;

        #print(idx, pair)
        i, j, k, m = idx
        MPO_I, MPO_II, MPO_III, MPO_IIII = pair
        
        if (i != j) or (i != k) or (i != m) or (j != k) or (j != m) or (k != m):
            #print(i,j,k,m,(i != j) or (i != k) or (i != m) or (j != k) or (j != m) or (k != m))
            MPO_2 = MPO_I.apply(MPO_II)
            MPO_2.compress( style_2, max_bond=chi, cutoff=cutoff )
            MPO_3 = MPO_2.apply(MPO_III)
            MPO_3.compress( style_1, max_bond=chi, cutoff=cutoff )
            MPO_4 = MPO_3.apply(MPO_IIII)
            MPO_4.compress( style_2, max_bond=chi, cutoff=cutoff )
            mpo_t_f = mpo_t_f + MPO_4 *  ((delta**4)/math.factorial(4))
            mpo_t_f.compress( style_1, max_bond=chi, cutoff=cutoff )
            count_ += 1 

    return  mpo_t_f 


def dpepo(lx, ly, theta = - 2*math.pi/48):
    pepo = qtn.PEPO.rand(Lx=lx, Ly=ly, bond_dim=1)
    data = [math.cos(theta), math.sin(theta)]
    data = qu.qu(data, qtype='ket')
    rho_ = qu.qu(data, qtype='dop', sparse=False)
    for t in pepo:
        shape=t.shape
        project = rho_
        t.modify(data = project.reshape(shape))
    return pepo


def peps_I(Lx, Ly, dtype="complex128", theta = 0):
    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=666, dtype=dtype)
    for t in peps:
        if len(t.data.shape) == 3:
            W = np.zeros([1,1,2], dtype=dtype)
            W[0,0,:] = np.array([math.cos(theta), math.sin(theta)])
            t.modify(data = W)
        if len(t.data.shape) == 4:
            W = np.zeros([1,1,1,2], dtype=dtype)
            W[0,0,0,:] = np.array([math.cos(theta), math.sin(theta)])
            t.modify(data = W)
        if len(t.data.shape) == 5:
            W = np.zeros([1,1,1,1,2], dtype=dtype)
            W[0,0,0,0,:] = np.array([math.cos(theta), math.sin(theta)])
            t.modify(data = W)
    peps.astype_(dtype)
    return peps

def pepo_identity(Lx, Ly, dtype="complex128"):
    pepo = qtn.PEPO.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=666, dtype=dtype)        
    I = qu.pauli('I', dtype=dtype)
    for t in pepo:
        if len(t.data.shape) == 4:
            W = np.zeros([1,1,2,2], dtype=dtype)
            W[0,0,:,:] = I
            t.modify(data = W)
        if len(t.data.shape) == 5:
            W = np.zeros([1,1,1,2,2], dtype=dtype)
            W[0,0,0,:,:] = I
            t.modify(data = W)
        if len(t.data.shape) == 6:
            W = np.zeros([1,1,1,1,2,2], dtype=dtype)
            W[0,0,0,0,:,:] = I
            t.modify(data = W)
    return pepo


def site_ind(pepo, i, j=None, lable = "lower"):
        """Return the physical index of site ``(i, j)``."""
        if j is None:
            i, j = i
        if not isinstance(i, str):
            i = i % pepo.Lx
        if not isinstance(j, str):
            j = j % pepo.Ly
        if lable == "upper":
            return pepo.upper_ind_id.format(i, j)
        if lable == "lower":
            return pepo.lower_ind_id.format(i, j)

        
        
def gate_f(
        pepo,
        G,
        where,
        lable = "upper",
        contract=False,
        tags=None,
        propagate_tags="sites",
        inplace=False,
        info=None,
        long_range_use_swaps=False,
        long_range_path_sequence=None,
        **compress_opts,
    ):
        check_opt("contract", contract, (False, True, "split", "reduce-split"))

        psi = pepo if inplace else pepo.copy()

        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)
        ng = len(where)

    
        dp = psi.phys_dim(*where[0])
        tags = tags_to_oset(tags)

        # allow a matrix to be reshaped into a tensor if it factorizes
        #     i.e. (4, 4) assumed to be two qubit gate -> (2, 2, 2, 2)
        G = maybe_factor_gate_into_tensor(G, dp, ng, where)

        site_ix = [site_ind(psi, i, j, lable=lable) for i, j in where]
        # new indices to join old physical sites to new gate
        bnds = [rand_uuid() for _ in range(ng)]
        reindex_map = dict(zip(site_ix, bnds))

        TG = qtn.Tensor(G, inds=site_ix + bnds, tags=tags, left_inds=bnds)

        if contract is False:
            #
            #       │   │      <- site_ix
            #       GGGGG
            #       │╱  │╱     <- bnds
            #     ──●───●──
            #      ╱   ╱
            #
            if propagate_tags:
                if propagate_tags == "register":
                    old_tags = oset(map(psi.site_tag, where))
                else:
                    old_tags = oset_union(
                        psi.tensor_map[tid].tags
                        for ind in site_ix
                        for tid in psi.ind_map[ind]
                    )

                if propagate_tags == "sites":
                    # use regex to take tags only matching e.g. 'I4,3'
                    rex = re.compile(psi.site_tag_id.format(r"\d+", r"\d+"))
                    old_tags = oset(filter(rex.match, old_tags))

                TG.modify(tags=TG.tags | old_tags)

            psi.reindex_(reindex_map)
            psi |= TG
            return psi

        if (contract is True) or (ng == 1):
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)

            # get the sites that used to have the physical indices
            site_tids = psi._get_tids_from_inds(bnds, which="any")

            # pop the sites, contract, then re-add
            pts = [psi.pop_tensor(tid) for tid in site_tids]
            psi |= tensor_contract(*pts, TG)

            return psi

        # following are all based on splitting tensors to maintain structure
        ij_a, ij_b = where

        # parse the argument specifying how to find the path between
        # non-nearest neighbours
        if long_range_path_sequence is not None:
            # make sure we can index
            long_range_path_sequence = tuple(long_range_path_sequence)
            # if the first element is a str specifying move sequence, e.g.
            #     ('v', 'h')
            #     ('av', 'bv', 'ah', 'bh')  # using swaps
            manuaLr_path = not isinstance(long_range_path_sequence[0], str)
            # otherwise assume a path has been manually specified, e.g.
            #     ((1, 2), (2, 2), (2, 3), ... )
            #     (((1, 1), (1, 2)), ((4, 3), (3, 3)), ...)  # using swaps
        else:
            manuaLr_path = False
        
        # check if we are not nearest neighbour and need to swap first
        if long_range_use_swaps:
            if manuaLr_path:
                *swaps, final = long_range_path_sequence
            else:
                # find a swap path
                *swaps, final = gen_long_range_swap_path(
                    ij_a, ij_b, sequence=long_range_path_sequence
                )

            # move the sites together
            SWAP = get_swap(
                dp, dtype=autoray.get_dtype_name(G), backend=autoray.infer_backend(G)
            )
            path = swaps+[final]
            # tags_compress = []
            # for cor in path:
            #     x, y = cor
            #     x0, x1 = x
            #     y0, y1 = y
            #     tags_compress.append( [f"I{x0},{x1}", f"I{y0},{y1}"])

            
            # tags_compress = [item for sublist in tags_compress for item in sublist]

            # psi.canonize_around(tags_compress, 
            #                     which='any', 
            #                     min_distance=0, 
            #                     max_distance=2, 
            #                     include=None, exclude=None, 
            #                     span_opts=None, absorb='right', 
            #                     gauge_links=False,
            #                     link_absorb='both',
            #                     equalize_norms=False,
            #                     inplace=True, 
            #                     #**canonize_opts
            #                     )



            for pair in swaps:
                psi=gate_f_(psi, SWAP, pair, lable=lable, contract=contract, absorb="right")

            compress_opts["info"] = info
            compress_opts["contract"] = contract

            # perform actual gate also compressing etc on 'way back'
            psi=gate_f_(psi,G, final, lable=lable, **compress_opts)

            compress_opts.setdefault("absorb", "both")
            for pair in reversed(swaps):
                psi=gate_f_(psi,SWAP, pair, lable=lable, **compress_opts)

            
            return psi

        if manuaLr_path:
            string = long_range_path_sequence
        else:
            string = tuple(
                gen_long_range_path(*where, sequence=long_range_path_sequence)
            )

        # the tensors along this string, which will be updated
        original_ts = [psi[coo] for coo in string]

        # the len(string) - 1 indices connecting the string
        bonds_along = [
            next(iter(bonds(t1, t2))) for t1, t2 in pairwise(original_ts)
        ]

        if contract == "split":
            #
            #       │╱  │╱          │╱  │╱
            #     ──GGGGG──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱   ╱
            #
            gate_string_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )

        elif contract == "reduce-split":
            #
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            gate_string_reduce_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )

        return psi

    
def gate_f_(
        pepo,
        G,
        where,
        lable = "upper",
        contract=False,
        tags=None,
        propagate_tags="sites",
        inplace=False,
        info=None,
        long_range_use_swaps=False,
        long_range_path_sequence=None,
        **compress_opts,
    ):
        check_opt("contract", contract, (False, True, "split", "reduce-split"))

        psi = pepo if inplace else pepo.copy()

        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)
        ng = len(where)

        dp = psi.phys_dim(*where[0])
        tags = tags_to_oset(tags)

        # allow a matrix to be reshaped into a tensor if it factorizes
        #     i.e. (4, 4) assumed to be two qubit gate -> (2, 2, 2, 2)
        G = maybe_factor_gate_into_tensor(G, dp, ng, where)

        site_ix = [site_ind(psi, i, j, lable=lable) for i, j in where]
        # new indices to join old physical sites to new gate
        bnds = [rand_uuid() for _ in range(ng)]
        reindex_map = dict(zip(site_ix, bnds))

        TG = qtn.Tensor(G, inds=site_ix + bnds, tags=tags, left_inds=bnds)

        if contract is False:
            #
            #       │   │      <- site_ix
            #       GGGGG
            #       │╱  │╱     <- bnds
            #     ──●───●──
            #      ╱   ╱
            #
            if propagate_tags:
                if propagate_tags == "register":
                    old_tags = oset(map(psi.site_tag, where))
                else:
                    old_tags = oset_union(
                        psi.tensor_map[tid].tags
                        for ind in site_ix
                        for tid in psi.ind_map[ind]
                    )

                if propagate_tags == "sites":
                    # use regex to take tags only matching e.g. 'I4,3'
                    rex = re.compile(psi.site_tag_id.format(r"\d+", r"\d+"))
                    old_tags = oset(filter(rex.match, old_tags))

                TG.modify(tags=TG.tags | old_tags)

            psi.reindex_(reindex_map)
            psi |= TG
            return psi

        if (contract is True) or (ng == 1):
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)

            # get the sites that used to have the physical indices
            site_tids = psi._get_tids_from_inds(bnds, which="any")

            # pop the sites, contract, then re-add
            pts = [psi.pop_tensor(tid) for tid in site_tids]
            psi |= tensor_contract(*pts, TG)

            return psi

        # following are all based on splitting tensors to maintain structure
        ij_a, ij_b = where

        # parse the argument specifying how to find the path between
        # non-nearest neighbours
        if long_range_path_sequence is not None:
            # make sure we can index
            long_range_path_sequence = tuple(long_range_path_sequence)
            # if the first element is a str specifying move sequence, e.g.
            #     ('v', 'h')
            #     ('av', 'bv', 'ah', 'bh')  # using swaps
            manuaLr_path = not isinstance(long_range_path_sequence[0], str)
            # otherwise assume a path has been manually specified, e.g.
            #     ((1, 2), (2, 2), (2, 3), ... )
            #     (((1, 1), (1, 2)), ((4, 3), (3, 3)), ...)  # using swaps
        else:
            manuaLr_path = False

        # check if we are not nearest neighbour and need to swap first
        if long_range_use_swaps:
            if manuaLr_path:
                *swaps, final = long_range_path_sequence
            else:
                # find a swap path
                *swaps, final = gen_long_range_swap_path(
                    ij_a, ij_b, sequence=long_range_path_sequence
                )

            # move the sites together
            SWAP = get_swap(
                dp, dtype=autoray.get_dtype_name(G), backend=autoray.infer_backend(G)
            )
            for pair in swaps:
                psi=gate_f_(psi, SWAP, pair, contract=contract, absorb="right")

            compress_opts["info"] = info
            compress_opts["contract"] = contract

            # perform actual gate also compressing etc on 'way back'
            psi=gate_f_(psi,G, final, **compress_opts)

            compress_opts.setdefault("absorb", "both")
            for pair in reversed(swaps):
                psi=gate_f_(psi,SWAP, pair, **compress_opts)

            return psi

        if manuaLr_path:
            string = long_range_path_sequence
        else:
            string = tuple(
                gen_long_range_path(*where, sequence=long_range_path_sequence)
            )

        # the tensors along this string, which will be updated
        original_ts = [psi[coo] for coo in string]

        # the len(string) - 1 indices connecting the string
        bonds_along = [
            next(iter(bonds(t1, t2))) for t1, t2 in pairwise(original_ts)
        ]

        if contract == "split":
            #
            #       │╱  │╱          │╱  │╱
            #     ──GGGGG──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱   ╱
            #
            gate_string_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )

        elif contract == "reduce-split":
            #
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            gate_string_reduce_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )

        return psi






    
def pepo_from_gate_(gate_dic, Lx, Ly, dtype="complex128"):
    
    pepo = pepo_identity(Lx, Ly, dtype=dtype)
    I = qu.pauli('I')

    x, y = 0, 0
    x_, y_ = 0, 0
    for where in gate_dic:
        if len(where) == 2:
            (x, y), (x_, y_) = where
            U_ = gate_dic[where]
        else:
            (x, y),  = where
            U_ = gate_dic[where]
            shape_ = pepo[f"I{x},{y}"].shape
            W = np.zeros(shape_, dtype=dtype)
            if len(shape_) == 4:
                W[0,0,:,:] = I
                pepo[f"I{x},{y}"].modify(data = W)
            if len(shape_) == 5:
                W[0,0,0,:,:] = I
                pepo[f"I{x},{y}"].modify(data = W)
            if len(shape_) == 6:
                W[0,0,0,0,:,:] = I
                pepo[f"I{x},{y}"].modify(data = W)
                
    if x < 0 or y <0 or x_ < 0 or y_ <0:
        print("negetive positoins")
    if x == x_ and y == y_:
        print("U acting on the same positoins")
    elif  abs(x - x_) > 1 or abs(y - y_) > 1:
        print("U acting on the long-rnage")
        
    tag_1 = f"I{x},{y}"
    tag_2 = f"I{x_},{y_}"
    shape_1 = pepo[f"I{x},{y}"].shape
    shape_2 = pepo[f"I{x_},{y_}"].shape
    #print(f"I{x},{y}", shape_1, f"I{x_},{y_}", shape_2)
    
    T = qtn.Tensor(data=U_, inds=("b0","b1","k0","k1" ), tags=[])

    T_l, T_r = qtn.tensor_split(T, ["b0","k0"], get = "tensors", cutoff=1e-16, bond_ind="x")
    T_l=T_l.transpose("x", "b0", "k0")
    T_r=T_r.transpose("x", "b1", "k1")
    bnd_ = T_l.ind_size("x")
    if x < x_ and y == y_: 
        if len(shape_1) == 4:
            W = np.zeros([bnd_, 1, 2, 2], dtype=dtype)
            W[:,0,:,:]=T_l.data
            pepo[tag_1].modify(data=W)    
        elif len(shape_1) == 5:
            W = np.zeros([bnd_,1,1, 2, 2], dtype=dtype)
            W[:, 0,0,:,:]=T_l.data
            pepo[tag_1].modify(data=W)    
        elif len(shape_1) == 6:
            W = np.zeros([bnd_,1,1,1, 2, 2], dtype=dtype)
            W[:,0,0,0,:,:]=T_l.data
            pepo[tag_1].modify(data=W)    

        if len(shape_2) == 4:
            if y==0:
                W = np.zeros([1, bnd_,2, 2], dtype=dtype)
                W[ 0,:,:,:]=T_r.data
                pepo[tag_2].modify(data=W)    
            else:
                W = np.zeros([bnd_, 1,2, 2], dtype=dtype)
                W[ :,0,:,:]=T_r.data
                pepo[tag_2].modify(data=W)    

        if len(shape_2) == 5:
            if y == 0:
                W = np.zeros([1, 1, bnd_, 2, 2], dtype=dtype)
                W[0, 0,:,:,:]=T_r.data
                pepo[tag_2].modify(data=W)    
            else:
                W = np.zeros([1, bnd_, 1, 2, 2], dtype=dtype)
                W[0,:,0,:,:]=T_r.data
                pepo[tag_2].modify(data=W)    

        if len(shape_2) == 6:
            W = np.zeros([1,1,bnd_,1, 2, 2], dtype=dtype)
            W[0, 0,:,0,:,:]=T_r.data
            pepo[tag_2].modify(data=W)    
    if x == x_ and y < y_:
        if len(shape_1) == 4:
            if x == 0:
                W = np.zeros([1, bnd_, 2, 2], dtype=dtype)
                W[0, :,:,:]=T_l.data
                pepo[tag_1].modify(data=W)    
            else:
                W = np.zeros([bnd_,1, 2, 2], dtype=dtype)
                W[:,0,:,:]=T_l.data
                pepo[tag_1].modify(data=W)    
        
        if len(shape_1) == 5:
            if x == Lx-1:
                W = np.zeros([bnd_,1,1, 2, 2], dtype=dtype)
                W[:,0,0,:,:]=T_l.data
                pepo[tag_1].modify(data=W)                    
            else:
                W = np.zeros([1,bnd_,1, 2, 2], dtype=dtype)
                W[0,:,0,:,:]=T_l.data
                pepo[tag_1].modify(data=W)    

        if len(shape_1) == 6:
            W = np.zeros([1,bnd_,1,1, 2, 2], dtype=dtype)
            W[0,:,0,0,:,:]=T_l.data
            pepo[tag_1].modify(data=W)    

        if len(shape_2) == 4:
            W = np.zeros([1,bnd_, 2, 2], dtype=dtype)
            W[0, :,:,:]=T_r.data
            pepo[tag_2].modify(data=W)    
        if len(shape_2) == 5:
            W = np.zeros([1,1, bnd_, 2, 2], dtype=dtype)
            W[0, 0,:,:,:]=T_r.data
            pepo[tag_2].modify(data=W)    
        if len(shape_2) == 6:
            W = np.zeros([1,1,1,bnd_, 2, 2], dtype=dtype)
            W[0, 0,0,:,:,:]=T_r.data
            pepo[tag_2].modify(data=W)    

            
    return pepo



def MPO_to_PEPO(x_mpo, Lx, Ly, cycle_peps= False):
    x_mpo = x_mpo.copy()
    dic = {}
    for i in range(Lx):
        if i%2 == 0:
            l = [ i*Ly + j for j in range(Ly)]
            cor = [ (i,j) for j in range(Ly)]
            dic = dic | dict(zip(l, cor))
        if i%2 == 1:
            l = [ i*Ly + j for j in range(Ly)]
            l.reverse()
            cor = [ (i,j) for j in range(Ly)]
            dic = dic | dict(zip(l, cor))

     

    for count, t in enumerate(x_mpo):
        x, y = dic[count]
        t.modify(tags = [f"I{x},{y}", f'X{x}', f'Y{y}'])
        t.reindex_({f"k{count}":f"k{x},{y}"})
        t.reindex_({f"b{count}":f"b{x},{y}"})
    
    for i in range(Lx-1):
        for j in range(Ly):
            if i % 2 == 0:
                if j < Ly-1:
                    x_mpo[f"I{i},{j}"].new_bond(x_mpo[f"I{i+1},{j}"], size=1)
            else:
                if j > 0:
                    x_mpo[f"I{i},{j}"].new_bond(x_mpo[f"I{i+1},{j}"], size=1)

    #  qtn.tensor_2d.TensorNetwork2DOperator
    x_mpo.view_as_(
        qtn.tensor_2d.PEPO,
        Lx=Lx, 
        Ly=Ly,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        upper_ind_id='k{},{}',
        lower_ind_id='b{},{}',
    )
    if cycle_peps:
        x_mpo = peps_cycle(x_mpo, int(1))

    return x_mpo

def MPO_to_PEPO_rotated(x_mpo, Lx, Ly, cycle_peps= False):
    x_mpo = x_mpo.copy()
    dic = {}
    for j in range(Ly):
        if j%2 == 0:
            l = [ j*Lx + i for i in range(Lx)]
            cor = [ (i,j) for i in range(Lx)]
            dic = dic | dict(zip(l, cor))
        else:
            l = [ j*Lx + i for i in range(Lx)]
            l.reverse()
            cor = [ (i,j) for i in range(Lx)]
            dic = dic | dict(zip(l, cor))

     

    for count, t in enumerate(x_mpo):
        x, y = dic[count]
        t.modify(tags = [f"I{x},{y}", f'X{x}', f'Y{y}'])
        t.reindex_({f"k{count}":f"k{x},{y}"})
        t.reindex_({f"b{count}":f"b{x},{y}"})
    
    for j in range(Ly-1):
        for i in range(Lx):
            if j % 2 == 0:
                if i < Lx-1:
                    x_mpo[f"I{i},{j}"].new_bond(x_mpo[f"I{i},{j+1}"], size=1)
            else:
                if i > 0:
                    x_mpo[f"I{i},{j}"].new_bond(x_mpo[f"I{i},{j+1}"], size=1)




    #  qtn.tensor_2d.TensorNetwork2DOperator
    x_mpo.view_as_(
        qtn.tensor_2d.PEPO,
        Lx=Lx, 
        Ly=Ly,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        upper_ind_id='k{},{}',
        lower_ind_id='b{},{}',
    )
    if cycle_peps:
        x_mpo = peps_cycle(x_mpo, int(1))
    return x_mpo




# mpo_approx_l = [0] * len(mpo_list_f)
# for count, mpo in enumerate(mpo_list_f):
#             if count == 0:
#                 mpo_approx_l[len(mpo_list_f) -1 - count ] =  mpo
#             else:
#                 print(len(mpo_list_f) -1 - count + 1, mpo_approx_l[len(mpo_list_f) -1 - count + 1 ].max_bond())
#                 mpo_f = mpo_approx_l[len(mpo_list_f) -1 - count + 1 ].apply(mpo)
#                 mpo_f.compress( "left", max_bond=260, cutoff=cutoff )
#                 mpo_approx_l[len(mpo_list_f) -1 - count ] =  mpo_f
# p_0.normalize()
# p_0=mpo_list_f[0].apply(p_0, form="left", compress=True,  max_bond=bond_dim, cutoff=cutoff)
# p_0=mpo_list_f[1].apply(p_0, form="left", compress=True,  max_bond=bond_dim, cutoff=cutoff)
# p_0=mpo_approx_l[2].apply(p_0, form="left", compress=True,  max_bond=bond_dim, cutoff=cutoff)
# p_0.normalize()

# print( energy_global(x_mpo, p_0, opt) / L   )


def gate_to_normpeps(mpo_list, p_0_, x_mpo, L):
    
    mpo_list = [ mpo.copy() for mpo in mpo_list]
    mpo_list_h = [ herm_inds_new(mpo) for mpo in mpo_list ]
    mpo_list_h.reverse()
    Depth_ = len(mpo_list)  
    
    p_0 = p_0_ * 1.0
    bond_inds = internal_inds(p_0)
    p_0_h=p_0.H
    p_0_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    Depth = 1 + Depth_ + 1 + Depth_ + 1 
    
    # mpo_middle = []
    # mpo_middle_h = []
    # for i in range(Iter):
    #     mpo_t = inds_new_mpolist(mpo_list)
    #     mpo_t_h = inds_new_mpolist(mpo_list_h)
    #     for t in mpo_t:
    #         mpo_middle.append(t)
    #     for t in mpo_t_h:
    #         mpo_middle_h.append(t)
            
    tn_l = [p_0] + mpo_list + [x_mpo] + mpo_list_h + [p_0_h]
    
    
    tn_list = []
    tags_list = []
    
    fix={}
    print("Depth", Depth, "L", L, "depth_mpo", Depth_)
    for iter_, tn in tqdm(enumerate(tn_l)):
        if "mpo_tag" in tn.tags:
            for count, t in enumerate(tn):
                    t.modify(tags = [f'I{iter_},{count}', f'X{iter_}', f'Y{count}'] + list(t.tags) )
                    tags_list.append(f'I{iter_},{count}')
                    fix = fix | {f'I{iter_},{count}': (iter_, count)}
            tn_list.append(tn)
        else:
            for count, t in enumerate(tn):
                    t.modify(tags = [f'I{iter_},{count}', f'X{iter_}', f'Y{count}']  )
                    tags_list.append(f'I{iter_},{count}')
                    fix = fix | {f'I{iter_},{count}': (iter_, count)}
            tn_list.append(tn)

    
    index=0
    for count, tn in tqdm(enumerate(tn_list)):
        if count == 0:
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{count}":index[count]   for count in range(L)}  )
        elif count == len(tn_list)-1:
            tn.reindex_({f"k{count}":index[count]    for count in range(L)}  )
        else:
            tn.reindex_({f"b{count}":index[count]    for count in range(L)}  )
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{count}":index[count]    for count in range(L)}  )
            
                
    # for count, tn in tqdm(enumerate(tn_list)):
    #     if count == 0:
    #         TN_peps = tn 
    #     else:
    #         TN_peps = TN_peps & tn
    
    TN_peps = qtn.TensorNetwork(tn_list)

    TN_peps.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=Depth, 
        Ly=L,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return TN_peps, tags_list, fix
    #print(TN_peps.Lx,TN_peps.Ly, TN_peps.max_bond) 
    #TN_peps.draw(color=tags_list, fix=fix, show_tags=False, legend=False, figsize=(18, 18))
    #print(TN_peps)

def gate_to_normpeps_norm(mpo_list, p_0_, Iter, x_mpo, L):
    mpo_list_h = [ herm_inds_new(mpo) for mpo in mpo_list ]
    mpo_list_h.reverse()
    Depth_ = len(mpo_list)  
    
    p_0 = p_0_ * 1.0
    bond_inds = internal_inds(p_0)
    p_0_h=p_0.H
    p_0_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    Depth = 1 + Depth_* Iter + Depth_* Iter + 1 
    
    mpo_middle = []
    mpo_middle_h = []
    for i in range(Iter):
        mpo_t = inds_new_mpolist(mpo_list)
        mpo_t_h = inds_new_mpolist(mpo_list_h)
        for t in mpo_t:
            mpo_middle.append(t)
        for t in mpo_t_h:
            mpo_middle_h.append(t)
            
    tn_l = [p_0] + mpo_middle + mpo_middle_h + [p_0_h]
    Depth = len(tn_l)
    
    tn_list = []
    tags_list = []
    
    fix={}
    print("Depth", Depth, "L", L, "depth_mpo", Depth_)
    for iter_, tn in tqdm(enumerate(tn_l)):
        for count, t in enumerate(tn):
                t.modify(tags = [f'I{iter_},{count}', f'X{iter_}', f'Y{count}'])
                tags_list.append(f'I{iter_},{count}')
                fix = fix | {f'I{iter_},{count}': (iter_, count)}
        tn_list.append(tn)
    
    
    index=0
    for count, tn in tqdm(enumerate(tn_list)):
        if count == 0:
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{count}":index[count]   for count in range(L)}  )
        elif count == len(tn_list)-1:
            tn.reindex_({f"k{count}":index[count]    for count in range(L)}  )
        else:
            tn.reindex_({f"b{count}":index[count]    for count in range(L)}  )
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{count}":index[count]    for count in range(L)}  )
            
                
    # for count, tn in tqdm(enumerate(tn_list)):
    #     if count == 0:
    #         TN_peps = tn 
    #     else:
    #         TN_peps = TN_peps & tn
    
    TN_peps = qtn.TensorNetwork(tn_list)

    TN_peps.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=Depth, 
        Ly=L,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return TN_peps, tags_list, fix
    #print(TN_peps.Lx,TN_peps.Ly, TN_peps.max_bond) 
    #TN_peps.draw(color=tags_list, fix=fix, show_tags=False, legend=False, figsize=(18, 18))
    #print(TN_peps)

def get_3d_pos(i, j, k, a=22, b=45, p=0.2):
    return (
        + i * math.cos(math.pi * a / 180) + j * math.cos(math.pi * b / 180) / 2**p,
        - i * math.sin(math.pi * a / 180) + j * math.sin(math.pi * b / 180) / 2**p + k       
    )

def gate_to_3dTN_(peps_, pepo , peps_h):

    Lx = peps_.Lx
    Ly = peps_.Ly
    L = Lx*Ly
    peps_ = peps_ * 1.

    pepo_l = [ inds_new_TN2d(pepo_) for pepo_ in pepo]

    
    bond_inds = internal_inds_TN2d(peps_h)
    peps_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    #apply
    tn_l = [peps_] + pepo_l + [peps_h]
    Depth = len(tn_l)
    tn_list = []
    tags_list = []
    fix={}
    
    for iter_, tn in tqdm(enumerate(tn_l)):
        
        if "pepo_tag" in tn.tags:
            for count, t in enumerate(tn):
                    #print(count)
                    rex = re.compile("I{},{}".format(r"\d+", r"\d+"))
                    I_tags = list(filter(rex.match, t.tags))
                    cor = re.findall( r"\d+", I_tags[0])
                    cor = list(map(int, cor))
                    #print(t.tags, cor)
                    t.modify(tags = [f'I{cor[0]},{cor[1]},{iter_}', f'X{cor[0]}', f'Y{cor[1]}', f'Z{iter_}'] + list(t.tags) )
                    tags_list.append(f'I{cor[0]},{cor[1]},{iter_}')
                    fix = fix | {f'I{cor[0]},{cor[1]},{count}': (cor[0], cor[1], iter_)}
        else:
            for count, t in enumerate(tn):
                    #print(count)
                    rex = re.compile("I{},{}".format(r"\d+", r"\d+"))
                    I_tags = list(filter(rex.match, t.tags))
                    cor = re.findall( r"\d+", I_tags[0])
                    cor = list(map(int, cor))
                    #print(t.tags, cor)
                    t.modify(tags = [f'I{cor[0]},{cor[1]},{iter_}', f'X{cor[0]}', f'Y{cor[1]}', f'Z{iter_}']+ list(t.tags) )
                    tags_list.append(f'I{cor[0]},{cor[1]},{iter_}')
                    fix = fix | {f'I{cor[0]},{cor[1]},{count}': (cor[0], cor[1], iter_)}
        
        tn_list.append(tn)
    
    index=0
    for count, tn in tqdm(enumerate(tn_list)):
        if count == 0:
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        elif count == len(tn_list)-1:
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        else:
            tn.reindex_({f"b{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) } )
    
    for count, tn in enumerate(tn_list):
        if count == 0:
            TN_peps = tn 
        else:
            TN_peps = TN_peps & tn
    
    def get_3d_pos(i, j, k, a=22, b=45, p=0.2):
        return (
            + i * math.cos(math.pi * a / 180) + j * math.cos(math.pi * b / 180) / 2**p,
            - i * math.sin(math.pi * a / 180) + j * math.sin(math.pi * b / 180) / 2**p + k       
        )
    pos_3d = {
        f'I{i},{j},{k}': get_3d_pos(i, j, k)
        for i in range(Lx)
        for j in range(Ly)
        for k in range(Depth)
    }
       
    TN_peps.view_as_(
        qtn.tensor_3d.TensorNetwork3DFlat,
        Lx=Lx, 
        Ly=Ly,
        Lz = Depth,
        site_tag_id='I{},{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        z_tag_id='Z{}',
    )
    return TN_peps, tags_list, pos_3d



def gate_to_3dTN_dangle(peps_, pepo ):

    Lx = peps_.Lx
    Ly = peps_.Ly
    L = Lx*Ly
    peps_ = peps_ * 1.

    pepo_l = [ inds_new_TN2d(pepo_) for pepo_ in pepo]

    
    #bond_inds = internal_inds_TN2d(peps_h)
    #peps_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    #apply
    tn_l = [peps_] + pepo_l #+ [peps_h]
    Depth = len(tn_l)
    tn_list = []
    tags_list = []
    fix={}
    
    for iter_, tn in tqdm(enumerate(tn_l)):
        
        if "pepo_tag" in tn.tags:
            for count, t in enumerate(tn):
                    #print(count)
                    rex = re.compile("I{},{}".format(r"\d+", r"\d+"))
                    I_tags = list(filter(rex.match, t.tags))
                    cor = re.findall( r"\d+", I_tags[0])
                    cor = list(map(int, cor))
                    #print(t.tags, cor)
                    t.modify(tags = [f'I{cor[0]},{cor[1]},{iter_}', f'X{cor[0]}', f'Y{cor[1]}', f'Z{iter_}'] + list(t.tags) )
                    tags_list.append(f'I{cor[0]},{cor[1]},{iter_}')
                    fix = fix | {f'I{cor[0]},{cor[1]},{count}': (cor[0], cor[1], iter_)}
        else:
            for count, t in enumerate(tn):
                    #print(count)
                    rex = re.compile("I{},{}".format(r"\d+", r"\d+"))
                    I_tags = list(filter(rex.match, t.tags))
                    cor = re.findall( r"\d+", I_tags[0])
                    cor = list(map(int, cor))
                    #print(t.tags, cor)
                    t.modify(tags = [f'I{cor[0]},{cor[1]},{iter_}', f'X{cor[0]}', f'Y{cor[1]}', f'Z{iter_}']+ list(t.tags) )
                    tags_list.append(f'I{cor[0]},{cor[1]},{iter_}')
                    fix = fix | {f'I{cor[0]},{cor[1]},{count}': (cor[0], cor[1], iter_)}
        
        tn_list.append(tn)
    
    index=0
    #length = len(tn_list)
    for count, tn in tqdm(enumerate(tn_list)):
        if count == 0:
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        elif count == len(tn_list)-1:
            tn.reindex_({f"b{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
             #tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        else:
            tn.reindex_({f"b{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) } )
    
    for count, tn in enumerate(tn_list):
        if count == 0:
            TN_peps = tn 
        else:
            TN_peps = TN_peps & tn
    
    def get_3d_pos(i, j, k, a=22, b=45, p=0.2):
        return (
            + i * math.cos(math.pi * a / 180) + j * math.cos(math.pi * b / 180) / 2**p,
            - i * math.sin(math.pi * a / 180) + j * math.sin(math.pi * b / 180) / 2**p + k       
        )
    pos_3d = {
        f'I{i},{j},{k}': get_3d_pos(i, j, k)
        for i in range(Lx)
        for j in range(Ly)
        for k in range(Depth+2)
    }
       
    TN_peps.view_as_(
        qtn.tensor_3d.TensorNetwork3DFlat,
        Lx=Lx, 
        Ly=Ly,
        Lz = Depth,
        site_tag_id='I{},{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        z_tag_id='Z{}',
    )
    return TN_peps, tags_list, pos_3d

















def gate_to_3dTN(peps_, pepo, x_pepo, Iter, L, Lx, Ly, Lz):


    peps_ = peps_ * 1.
    x_pepo = x_pepo * 1.

    pepo_l = [ inds_new_TN2d(pepo_) for pepo_ in pepo]
    pepo_h_l = [herm_inds_new_TN2d(pepo_) for pepo_ in pepo]
    pepo_h_l.reverse()
    
    
    bond_inds = internal_inds_TN2d(peps_)
    peps_h=peps_.H
    peps_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    #apply
    tn_l = [peps_] + pepo_l + [x_pepo] + pepo_h_l + [peps_h]
    Depth = len(tn_l)
    tn_list = []
    tags_list = []
    fix={}
    
    for iter_, tn in tqdm(enumerate(tn_l)):
        
        if "pepo_tag" in tn.tags:
            for count, t in enumerate(tn):
                    #print(count)
                    rex = re.compile("I{},{}".format(r"\d+", r"\d+"))
                    I_tags = list(filter(rex.match, t.tags))
                    cor = re.findall( r"\d+", I_tags[0])
                    cor = list(map(int, cor))
                    #print(t.tags, cor)
                    t.modify(tags = [f'I{cor[0]},{cor[1]},{iter_}', f'X{cor[0]}', f'Y{cor[1]}', f'Z{iter_}'] + list(t.tags) )
                    tags_list.append(f'I{cor[0]},{cor[1]},{iter_}')
                    fix = fix | {f'I{cor[0]},{cor[1]},{count}': (cor[0], cor[1], iter_)}
        else:
            for count, t in enumerate(tn):
                    #print(count)
                    rex = re.compile("I{},{}".format(r"\d+", r"\d+"))
                    I_tags = list(filter(rex.match, t.tags))
                    cor = re.findall( r"\d+", I_tags[0])
                    cor = list(map(int, cor))
                    #print(t.tags, cor)
                    t.modify(tags = [f'I{cor[0]},{cor[1]},{iter_}', f'X{cor[0]}', f'Y{cor[1]}', f'Z{iter_}'] )
                    tags_list.append(f'I{cor[0]},{cor[1]},{iter_}')
                    fix = fix | {f'I{cor[0]},{cor[1]},{count}': (cor[0], cor[1], iter_)}
        
        tn_list.append(tn)
    
    index=0
    for count, tn in tqdm(enumerate(tn_list)):
        if count == 0:
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        elif count == len(tn_list)-1:
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        else:
            tn.reindex_({f"b{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) } )
    
    for count, tn in enumerate(tn_list):
        if count == 0:
            TN_peps = tn 
        else:
            TN_peps = TN_peps & tn
    
    def get_3d_pos(i, j, k, a=22, b=45, p=0.2):
        return (
            + i * math.cos(math.pi * a / 180) + j * math.cos(math.pi * b / 180) / 2**p,
            - i * math.sin(math.pi * a / 180) + j * math.sin(math.pi * b / 180) / 2**p + k       
        )
    pos_3d = {
        f'I{i},{j},{k}': get_3d_pos(i, j, k)
        for i in range(Lx)
        for j in range(Ly)
        for k in range(Depth)
    }
       
    TN_peps.view_as_(
        qtn.tensor_3d.TensorNetwork3DFlat,
        Lx=Lx, 
        Ly=Ly,
        Lz = Depth,
        site_tag_id='I{},{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        z_tag_id='Z{}',
    )
    return TN_peps, tags_list, pos_3d
    

def gate_to_3dTN_norm(peps_, pepo, x_pepo, Iter, L, Lx, Ly, Lz):


    pepo_l = [ inds_new_TN2d(pepo_) for pepo_ in pepo]
    pepo_h_l = [herm_inds_new_TN2d(pepo_) for pepo_ in pepo]
    pepo_h_l.reverse()
    
    
    bond_inds = internal_inds_TN2d(peps_)
    peps_h=peps_.H
    peps_h.reindex_({i:qtn.rand_uuid()   for i in bond_inds}  )
    
    #apply
    tn_l = [peps_] + pepo_l + pepo_h_l + [peps_h]
    Depth = len(tn_l)
    tn_list = []
    tags_list = []
    fix={}
    
    for iter_, tn in tqdm(enumerate(tn_l)):
        for count, t in enumerate(tn):
                #print(count)
                rex = re.compile("I{},{}".format(r"\d+", r"\d+"))
                I_tags = list(filter(rex.match, t.tags))
                cor = re.findall( r"\d+", I_tags[0])
                cor = list(map(int, cor))
                #print(t.tags, cor)
                t.modify(tags = [f'I{cor[0]},{cor[1]},{iter_}', f'X{cor[0]}', f'Y{cor[1]}', f'Z{iter_}'])
                tags_list.append(f'I{cor[0]},{cor[1]},{iter_}')
                fix = fix | {f'I{cor[0]},{cor[1]},{count}': (cor[0], cor[1], iter_)}
        tn_list.append(tn)
    
    index=0
    for count, tn in tqdm(enumerate(tn_list)):
        if count == 0:
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        elif count == len(tn_list)-1:
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
        else:
            tn.reindex_({f"b{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) }  )
            index = [qtn.rand_uuid()   for count in range(L)]
            tn.reindex_({f"k{i},{j}":index[i*Ly + j]   for i,j in itertools.product(range(Lx), range(Ly)) } )
    
    for count, tn in enumerate(tn_list):
        if count == 0:
            TN_peps = tn 
        else:
            TN_peps = TN_peps & tn
    
    def get_3d_pos(i, j, k, a=22, b=45, p=0.2):
        return (
            + i * math.cos(math.pi * a / 180) + j * math.cos(math.pi * b / 180) / 2**p,
            - i * math.sin(math.pi * a / 180) + j * math.sin(math.pi * b / 180) / 2**p + k       
        )
    pos_3d = {
        f'I{i},{j},{k}': get_3d_pos(i, j, k)
        for i in range(Lx)
        for j in range(Ly)
        for k in range(Depth)
    }
       
    TN_peps.view_as_(
        qtn.tensor_3d.TensorNetwork3DFlat,
        Lx=Lx, 
        Ly=Ly,
        Lz = Depth,
        site_tag_id='I{},{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        z_tag_id='Z{}',
    )
    return TN_peps, tags_list, pos_3d



def peps_coulmn_details(peps, peps_, x_, Lx, Ly, bond_dim,  layer_tags=['KET', 'BRA'], canonize=True, progbar=False):
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})

    peps_.add_tag('KET')
    peps_H = peps_.conj().retag({'KET': 'BRA'})

    norm_ = peps_H & peps
    norm = pepsH & peps


    norm = norm.contract_boundary(max_bond=bond_dim, 
                                canonize=canonize, 
                                layer_tags=layer_tags,
                                sequence =  ['xmin', 'xmax' ], 
                                around=[ [x_, j] for j in range(Ly) ] , 
                                cutoff=1e-16, 
                                equalize_norms = False,
                                progbar =progbar
                                )
    norm_=norm_.contract_boundary(max_bond=bond_dim, 
                                canonize=canonize, 
                                layer_tags=layer_tags,
                                sequence =  ['xmin', 'xmax' ], 
                                around=[ (x_, j) for j in range(Ly) ], 
                                cutoff=1e-16, 
                                equalize_norms = False,
                                progbar =progbar,
                                )

    t_tid_l = [ norm._get_tids_from_tags([f'I{x_},{j}', 'KET'], which='all') for j in range(Ly) ]
    t_BRA_tid_l = [ norm._get_tids_from_tags([f'I{x_},{j}', 'BRA'], which='all' ) for j in range(Ly) ]
    t_tid_l_ = [ norm_._get_tids_from_tags([f'I{x_},{j}', 'KET'], which='all') for j in range(Ly) ]


    t_l = []
    for t_tid in t_tid_l:
        for tid in t_tid:
            t_l.append(norm.tensor_map[tid])  

    tags = []
    for count, tensor in enumerate(t_l):
        tensor.add_tag(f'opt{x_},{count}')
        tags.append(f'opt{x_},{count}')


    t_ket_inds_l = []
    for t_tid in t_tid_l:
        for tid in t_tid:
            t_ket_inds_l.append(norm.tensor_map[tid].inds)

    t_bra_inds_l = []
    for t_tid in t_BRA_tid_l:
        for tid in t_tid:
            t_bra_inds_l.append(norm.tensor_map[tid].inds)

    t_ket_inds_l_ = []
    for t_tid in t_tid_l_:
        for tid in t_tid:
            t_ket_inds_l_.append(norm_.tensor_map[tid].inds)

    for tid in list(t_tid_l + t_BRA_tid_l):
        norm.pop_tensor(*tid)

    for tid in t_tid_l_:
        t_test = norm_.pop_tensor(*tid)

    Tn = qtn.TensorNetwork(t_l)

    return Tn, t_bra_inds_l, t_ket_inds_l, t_ket_inds_l_, norm, norm_, tags



def peps_row_details(peps, peps_, x_, Lx, Ly, bond_dim,  layer_tags=['KET', 'BRA'], canonize=True, progbar=False):
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})

    peps_.add_tag('KET')
    peps_H = peps_.conj().retag({'KET': 'BRA'})

    norm_ = peps_H & peps
    norm = pepsH & peps


    norm = norm.contract_boundary(max_bond=bond_dim, 
                                canonize=canonize, 
                                layer_tags=layer_tags,
                                sequence =  ['ymin', 'ymax' ], 
                                around=[ (j, x_) for j in range(Lx) ] , 
                                cutoff=1e-16, 
                                equalize_norms = False,
                                progbar =progbar,
                                )
    norm_=norm_.contract_boundary(max_bond=bond_dim, 
                                canonize=canonize, 
                                layer_tags=layer_tags,
                                sequence =  ['ymin', 'ymax' ], 
                                around=[ (j, x_) for j in range(Lx) ], 
                                cutoff=1e-16, 
                                equalize_norms = False,
                                progbar =progbar,)

    t_tid_l = [ norm._get_tids_from_tags([f'I{j},{x_}', 'KET'], which='all') for j in range(Lx) ]
    t_BRA_tid_l = [ norm._get_tids_from_tags([f'I{j},{x_}', 'BRA'], which='all' ) for j in range(Lx) ]
    t_tid_l_ = [ norm_._get_tids_from_tags([f'I{j},{x_}', 'KET'], which='all') for j in range(Lx) ]


    t_l = []
    for t_tid in t_tid_l:
        for tid in t_tid:
            t_l.append(norm.tensor_map[tid])  

    tags = []
    for count, tensor in enumerate(t_l):
        tensor.add_tag(f'opt{count},{x_}')
        tags.append(f'opt{count},{x_}')


    t_ket_inds_l = []
    for t_tid in t_tid_l:
        for tid in t_tid:
            t_ket_inds_l.append(norm.tensor_map[tid].inds)

    t_bra_inds_l = []
    for t_tid in t_BRA_tid_l:
        for tid in t_tid:
            t_bra_inds_l.append(norm.tensor_map[tid].inds)

    t_ket_inds_l_ = []
    for t_tid in t_tid_l_:
        for tid in t_tid:
            t_ket_inds_l_.append(norm_.tensor_map[tid].inds)


    for tid in list(t_tid_l + t_BRA_tid_l):
        norm.pop_tensor(*tid)


    for tid in t_tid_l_:
        t_test = norm_.pop_tensor(*tid)



    Tn = qtn.TensorNetwork(t_l)

    return Tn, t_bra_inds_l, t_ket_inds_l, t_ket_inds_l_, norm, norm_, tags



def fidel_peps(peps, peps_, opt, max_bond=10, canonize = True, 
            layer_tags=['KET', 'BRA'], 
            equalize_norms=False, 
            max_separation=2, 
            sequence = None,
            progbar=False,
            mode = 'mps',
            ):
    
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    
    peps_.add_tag('KET')
    pepsH_ = peps_.conj().retag({'KET': 'BRA'})
    
    norm = pepsH & peps   #peps norm
    norm_overlap = pepsH_ & peps    #overlap
    norm_ = pepsH_ & peps_    #peps norm_fix
    
    peps_norm = norm.contract_boundary(max_bond=max_bond, 
                                   mode=mode,  #'full-bond', 'mps'
                                   sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance":canonize_distance},
                                   canonize=canonize,
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                   layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )
    
    peps_overlap = norm_overlap.contract_boundary(max_bond=max_bond, 
                                   mode=mode,  #'full-bond', 'mps'
                                   sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance":canonize_distance},
                                   canonize=canonize,
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                   layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )
 
    # peps_norm_ = norm_.contract_boundary(max_bond=max_bond, 
    #                                 mode=mode,  #'full-bond', 'mps'
    #                                sequence = sequence, # {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
    #                                #compress_opts = {"canonize_distance":canonize_distance},
    #                                canonize=canonize,
    #                                final_contract_opts={"optimize": opt}, 
    #                                cutoff=1e-14,
    #                                progbar=progbar,
    #                                 layer_tags=layer_tags,
    #                                max_separation=max_separation,
    #                                equalize_norms = equalize_norms,
    #                                #**{"equalize_norms":True}
    #                               )

    #print(norm_fix, peps_norm_)
    if abs(peps_norm) > 1.e-9: 
        peps = peps * (1/abs(peps_norm))**0.5
    else:
        print("warnning-norm", peps_norm)
    return peps, abs(peps_overlap)**2 / ( abs(peps_norm) )


def dis_peps(peps, peps_, opt, chi=20, canonize = True, 
            layer_tags=['KET', 'BRA'], 
            equalize_norms=False, 
            max_separation=1, 
            sequence = ['xmin', 'xmax', 'ymin', 'ymax'],
            progbar=False,
             mode='mps'
            ):
    
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    
    peps_.add_tag('KET')
    peps_H = peps_.conj().retag({'KET': 'BRA'})
    
    norm_ = peps_H & peps
    norm = pepsH & peps
    norm_fix = peps_H & peps_


    peps_norm = norm.contract_boundary(max_bond=chi, 
                                   mode=mode,  #'full-bond', 'mps'
                                   sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance":canonize_distance},
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                   layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )
    peps_norm_ = norm_.contract_boundary(max_bond=chi, 
                                   mode=mode,  #'full-bond', 'mps'
                                   sequence = sequence, # {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance":canonize_distance},
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                    layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )
    # peps_norm_fix = norm_fix.contract_boundary(max_bond=max_bond, 
    #                                mode='mps',  #'full-bond', 'mps'
    #                                 canonize=canonize,
    #                                sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
    #                                compress_opts = {"canonize_distance": canonize_distance},
    #                                final_contract_opts={"optimize": opt}, 
    #                                cutoff=1e-14,
    #                                progbar=progbar,
    #                                 layer_tags=layer_tags,
    #                                max_separation=max_separation,
    #                                equalize_norms = equalize_norms,
    #                                #**{"equalize_norms":True}
    #                               )

    peps_norm_sqrt = autoray.do("sqrt", peps_norm)
    peps_norm_sqrt_fix = autoray.do("sqrt", 1.)
    abs_peps_norm_ = autoray.do("abs", peps_norm_)
    peps_norm_sqrt_abs = autoray.do("abs", peps_norm_sqrt)
    peps_norm_sqrt_fix_abs = autoray.do("abs", peps_norm_sqrt_fix)



    infidelity = 1. - ( abs_peps_norm_/ (peps_norm_sqrt_abs*peps_norm_sqrt_fix_abs) )
    dis = autoray.do("abs", peps_norm + 1. - (peps_norm_+autoray.do("conj",peps_norm_)))
    return  abs(complex(dis)),  abs(complex(infidelity))

def dis_peps_total(peps, peps_, opt, chi=10, canonize = True, 
            layer_tags=['KET', 'BRA'], 
            equalize_norms=False, 
            max_separation=1, 
            canonize_distance=0,
            sequence = ['xmin', 'xmax'],
            progbar=False,
            ):
    
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    
    peps_.add_tag('KET')
    peps_H = peps_.conj().retag({'KET': 'BRA'})
    
    norm_ = peps_H & peps
    norm = pepsH & peps
    norm_fix = peps_H & peps_
    
    peps_norm = norm.contract_boundary(max_bond=chi, 
                                   mode='mps',  #'full-bond', 'mps'
                                   sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance":canonize_distance},
                                   canonize=canonize,
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                   layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )
    peps_norm_ = norm_.contract_boundary(max_bond=chi, 
                                   mode='mps',  #'full-bond', 'mps'
                                   sequence = sequence, # {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance":canonize_distance},
                                   canonize=canonize,
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                    layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )
    peps_norm_fix = norm_fix.contract_boundary(max_bond=chi, 
                                   mode='mps',  #'full-bond', 'mps'
                                    canonize=canonize,
                                   sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance": canonize_distance},
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                    layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )

    peps_norm_sqrt = autoray.do("sqrt", peps_norm)
    peps_norm_sqrt_fix = autoray.do("sqrt", peps_norm_fix)

    print("peps_norm", peps_norm)
    print("peps_norm_fix", peps_norm_fix)
    print("peps_norm_", peps_norm_)

    infidelity = 1. - ( abs(peps_norm_)/ (abs(peps_norm_sqrt_fix)*abs(peps_norm_sqrt)) )
    return abs(peps_norm + peps_norm_fix - 2. * peps_norm_.real),  infidelity, peps_norm_fix


def optimize_PEPS_coulmn(peps, Tn, t_bra_inds_l, t_ket_inds_l,t_ket_inds_l_, norm, norm_, Lx, Ly, opt, 
                        tags, x_, norm_fix=1, iterate = 20, 
                        cost_f = "distance",
                        optimizer = "L-BFGS-B",
                        progbar = False, loss_res = [],
                        ):
    #cost_f = "distance"
    #cost_f = "fidelity"
    to_backend = "numpy-cpu-double"
    to_backend_ = get_to_backend(to_backend)
    
    # print(peps)
    peps.apply_to_arrays(to_backend_)
    Tn.apply_to_arrays(to_backend_)


    
    threshold = 1.e-12
    def loss_(Tn, opt, cost_f, t_bra_inds_l, t_ket_inds_l, norm, norm_, norm_fix):
        TnH=Tn.H
        TnH=Tn.H
        map_ = []
        for count_, t in enumerate(Tn):
            map_.append({index:t_bra_inds_l[count_][count] for count, index in enumerate(t.inds)})
        for count_, t_H in enumerate(TnH):
            t_H.reindex_( map_[count_] )
        
        norm_cost = norm & Tn & TnH
        
        map_ = []
        for count_, t in enumerate(Tn):
            map_.append({index:t_ket_inds_l_[count_][count] for count, index in enumerate(t.inds)})
        
        t_list_ = []
        for count_, t in enumerate(Tn):
            t_list_.append(t.reindex( map_[count_] ))
        tn_ = qtn.TensorNetwork(t_list_)
        norm_cost_ = norm_ & tn_


        val_0 = norm_cost.contract(all, optimize=opt) 
        val_1 = norm_cost_.contract(all, optimize=opt)
        #print("loss", norm_fix, abs(val_0), abs(val_1) )
        val_2 = autoray.do("conj", val_1)
        val_3 = autoray.do("sqrt", val_0)
        val_4 = autoray.do("sqrt", norm_fix)

        val_1_abs = autoray.do("abs",val_1)
        val_3_abs = autoray.do("abs",val_3)
        val_4_abs = autoray.do("abs",val_4)

        if cost_f == "fidelity":  
            return abs( 1- ( val_1_abs/ (val_3_abs*val_4_abs) ) )
        elif cost_f == "distance":
            return abs( abs(val_0) + abs(norm_fix) -  val_1 - val_2)

    optimizer = qtn.TNOptimizer(
        Tn,                                # our initial input, the tensors of which to optimize
        loss_fn=loss_,
        loss_constants={"norm":norm, "norm_":norm_},  # additional tensor/tn kwargs
        loss_kwargs={'opt': opt, "t_bra_inds_l":t_bra_inds_l, "t_ket_inds_l":t_ket_inds_l, "cost_f":cost_f, "norm_fix":norm_fix},    
        autodiff_backend = "torch", #tensorflow,"torch", #'autograd',      # {'jax', 'tensorflow', 'autograd'}
        optimizer = "L-BFGS-B",  #'L-BFGS-B',               # supplied to scipy.minimize
        tags=tags,
        shared_tags=[],
        device = "cpu",
        progbar = progbar,
    )
    # Tn = optimizer.optimize(n=iterate, ftol=threshold, maxfun=10e+9, gtol= 1e-12, eps=1.49016e-08, maxls=400, iprint = 0, disp=False)
    Tn = optimizer.optimize_nlopt(n=iterate, tol=threshold, ftol_rel=threshold, ftol_abs=threshold)

    
    t_tid_l = [ Tn._get_tids_from_tags([f"opt{x_},{j}"], which='all') for j in range(Ly) ]

    # print(optimizer.losses, len(optimizer.losses) )

    loss_res += list(optimizer.losses)
    
    for count, t_tid in enumerate(t_tid_l):
        for tid in t_tid:
            peps[x_, count].modify(data=Tn.tensor_map[tid].data)
    # for count, t_tid in enumerate(t_tid_l):
    #     for tid in t_tid:
    #         print(peps[x_, count].data == Tn.tensor_map[tid].data)


    res = req_backend(progbar=False)
    # to_backend_ = get_to_backend(to_backend)
    to_backend_ = res["backend_"] 
    to_backend = res["backend"]
    
    peps.apply_to_arrays(to_backend_)
    Tn.apply_to_arrays(to_backend_)

    
    return peps


def optimize_PEPS_row(peps, Tn, t_bra_inds_l, t_ket_inds_l,t_ket_inds_l_, norm, norm_, Lx, Ly, opt, 
                        tags, x_, norm_fix=1, iterate = 20, 
                        cost_f = "distance", 
                        optimizer = "L-BFGS-B",
                        progbar = True, loss_res = []
                        ):
    #cost_f = "distance"
    #cost_f = "fidelity"
    threshold = 1.e-12
    def loss_(Tn, opt, cost_f, t_bra_inds_l, t_ket_inds_l, norm, norm_, norm_fix):
        TnH=Tn.H
        TnH=Tn.H
        map_ = []
        for count_, t in enumerate(Tn):
            map_.append({index:t_bra_inds_l[count_][count] for count, index in enumerate(t.inds)})
        for count_, t_H in enumerate(TnH):
            t_H.reindex_( map_[count_] )
        
        norm_cost = norm & Tn & TnH
        
        map_ = []
        for count_, t in enumerate(Tn):
            map_.append({index:t_ket_inds_l_[count_][count] for count, index in enumerate(t.inds)})
        
        t_list_ = []
        for count_, t in enumerate(Tn):
            t_list_.append(t.reindex( map_[count_] ))
        tn_ = qtn.TensorNetwork(t_list_)
        norm_cost_ = norm_ & tn_


        val_0 = norm_cost.contract(all, optimize=opt) 
        val_1 = norm_cost_.contract(all, optimize=opt)
        #print("loss", norm_fix, abs(val_0), abs(val_1) )
        val_2 = autoray.do("conj", val_1)
        val_3 = autoray.do("sqrt", val_0)
        val_4 = autoray.do("sqrt", norm_fix)
        if cost_f == "fidelity":  
            return abs( 1- ( abs(val_1)/ (abs(val_3)*abs(val_4)) ) )
        elif cost_f == "distance":
            return abs( abs(val_0) + abs(norm_fix) -  val_1 - val_2)

    optimizer = qtn.TNOptimizer(
        Tn,                                # our initial input, the tensors of which to optimize
        loss_fn=loss_,
        loss_constants={"norm":norm, "norm_":norm_},  # additional tensor/tn kwargs
        loss_kwargs={'opt': opt, "t_bra_inds_l":t_bra_inds_l, "t_ket_inds_l":t_ket_inds_l, "cost_f":cost_f, "norm_fix":norm_fix},    
        autodiff_backend = "jax", #tensorflow,"torch", #'autograd',      # {'jax', 'tensorflow', 'autograd'}
        optimizer = optimizer,  #'L-BFGS-B',               # supplied to scipy.minimize
        tags=tags,
        shared_tags=[],
        device = "cpu",
        progbar = progbar,
    )
    # Tn = optimizer.optimize(n=iterate, ftol=threshold, maxfun=10e+9, gtol= 1e-12, eps=1.49016e-09, maxls=400, iprint = 0, disp=False)
    Tn = optimizer.optimize_nlopt(n=iterate, tol=threshold, ftol_rel=threshold, ftol_abs=threshold)

    t_tid_l = [ Tn._get_tids_from_tags([f"opt{j},{x_}"], which='all') for j in range(Lx) ]


    
    loss_res += list(optimizer.losses)
    # print(optimizer.losses)

    
    for count, t_tid in enumerate(t_tid_l):
        for tid in t_tid:
            peps[count, x_].modify(data=Tn.tensor_map[tid].data)
    # for count, t_tid in enumerate(t_tid_l):
    #     for tid in t_tid:
    #         print(peps[x_, count].data == Tn.tensor_map[tid].data)

    
    return peps





def peps_normalize(peps, opt=None, copt=None, chi=20, 
            layer_tags=['KET', 'BRA'], 
            equalize_norms=False, mode = "exact",
            max_separation=1, 
            sequence = ['xmin', 'xmax', 'ymin', 'ymax'],
            progbar=False,
            ):

    def apply_hyperoptimized_compressed(tn, copt, chi, output_inds=None, tree_gauge_distance=4, progbar=False, 
                                        cutoff=1.e-12, equalize_norms=False):
        
        tn.full_simplify_(seq='R', split_method='svd', inplace=True)
        
        tree = tn.contraction_tree(copt)
        tn_ = tn.copy()
        
        flops = tree.contraction_cost(log=10)
        peak = tree.peak_size(log=2)
        
        tn_.contract_compressed_(
            optimize=tree,
            output_inds=output_inds,
            max_bond=chi,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
        )
        return tn_, (flops, peak)



    
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    norm = pepsH | peps
    
    
    if mode=="mps":
        peps_norm = norm.contract_boundary(max_bond=chi, 
                                       mode="mps",  #'full-bond', 'mps'
                                       sequence = sequence, 
                                       #compress_opts = {"canonize_distance":canonize_distance},
                                       final_contract_opts={"optimize": opt}, 
                                       cutoff=1e-14,
                                       progbar=progbar,
                                       layer_tags=layer_tags,
                                       max_separation=max_separation,
                                       equalize_norms = equalize_norms,
                                       #**{"equalize_norms":True})
                                          )

    if mode=="exact":
        peps_norm = norm.contract(all, optimize=opt)

    if mode=="hyper":
        overlap, (flops, peak) = apply_hyperoptimized_compressed(norm, copt, chi_bmps, cutoff=cutoff)
        # main, exp=(overlap.contract(all), overlap.exponent)
        peps_norm = overlap^all 


    
    peps = peps*peps_norm**(-0.5)
    return peps


def peps_norm(peps, opt=None, copt=None, chi=20, 
            layer_tags=['KET', 'BRA'], 
            equalize_norms=False, mode = "exact",
            max_separation=1, 
            sequence = ['xmin', 'xmax', 'ymin', 'ymax'],
            progbar=False,
            ):

    def apply_hyperoptimized_compressed(tn, copt, chi, output_inds=None, tree_gauge_distance=4, progbar=False, 
                                        cutoff=1.e-12, equalize_norms=False):
        
        tn.full_simplify_(seq='R', split_method='svd', inplace=True)
        
        tree = tn.contraction_tree(copt)
        tn_ = tn.copy()
        
        flops = tree.contraction_cost(log=10)
        peak = tree.peak_size(log=2)
        
        tn_.contract_compressed_(
            optimize=tree,
            output_inds=output_inds,
            max_bond=chi,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
        )
        return tn_, (flops, peak)



    
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    norm = pepsH | peps
    
    
    if mode=="mps":
        peps_norm = norm.contract_boundary(max_bond=chi, 
                                       mode="mps",  #'full-bond', 'mps'
                                       sequence = sequence, 
                                       #compress_opts = {"canonize_distance":canonize_distance},
                                       final_contract_opts={"optimize": opt}, 
                                       cutoff=1e-14,
                                       progbar=progbar,
                                       layer_tags=layer_tags,
                                       max_separation=max_separation,
                                       equalize_norms = equalize_norms,
                                       #**{"equalize_norms":True})
                                          )

    if mode=="exact":
        norm.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)
        peps_norm = norm.contract(all, optimize=opt)

    if mode=="hyper":
        overlap, (flops, peak) = apply_hyperoptimized_compressed(norm, copt, chi_bmps, cutoff=cutoff)
        # main, exp=(overlap.contract(all), overlap.exponent)
        peps_norm = overlap^all 


    
    
    return peps_norm




def peps_fitting_coulmn(peps, peps_, Lx, Ly, chi, opt, x_range,y_range, opt_distance, 
                        iterate = 20, 
                        cost_f = "distance", 
                        optimizer = "adam", normalize=False,
                        progbar = False, sequence=None, norm_fix = 1, loss_=[],
                        ):
    

    x_min = x_range[0]
    x_max = x_range[1] 
    y_min = y_range[0]
    y_max = y_range[1] 

    x_min = x_min - opt_distance
    x_max = x_max + opt_distance

    if  x_max > Lx:
        x_max = Lx 

    if  x_min < 0:
        x_min = 0
        x_max += 1
        if  x_max > Lx:
            x_max = Lx 

    
    dis_val, infidelity = dis_peps(peps, peps_, opt, 
                                            chi=chi, 
                                            canonize=True, 
                                            layer_tags=['KET', 'BRA'], 
                                            max_separation=1, 
                                            #canonize_distance=2,
                                            sequence = sequence,
                                            progbar = progbar,
                                            )
    
    
    print("start_optimization", "infidelity", infidelity, "optimizer", optimizer)
    print("coulmn_optimization", "cor", x_min, x_max)
    if abs(infidelity) < 1.e-6:
        print("return_peps_coulmn")
        return peps

    for x_ in range(x_min, x_max, 1):
        print("x_", x_)
        peps.balance_bonds_()
        #peps.equalize_norms_(1.0)

        Tn, t_bra_inds_l, t_ket_inds_l,t_ket_inds_l_, norm, norm_, tags = peps_coulmn_details(peps, peps_, x_, Lx, Ly, chi,progbar = progbar)
        peps = optimize_PEPS_coulmn(peps, Tn, t_bra_inds_l, t_ket_inds_l,t_ket_inds_l_, norm, norm_, 
                                    Lx, Ly, opt,tags, x_, 
                                    norm_fix=norm_fix, iterate = iterate,
                                    cost_f = cost_f, #"fidelity"
                                    optimizer = optimizer,
                                    progbar = progbar, loss_res = loss_,
                                    )
        
        # peps.balance_bonds_()
        if normalize:
            peps = peps_normalize(peps, opt, chi=chi, sequence=sequence,progbar = progbar)
    
        dis_val, infidelity = dis_peps(peps, peps_, opt, 
                                                chi=chi, 
                                                canonize=True, 
                                                layer_tags=['KET', 'BRA'], 
                                                max_separation=1, 
                                                #canonize_distance=2,
                                                sequence = sequence,
                                                progbar = progbar,
                                                )
        print("final", "infidelity", infidelity)
        loss_.append(infidelity)
    return peps


def peps_fitting_row(peps, peps_, Lx, Ly, chi, opt, x_range, y_range, opt_distance, iterate = 50, 
                    cost_f = "distance", optimizer = "cg", normalize=False,
                    progbar = True, sequence=None, norm_fix = 1, loss_ = [],
                    ):

    x_min = x_range[0]
    x_max = x_range[1] 
    y_min = y_range[0]
    y_max = y_range[1]

    y_min = y_min - opt_distance
    y_max = y_max + opt_distance

    if  y_max > Ly:
        y_max = Ly 

    if  y_min < 0:
        y_min = 0
        y_max += 1
        if  y_max > Ly:
            y_max = Ly 

    dis_val, infidelity = dis_peps(peps, peps_, opt, 
                                             chi=chi, canonize=True, 
                                             layer_tags=['KET', 'BRA'], 
                                             max_separation=1, 
                                             #canonize_distance=2,
                                             progbar = progbar,
                                             sequence=sequence)
    
    print("row_optimization", "infidelity", infidelity, "optimizer", optimizer)
    print("row_optimization", "cor", y_min, y_max)
    if abs(infidelity) < 1.e-6:
        print("return_peps_row")
        return peps
    
    for y_ in range(y_min, y_max, 1):
        print("y_", y_)
        peps.balance_bonds_()
        #peps.equalize_norms_(4.0)
        Tn, t_bra_inds_l, t_ket_inds_l,t_ket_inds_l_, norm, norm_, tags = peps_row_details(peps, peps_, y_, Lx, Ly, chi,progbar = progbar)
        peps = optimize_PEPS_row(peps, Tn, t_bra_inds_l, t_ket_inds_l,t_ket_inds_l_, norm, norm_, 
                                    Lx, Ly, opt,tags, y_, 
                                    norm_fix=norm_fix, iterate = iterate,
                                    cost_f = cost_f,
                                    optimizer = optimizer,
                                    progbar = progbar, loss_res=loss_,
                                    )
        #peps.balance_bonds_()
        if normalize:
            peps = peps_normalize(peps, opt, chi=chi,sequence=sequence, progbar = progbar)
        
    
        dis_val, infidelity = dis_peps(peps, peps_, opt, 
                                                chi=chi, 
                                                canonize=True, 
                                                layer_tags=['KET', 'BRA'], 
                                                max_separation=1, 
                                                #canonize_distance=2,
                                                sequence = sequence,
                                                progbar = progbar,
                                                )
        loss_.append(infidelity)
        print("final", "infidelity", infidelity)

    return peps

def peps_fit(peps, peps_fix, chi, opt,
                        iter_=4,
                        iterate = 220,
                        max_seperation =1, 
                        normalize=True,
                        progbar=False,
                        cost_f = "fidelity",
                        sequence = ["xmin", "xmax","ymin", "ymax" ],  optimizer = "L-BFGS-B",
                        ):

    loss_ = []
    Lx = peps.Lx
    Ly = peps.Ly
    x_range = [0, Lx]
    y_range = [0, Ly]
    opt_distance = 4

    if normalize:
        peps = peps_normalize(peps, opt, chi=chi, max_separation=max_seperation)


    dis_val, infidelity = dis_peps(peps, peps_fix, opt, 
                                            chi=chi, 
                                            layer_tags=['KET', 'BRA'], 
                                            max_separation=max_seperation, 
                                            sequence = sequence,
                                            progbar = progbar,  
                                            )

        
    #print("norm_fix", 2**(Lx*Ly), abs(norm_fix) )
    #print("peps_fix", peps_fix.show())
    #print("peps", peps.show())
    print("---------------dmrg--------------")
    print(f"dmrg: chi={chi}")
    print("start", "infidelity", abs(complex(infidelity)) )
    loss_.append(infidelity)
    
    if abs(infidelity) < 1.e-6:
        print("return_peps")
        return peps, abs(complex(infidelity))


    
    for _ in tqdm(range(iter_)):
        if min(y_range) != max(y_range):
            peps = peps_fitting_coulmn(peps, peps_fix, Lx, Ly, chi, opt, x_range, y_range, opt_distance, iterate = iterate, sequence = sequence,
                                            cost_f = cost_f, optimizer = optimizer, normalize=normalize, progbar=progbar, loss_=loss_)
        
        
        if min(x_range) != max(x_range):
            peps = peps_fitting_row(peps, peps_fix, Lx, Ly, chi, opt, x_range, y_range, opt_distance, iterate = iterate, sequence = sequence,
                                     cost_f = cost_f, optimizer =optimizer, normalize=normalize,progbar=progbar, loss_=loss_)


    dis_val, infidelity = dis_peps(peps, peps_fix, opt, 
                                            chi=chi, 
                                            max_separation=max_seperation, 
                                            sequence = sequence,
                                            progbar = progbar,
                                            )

        
        #print("norm_fix", 2**(Lx*Ly), abs(norm_fix) )
    if normalize:
        peps = peps_normalize(peps, opt, chi=chi, max_separation=max_seperation)

    print("final", "infidelity", abs(complex(infidelity)) )
    print("---------------dmrg-end--------------")
    
    return peps, abs(infidelity), loss_




def pepo_cal(peps, x_pepo, chi, opt, 
                progbar = False, 
                max_separation=2, 
                mode='mps', 
                cutoff=1e-16,
                equalize_norms = False,
                sequence = ['xmin', 'xmax', 'ymin', 'ymax'],
                normalize=False,
                Falt = False, 
                ):
    
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})

    peps_ = peps.copy()
    peps_ = x_pepo.apply(peps_)

    #pepsH.reindex_({f"k{i},{j}":f"b{i},{j}" for (i, j)  in itertools.product(range(peps.Lx), range(peps.Ly))})
    tn = pepsH  & peps_
    norm = pepsH  & peps
    #tn.flatten(fuse_multibonds=True, inplace=True)
    layer_tags = None
    if Falt:
        tn.flatten(fuse_multibonds=True, inplace=True)
    else:
        layer_tags = ["KET", "BRA"]

    tn.contract_boundary_( max_bond=chi, 
                                mode=mode,  #'full-bond', 'mps'
                                sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                #compress_opts = {"canonize_distance":2},
                                #around = [site,],
                                canonize = True,
                                final_contract_opts = {"optimize": opt}, 
                                cutoff = cutoff,
                                progbar = progbar,
                                layer_tags = layer_tags,
                                max_separation = max_separation,
                                equalize_norms = equalize_norms,
                                final_contract = True,
                                #**{"equalize_norms":True}
                                )

    tn = tn^all
    if normalize:
        layer_tags = None
        if Falt:
            norm.flatten(fuse_multibonds=True, inplace=True)
        else:
            layer_tags = ["KET", "BRA"]

        norm.contract_boundary_( max_bond=chi, 
                            mode=mode,  #'full-bond', 'mps'
                            sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                            #compress_opts = {"canonize_distance":2},
                            #around = [site,],
                            canonize = True,
                            final_contract_opts = {"optimize": opt}, 
                            cutoff = cutoff,
                            progbar = progbar,
                            layer_tags = layer_tags,
                            max_separation = max_separation,
                            equalize_norms = equalize_norms,
                            final_contract = True,
                            #**{"equalize_norms":True}
                            )
        norm = norm^all

        return tn/norm, norm

    return tn






def dpepo_cal(xpepo, xpepo_, chi, opt, 
                progbar = False, 
                max_separation=2, 
                mode='mps', 
                cutoff=1e-16,
                equalize_norms = False,
                sequence = ['xmin', 'xmax'],
                normalize=False,
                Falt = False, 
                ):
    xpepo_ = xpepo_ * 1.
    xpepo = xpepo * 1.

    norm = trace_2d(xpepo)

    xpepo_.reindex_({f"b{i},{j}":f"g{i},{j}" for (i, j)  in itertools.product(range(xpepo_.Lx), range(xpepo_.Ly))})
    xpepo_.reindex_({f"k{i},{j}":f"b{i},{j}" for (i, j)  in itertools.product(range(xpepo_.Lx), range(xpepo_.Ly))})

    tn = xpepo_  & xpepo
    tn.flatten(fuse_multibonds=True, inplace=True)
    tn.reindex_({f"g{i},{j}":f"b{i},{j}" for (i, j)  in itertools.product(range(xpepo_.Lx), range(xpepo_.Ly))})

    
    tn = trace_2d(tn)

    tn.contract_boundary_( max_bond=chi, 
                                mode=mode,  #'full-bond', 'mps'
                                sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                #compress_opts = {"canonize_distance":2},
                                #around = [site,],
                                canonize = True,
                                final_contract_opts = {"optimize": opt}, 
                                cutoff = cutoff,
                                progbar = progbar,
                                #layer_tags = layer_tags,
                                max_separation = max_separation,
                                equalize_norms = equalize_norms,
                                final_contract = True,
                                #**{"equalize_norms":True}
                                )

    tn = tn^all

    norm.contract_boundary_( max_bond=chi, 
                        mode=mode,  #'full-bond', 'mps'
                        sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                        #compress_opts = {"canonize_distance":2},
                        #around = [site,],
                        canonize = True,
                        final_contract_opts = {"optimize": opt}, 
                        cutoff = cutoff,
                        progbar = progbar,
                        #layer_tags = layer_tags,
                        max_separation = max_separation,
                        equalize_norms = equalize_norms,
                        final_contract = True,
                        #**{"equalize_norms":True}
                        )
    norm = norm^all
    
    return complex(tn/norm), abs(complex(norm))

















def pepo_cal_exact(peps, x_pepo, chi, opt, bond_dim_simple = 8, bnd_thershold = 32, canonize_distance=4, max_separation=2):
    peps_X = x_pepo.apply(peps)
    peps_X.balance_bonds_()

    print("peps_X", peps_X.max_bond(), "peps", peps.max_bond())

    peps_X.add_tag('KET')
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})

    norm = pepsH & peps_X
    norm.squeeze(fuse=True, inplace=True)
 

    return norm.contract(all, optimize=opt)








def peps_local_cal(peps, chi, opt, dtype="complex128"):
    Z = qu.pauli('Z',dtype=dtype) 
    X = qu.pauli('X',dtype=dtype) 
    Y = np.array([[0, -1],[1,0]])
    I = qu.pauli('I',dtype=dtype)
    Lx = peps.Lx
    single_site = [ ((i,j)) for i,j in itertools.product([0,4,8,12],[1,5,9]) ] + [ ((i,j)) for i,j in itertools.product([2,6,10,14],[3,7,11]) ] 
    
    terms = { (i,0): Z  for i in range(0,Lx-1,1) }
    terms = terms | { (i,2) : Z   for i in range(0,Lx,1) }
    terms = terms | { (i,4) : Z   for i in range(0,Lx,1) }
    terms = terms | { (i,6) : Z   for i in range(0,Lx,1) }
    terms = terms | { (i,8) : Z   for i in range(0,Lx,1) }
    terms = terms | { (i,10) : Z   for i in range(0,Lx,1) }
    terms = terms | { (i,12) : Z   for i in range(1,Lx,1) }
    terms = terms | { i : Z for i in single_site}
    
    
    
    res_terms = peps.compute_local_expectation(terms, max_bond=chi, cutoff=1e-16, canonize=True, mode='mps', 
                            layer_tags=('KET', 'BRA'), normalized=True, autogroup=True, 
                            contract_optimize=opt, return_all=False, 
                            plaquette_envs=None, plaquette_map=None)
    
    return res_terms




















def cube_z(TN_peps, Lx, Ly, Lz, bond_dim, canonize_distance, opt, canonize, equalize_norms):
    Lx = TN_peps.Lx
    Ly = TN_peps.Ly
    Lz = TN_peps.Lz
    
    X_ = TN_peps.contract_boundary(max_bond=bond_dim, 
                             cutoff=1e-16, 
                              mode='peps',  #peps
                              canonize=canonize, 
                              #compress_opts = {"canonize_distance": canonize_distance}, 
                              sequence=["zmin", "zmax"], xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, 
                              max_separation=1, 
                              max_unfinished=0,
                              around=None, 
                              equalize_norms=equalize_norms, 
                              final_contract=False, 
                              final_contract_opts={"optimize": opt},
                              progbar=True, 
                              inplace=False,                  
                         )
    
    for t in X_:
        tags = list(t.tags)
        x_tag = [x for x in tags if x.startswith('X')]
        y_tag = [x for x in tags if x.startswith('Y')]
        x_tag=x_tag[0]
        y_tag=y_tag[0]
        x=int(re.findall(r'\d+', x_tag)[0])
        y=int(re.findall(r'\d+', y_tag)[0])
        #print(x, y, type(x_tag), y_tag )
        #print(tags)
        t.modify(tags = [x_tag, y_tag, f"I{x},{y}"])
    
    
    X_.view_as_(
            qtn.tensor_2d.TensorNetwork2DFlat,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
        )

    X_.flatten(fuse_multibonds=True, inplace=True)
    
    return X_

def cube_x(TN_peps, Lx, Ly, Lz, bond_dim, canonize_distance, opt, canonize, equalize_norms):
    Lx = TN_peps.Lx
    Ly = TN_peps.Ly
    Lz = TN_peps.Lz
    
    X_ = TN_peps.contract_boundary(max_bond=bond_dim, 
                             cutoff=1e-16, 
                              mode='peps',  #peps
                              canonize=canonize, 
                              #compress_opts = {"canonize_distance": canonize_distance}, 
                              sequence=["xmin", "xmax"], xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, 
                              max_separation=1, 
                              max_unfinished=0,
                              around=None, 
                              equalize_norms=equalize_norms, 
                              final_contract=False, 
                              final_contract_opts={"optimize": opt},
                              progbar=True, 
                              inplace=False,                  
                         )
    
    for t in X_:
        tags = list(t.tags)
        x_tag = [x for x in tags if x.startswith('Z')]
        y_tag = [x for x in tags if x.startswith('Y')]
        x_tag=x_tag[0]
        y_tag=y_tag[0]
        x=int(re.findall(r'\d+', x_tag)[0])
        y=int(re.findall(r'\d+', y_tag)[0])
        #print(x, y, type(x_tag), y_tag )
        #print(tags)
        t.modify(tags = [f"X{x}",f"Y{y}", f"I{x},{y}"])
    
    
    X_.view_as_(
            qtn.tensor_2d.TensorNetwork2DFlat,
            Lx=Lz, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
        )

    X_.flatten(fuse_multibonds=True, inplace=True)
    
    return X_



def cube_y(TN_peps, Lx, Ly, Lz, bond_dim, canonize_distance, opt, canonize, equalize_norms):
    Lx = TN_peps.Lx
    Ly = TN_peps.Ly
    Lz = TN_peps.Lz
    
    X_ = TN_peps.contract_boundary(max_bond=bond_dim, 
                             cutoff=1e-16, 
                              mode='peps',  #peps
                              canonize=canonize, 
                              #compress_opts = {"canonize_distance": canonize_distance}, 
                              sequence=["ymin", "ymax"], xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, 
                              max_separation=1, 
                              max_unfinished=0,
                              around=None, 
                              equalize_norms=equalize_norms, 
                              final_contract=False, 
                              final_contract_opts={"optimize": opt},
                              progbar=True, 
                              inplace=False,                  
                         )
    
    for t in X_:
        tags = list(t.tags)
        x_tag = [x for x in tags if x.startswith('X')]
        y_tag = [x for x in tags if x.startswith('Z')]
        x_tag=x_tag[0]
        y_tag=y_tag[0]
        x=int(re.findall(r'\d+', x_tag)[0])
        y=int(re.findall(r'\d+', y_tag)[0])
        #print(x, y, type(x_tag), y_tag )
        #print(tags)
        t.modify(tags = [x_tag, y_tag, f"I{x},{y}"])
    
    
    X_.view_as_(
            qtn.tensor_2d.TensorNetwork2DFlat,
            Lx=Lx, 
            Ly=Lz,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Z{}',
        )

    X_.flatten(fuse_multibonds=True, inplace=True)
    
    return X_


# def norm_mpo_select(X_, L):

#     norm = X_.select("mpo_tag", which='!all', virtual=True)
#     mpo = X_.select("mpo_tag", which='all', virtual=True)
#     MPO_I=qtn.MPO_identity(L, phys_dim=2)
#     MPO_I.add_tag("mpo_I", where=None, which='all')
#     for count in range(L):
#         MPO_I[count].modify(inds= mpo[f"O{count}"].inds)
#     return norm & MPO_I


def norm_mpo_select(X_, L):
    MPO_I=qtn.MPO_identity(L, phys_dim=2)
    X_ = X_.copy()
    for count in range(L):
        t = X_.select(f"O{count}", which='all', virtual=True)
        for t_ in t:
            t_.modify(data = MPO_I[count].data)
    return X_



def cal_local_O_tn(tn, site, opt, pauli="Z"):
    Z_ = tn.select(f"O{site}", which='all', virtual=True)
    tn_ = tn.select(f"O{site}", which='!all', virtual=True)

    t = Z_[f"O{site}"]
    inds_l=list(t.inds)
    I_ = t * 1. 
    Z_re = t * 1.
    Z_ = t * 1.
    Z_inv = t * 1.

    Z_ = qtn.Tensor(data=qu.pauli(pauli), inds=[inds_l[-2], "y_"])
    I_ = qtn.Tensor(data=qu.pauli("I"), inds=[inds_l[-2], "y_"])
    Z_inv = qtn.Tensor(data=qu.pauli(pauli), inds=["y_", "x_"])
    Z_re.reindex_({inds_l[-2]: "x_"})

    Z_tn = (tn_ & Z_ & Z_inv & Z_re )
    I_tn = (tn_ & I_ & Z_inv & Z_re )
    
    #Z_tn.squeeze(fuse=True, inplace=True)
    #I_tn.squeeze(fuse=True, inplace=True)

    Z_cal = (Z_tn).contract(all, optimize=opt)
    norm_cal = (I_tn).contract(all, optimize=opt)
    
    return Z_cal, norm_cal

def cal_local_O_3dtn(tn, site, opt, pauli="Z"):
    Z_ = tn.select(f"O{site}", which='all', virtual=True)
    tn_ = tn.select(f"O{site}", which='!all', virtual=True)

    t = Z_[f"O{site}"]
    inds_l = list(t.inds)
    #print(inds_l, t.data)
    I_ = t * 1. 
    Z_re = t * 1.
    Z_ = t * 1.
    Z_inv = t * 1.

    Z_ = qtn.Tensor(data=qu.pauli(pauli), inds=[inds_l[-2], "y_"], tags = ["add"] )
    I_ = qtn.Tensor(data=qu.pauli("I"), inds=[inds_l[-2], "y_"], tags = ["add"])
    Z_inv = qtn.Tensor(data=qu.pauli(pauli), inds=["y_", "x_"], tags = ["add"])
    Z_re.reindex_({inds_l[-2]: "x_"})

    Z_tn = (tn_ & Z_ & Z_inv & Z_re )
    I_tn = (tn_ & I_ & Z_inv & Z_re )

    #Z_tn.squeeze(fuse=True, inplace=True)
    #I_tn.squeeze(fuse=True, inplace=True)

    Z_cal = (Z_tn).contract(all, optimize=opt)
    norm_cal = (I_tn).contract(all, optimize=opt)

    return Z_cal, norm_cal


def re_tn_ymin(tn_g, Lx, Ly, max_separation):
    Ly_ = max_separation + 1
    start_ = Ly - Ly_ + 1
    
    #print("start_1", start_, "Ly_", Ly_)

    for i in range(Lx):
        tn = tn_g.select([f"I{i},{0}", f"X{i}", f"Y{0}"], which="all")
        for count, t in enumerate(tn):
            t.modify(tags = [f"I{i},{0}", f"X{i}", f"Y{0}"])
    Y_map = { f"Y{i}": f"Y{i-start_ + 1}"  for i in range(start_, Ly)}
    tn_g.retag_(Y_map)
    for i in range(Lx):
        I_map = { f"I{i},{j}": f"I{i},{j-start_ + 1}"  for j in range(start_ , Ly)}
        tn_g.retag_(I_map)

    tn_g.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=Lx,
        Ly=Ly_,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return tn_g


def re_tn_ymax(tn_g, Lx, Ly, max_separation):
    Ly_ = max_separation + 1
    start_ = Ly - Ly_ + 1

    #print("start_2", start_, "Ly_", Ly_)
    for i in range(Lx):
        tn = tn_g.select([f"I{i},{Ly-1}", f"X{i}", f"Y{Ly-1}"], which="all")
        for count, t in enumerate(tn):
            t.modify(tags = [f"I{i},{Ly_-1}", f"X{i}", f"Y{Ly_-1}"])

    tn_g.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=Lx,
        Ly=Ly_,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return tn_g


def re_tn_xmin(tn_g, Lx, Ly, max_separation):
    Lx_ = max_separation + 1
    start_ = Lx - Lx_ + 1
    for j in range(Ly):
        tn = tn_g.select([f"I{0},{j}", f"X{0}", f"Y{j}"], which="all")
        for count, t in enumerate(tn):
            t.modify(tags = [f"I{0},{j}", f"X{0}", f"Y{j}"])
    X_map = { f"X{i}": f"X{i-start_ + 1}"  for i in range(start_, Lx)}
    tn_g.retag_(X_map)
    for j in range(Ly):
        I_map = { f"I{i},{j}": f"I{i-start_ + 1},{j}"  for i in range(start_ , Lx)}
        tn_g.retag_(I_map)

    tn_g.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=Lx_,
        Ly=Ly,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return tn_g

def re_tn_xmax(tn_g, Lx, Ly, max_separation):
    Lx_ = max_separation + 1
    for j in range(Ly):
        tn = tn_g.select([f"I{Lx-1},{j}", f"X{Lx-1}", f"Y{j}"], which="all")
        for count, t in enumerate(tn):
            t.modify(tags = [f"I{Lx_-1},{j}", f"X{Lx_-1}", f"Y{j}"])
    #X_map = { f"X{i}": f"X{i-start_ + 1}"  for i in reversed(range(start_, Lx))}
    #tn_g.retag_(X_map)
    #for j in range(Ly):
        #I_map = { f"I{i},{j}": f"I{i-start_ + 1},{j}"  for i in range(start_ , Lx)}
        #tn_g.retag_(I_map)
    tn_g.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=Lx_,
        Ly=Ly,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return tn_g


def x_inward_compress_min(tn_, bond_dim = 10, max_distance=4, opt = 'auto-hq', inward_=4, mode="mps", equalize_norms = True, canonize = True):
    tn_ = tn_.copy()
    Lx = tn_.Lx
    Ly = tn_.Ly
    max_separation = Lx - inward_ 

    tn_g = tn_.canonize_around([f"I{0},{j}" for j in range(Ly)], which='any', 
                            max_distance=max_distance,
                            absorb='right',
                            link_absorb='both',
                            equalize_norms=True, 
                            inplace=False)

    tn_g = tn_g.contract_boundary(max_bond = bond_dim, 
                                    mode = mode,  #'full-bond', 'mps'
                                    sequence =  {'xmin'}, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},                               
                                    #sequence =  {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                    #compress_opts = {"canonize_distance": canonize_distance},
                                    around = [(inward_, 0), (inward_, Ly-1), (Lx-1, 0), (Lx-1, Ly-1)],
                                    final_contract_opts = {"optimize": opt}, 
                                    cutoff = 1e-14,
                                    progbar = True,
                                    #max_separation = max_separation,
                                    equalize_norms = equalize_norms,
                                    canonize = canonize,
                                    final_contract = False,
                                    #**{"equalize_norms":True}
                                    )

    tn_g = re_tn_xmin(tn_g, Lx, Ly, max_separation)
    return tn_g


def x_inward_compress_max(tn_, bond_dim = 10, max_distance=4, opt = 'auto-hq', inward_=4, mode="mps", equalize_norms = True, canonize = True):
    tn_ = tn_.copy()
    Lx = tn_.Lx
    Ly = tn_.Ly
    tn_g = tn_.canonize_around([f"I{Lx -1},{j}" for j in range(Ly)], which='any', 
                            max_distance=max_distance,
                            absorb='right',
                            link_absorb='both',
                            equalize_norms=True, 
                            inplace=False)

    max_separation = Lx - inward_ 

    tn_g = tn_g.contract_boundary(max_bond = bond_dim, 
                                    mode = mode,  #'full-bond', 'mps'
                                    sequence =  {'xmax'}, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},                               
                                    #sequence =  {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                    #compress_opts = {"canonize_distance": canonize_distance},
                                    around = [(0, 0), (0, Ly-1), (Lx-inward_-1, 0), (Lx-inward_-1, Ly-1) ],
                                    final_contract_opts = {"optimize": opt}, 
                                    cutoff = 1e-14,
                                    progbar = True,
                                    #max_separation = max_separation,
                                    equalize_norms = equalize_norms,
                                    canonize = canonize,
                                    final_contract = False,
                                    #**{"equalize_norms":True}
                                    )

    tn_g = re_tn_xmax(tn_g, Lx, Ly, max_separation)
    return tn_g


def y_inward_compress_min(tn_, bond_dim = 10, max_distance=4, opt = 'auto-hq', inward_=4, mode="mps", equalize_norms = True, canonize = True):
    
    tn_ = tn_.copy()
    Lx = tn_.Lx
    Ly = tn_.Ly
    max_separation = Ly - inward_ 
    #print("max_separation_1", max_separation, (0, inward_), (Lx-1, inward_), (0, Ly-1), (Lx-1, Ly-1))

    tn_g = tn_.canonize_around([f"I{i},{0}" for i in range(Lx)], which='any', 
                            max_distance=max_distance,
                            absorb='right',
                            link_absorb='both',
                            equalize_norms=True, 
                            inplace=False)


    tn_g = tn_g.contract_boundary(  max_bond = bond_dim, 
                                    mode = mode,  #'full-bond', 'mps'
                                    #sequence =  {'ymin'}, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},                               
                                    #sequence =  {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                    #compress_opts = {"canonize_distance": canonize_distance},
                                    around = [(0, inward_), (Lx-1, inward_), (0, Ly-1), (Lx-1, Ly-1)],
                                    final_contract_opts = {"optimize": opt}, 
                                    cutoff = 1e-14,
                                    progbar = True,
                                    #max_separation = max_separation,
                                    equalize_norms = equalize_norms,
                                    canonize = canonize,
                                    final_contract = False,
                                    #**{"equalize_norms":True}
                                    )
        
    #print("first", tn_g)
    tn_g = re_tn_ymin(tn_g, Lx, Ly, max_separation)
    return tn_g
def y_inward_compress_max(tn_, bond_dim = 10, max_distance=4, opt = 'auto-hq', inward_=4, mode="mps", equalize_norms = True, canonize = True):
    tn_ = tn_.copy()
    #print(tn_g)
    Lx = tn_.Lx
    Ly = tn_.Ly

    tn_g = tn_.canonize_around([f"I{i},{Ly-1}" for i in range(Lx)], which='any',
                            max_distance=max_distance,
                            absorb='right',
                            link_absorb='both',
                            equalize_norms=True, 
                            inplace=False)
    
    
    max_separation = Ly - inward_ 
    #print("max_separation_2", max_separation, (0, Ly-inward_-1), (Lx-1, Ly-inward_-1), (0, 0), (Lx-1, 0))

    tn_g = tn_g.contract_boundary(max_bond = bond_dim, 
                                    mode = mode,  #'full-bond', 'mps'
                                    sequence =  ['ymax'], #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},                               
                                    #sequence =  {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                    #compress_opts = {"canonize_distance": canonize_distance},
                                    around=[(0, Ly-inward_-1), (Lx-1, Ly-inward_-1), (0, 0), (Lx-1, 0)],
                                    final_contract_opts = {"optimize": opt}, 
                                    cutoff = 1e-14,
                                    progbar = True,
                                    #max_separation = max_separation,
                                    equalize_norms = equalize_norms,
                                    canonize = canonize,
                                    final_contract = False,
                                    #**{"equalize_norms":True}
                                    )

    tn_g = re_tn_ymax(tn_g, Lx, Ly, max_separation)
    return tn_g


def compress_handy(TN_peps, site, bond_dim, equalize_norms, mode, canonize, opt, inward_x=10 ):
    L_y = TN_peps.Ly
    TN_peps = x_inward_compress_min(TN_peps, opt=opt,  
                                    inward_ = inward_x, max_distance = inward_x, 
                                    bond_dim=bond_dim, 
                                    mode=mode, equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    TN_peps = x_inward_compress_max(TN_peps, opt=opt,  
                                    inward_ = inward_x, max_distance = inward_x, 
                                    bond_dim=bond_dim, 
                                    mode=mode, equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    TN_peps.balance_bonds_()
    TN_peps = y_inward_compress_min(TN_peps, opt=opt, bond_dim = bond_dim, 
                                    inward_ = site  , 
                                    max_distance = site, 
                                    mode="mps", 
                                    equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )


    if site < L_y-1:
        TN_peps = y_inward_compress_max(TN_peps, opt=opt, bond_dim = bond_dim, 
                                        inward_ = TN_peps.Ly - 2, 
                                        max_distance = TN_peps.Ly - 2, 
                                        mode="mps", 
                                        equalize_norms = equalize_norms, 
                                        canonize = canonize
                                    )


    #print(TN_peps.Ly,TN_peps[f"O{site}"])
    TN_peps.balance_bonds_()
    #TN_peps.equalize_norms_()
    TN_peps = x_inward_compress_max(TN_peps, opt=opt,  
                                    inward_=TN_peps.Lx//2 , max_distance=TN_peps.Lx//2, 
                                    bond_dim=bond_dim, 
                                    mode=mode, equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    TN_peps = x_inward_compress_min(TN_peps, opt=opt,  
                                    inward_=TN_peps.Lx - 2, 
                                    max_distance=TN_peps.Lx - 2, 
                                    bond_dim=bond_dim, 
                                    mode=mode, equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    return TN_peps

def compress_x(TN_peps, site,bond_dim, equalize_norms, mode, canonize, opt):

    # TN_peps = TN_peps.contract_boundary(max_bond = bond_dim, 
    #                             mode = mode,  #'full-bond', 'mps'
    #                             sequence =  { 'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},                               
    #                             #sequence =  {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
    #                             #compress_opts = {"canonize_distance": canonize_distance},
    #                             final_contract_opts = {"optimize": opt}, 
    #                             cutoff = 1e-14,
    #                             progbar = True,
    #                             max_separation = 2,
    #                             around = ([TN_peps.Lx//2-2, TN_peps.Ly//2-2], [TN_peps.Lx//2+2, TN_peps.Ly//2+2]),
    #                             equalize_norms = equalize_norms,
    #                             canonize = canonize,
    #                             final_contract = False,
    #                             #**{"equalize_norms":True}
    #                             )

    L_y = TN_peps.Ly
    TN_peps.balance_bonds_()
    TN_peps = x_inward_compress_min(TN_peps, opt=opt, bond_dim = bond_dim, 
                                    inward_ = TN_peps.Lx//2 , 
                                    max_distance = TN_peps.Lx//2, 
                                    mode="mps", 
                                    equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    TN_peps = x_inward_compress_max(TN_peps, opt=opt, bond_dim = bond_dim, 
                                    inward_ = TN_peps.Lx - 2 , 
                                    max_distance = TN_peps.Lx - 2, 
                                    mode="mps", 
                                    equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    
    TN_peps.balance_bonds_()
    #TN_peps.equalize_norms_()
    TN_peps = y_inward_compress_min(TN_peps, opt=opt,  
                                    inward_=site , max_distance=site, 
                                    bond_dim=bond_dim, 
                                    mode=mode, equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    if site < L_y-1:
        TN_peps = y_inward_compress_max(TN_peps, opt=opt,  
                                        inward_=TN_peps.Ly - 2, 
                                        max_distance=TN_peps.Ly - 2, 
                                        bond_dim=bond_dim, 
                                        mode=mode, equalize_norms = equalize_norms, 
                                        canonize = canonize
                                    )
    return TN_peps



def compress_y(TN_peps, site,bond_dim, equalize_norms, mode, canonize, opt):

    # TN_peps = TN_peps.contract_boundary(max_bond = bond_dim, 
    #                             mode = mode,  #'full-bond', 'mps'
    #                             sequence =  { 'ymin', 'ymax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},                               
    #                             #sequence =  {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
    #                             #compress_opts = {"canonize_distance": canonize_distance},
    #                             final_contract_opts = {"optimize": opt}, 
    #                             around = ([TN_peps.Lx//2-2, TN_peps.Ly//2-2], [TN_peps.Lx//2+2, TN_peps.Ly//2+2]),
    #                             cutoff = 1e-14,
    #                             progbar = True,
    #                             max_separation = 3,
    #                             equalize_norms = equalize_norms,
    #                             canonize = canonize,
    #                             final_contract = False,
    #                             #**{"equalize_norms":True}
    #                             )
    L_y = TN_peps.Ly

    TN_peps = y_inward_compress_min(TN_peps, opt=opt,  
                                    inward_=site , max_distance=site, 
                                    bond_dim=bond_dim, 
                                    mode=mode, equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )
    if site < L_y-1:

        TN_peps = y_inward_compress_max(TN_peps, opt=opt,  
                                        inward_=TN_peps.Ly - 2, 
                                        max_distance=TN_peps.Ly - 2, 
                                        bond_dim=bond_dim, 
                                        mode=mode, equalize_norms = equalize_norms, 
                                        canonize = canonize
                                    )

    TN_peps.balance_bonds_()
    TN_peps = x_inward_compress_min(TN_peps, opt=opt, bond_dim = bond_dim, 
                                    inward_ = TN_peps.Lx//2 , 
                                    max_distance = TN_peps.Lx//2, 
                                    mode="mps", 
                                    equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )

    TN_peps = x_inward_compress_max(TN_peps, opt=opt, bond_dim = bond_dim, 
                                    inward_ = TN_peps.Lx - 2 , 
                                    max_distance = TN_peps.Lx - 2, 
                                    mode="mps", 
                                    equalize_norms = equalize_norms, 
                                    canonize = canonize
                                )


    return TN_peps


def compress_xy(TN_peps, site,bond_dim, equalize_norms, mode, canonize, opt):

    TN_peps = TN_peps.contract_boundary(max_bond = bond_dim, 
                                mode = mode,  #'full-bond', 'mps'
                                sequence =  [ 'xmin', 'xmax', 'ymin', 'ymax' ], #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},                               
                                #sequence =  {'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                #compress_opts = {"canonize_distance": canonize_distance},
                                final_contract_opts = {"optimize": opt},
                                around = ([TN_peps.Lx//2-2, site-2], [TN_peps.Lx//2+1, site+1]), 
                                cutoff = 1e-14,
                                progbar = True,
                                max_separation = 2,
                                equalize_norms = equalize_norms,
                                canonize = canonize,
                                final_contract = False,
                                #**{"equalize_norms":True}
                                )


    return TN_peps
def tn_H(U, L):
    U = U * 1.
    U_ = U.conj()
    U_.reindex_({ f"b{i}":f"g{i}"  for i in range(L)})
    U_.reindex_({ f"k{i}":f"b{i}"  for i in range(L)})
    U_.reindex_({ f"g{i}":f"k{i}"  for i in range(L)})
    return U_
def tn_T(U, L):
    U_ = U * 1.
    U_.reindex_({ f"b{i}":f"g{i}"  for i in range(L)})
    U_.reindex_({ f"k{i}":f"b{i}"  for i in range(L)})
    U_.reindex_({ f"g{i}":f"k{i}"  for i in range(L)})
    return U_

def norm_MPO(mpo, opt):
    return (mpo & mpo.H).contract(all, optimize=opt)

def norm_MPO_f(mpo, opt, L):
    I_mpo=qtn.MPO_identity(L, phys_dim=2, dtype=dtype)
    return (mpo & I_mpo.H).contract(all, optimize=opt)


def apply_pepo_peps_flat(x_pepo, peps):
    x_pepo = x_pepo.copy()
    peps = peps.copy()
    x_pepo=x_pepo.conj()
    Lx = peps.Lx
    Ly = peps.Ly
    #peps=peps.conj()
    peps.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    x_pepo.reindex_({ f"b{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn=(x_pepo & peps )
    tn.flatten(fuse_multibonds=True, inplace=True)
    #tn.reindex_({ f"b{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn.view_as_(
            qtn.tensor_2d.PEPS,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            site_ind_id='k{},{}',
    )
    return tn

def apply_pepo_peps(x_pepo, peps):
    x_pepo = x_pepo.copy()
    peps = peps.copy()
    x_pepo=x_pepo.conj()
    Lx = peps.Lx
    Ly = peps.Ly
    peps.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    x_pepo.reindex_({ f"b{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn=(x_pepo & peps )
    tn.view_as_(
            qtn.tensor_2d.PEPS,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            site_ind_id='k{},{}',
    )
    return tn


def apply_mpo_sandwich(x_pepo, pepo):
    L = pepo.L
    pepoH=pepo.H
    pepo.reindex_({ f"k{i}":f"g{i}"  for i in range(L)})
    pepo.reindex_({ f"b{i}":f"m{i}"  for i in range(L)})
    x_pepo.reindex_({ f"k{i}":f"g{i}"  for i in range(L)})
    tn=(x_pepo & pepo )

    x_pepo.reindex_({ f"g{i}":f"k{i}"  for i in range(L)})
    pepo.reindex_({ f"m{i}":f"b{i}"  for i in range(L)})
    pepo.reindex_({ f"g{i}":f"k{i}"  for i in range(L)})

    tn.reindex_({ f"m{i}":f"k{i}"  for i in range(L)})

    tn.reindex_({ f"b{i}":f"p{i}"  for i in range(L)})
    pepoH.reindex_({ f"k{i}":f"p{i}"  for i in range(L)})

    tn=(pepoH & tn )
    return tn


def apply_pepo_sandwich(x_pepo, pepo):

    pepoH=pepo.H
    Lx = pepo.Lx
    Ly = pepo.Ly
    pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"b{i},{j}":f"m{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    x_pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn=(x_pepo & pepo )

    x_pepo.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"m{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})

    tn.reindex_({ f"m{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})

    tn.reindex_({ f"b{i},{j}":f"p{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepoH.reindex_({ f"k{i},{j}":f"p{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})

    tn=(pepoH & tn )
    return tn


def apply_pepo_sandwich_1(x_pepo, pepo):

    pepoH=pepo.H
    Lx = pepo.Lx
    Ly = pepo.Ly
    pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"b{i},{j}":f"m{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    x_pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn=(x_pepo & pepo )

    x_pepo.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"m{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})

    tn.reindex_({ f"m{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})

    return tn

def apply_pepo_sandwich_2(x_pepo, pepo):

    pepoH=pepo.H
    Lx = pepo.Lx
    Ly = pepo.Ly
    x_pepo.reindex_({ f"b{i},{j}":f"p{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepoH.reindex_({ f"k{i},{j}":f"p{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})

    tn=(pepoH & x_pepo )
    return tn









def apply_pepo_sandwich_flat(x_pepo, pepo):
    pepoH=pepo.H
    Lx = pepo.Lx
    Ly = pepo.Ly
    pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"b{i},{j}":f"m{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    x_pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn=(x_pepo & pepo )
    tn.flatten(fuse_multibonds=True, inplace=True)

    x_pepo.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"m{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})


    tn.reindex_({ f"m{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn.view_as_(
            qtn.tensor_2d.PEPO,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            upper_ind_id='k{},{}',
            lower_ind_id='b{},{}',
    )
   

    tn.reindex_({ f"b{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepoH.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})


    tn=(pepoH & tn )
    tn.flatten(fuse_multibonds=True, inplace=True)

    tn.view_as_(
            qtn.tensor_2d.PEPO,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            upper_ind_id='k{},{}',
            lower_ind_id='b{},{}',
    )

    return tn


def apply_pepo(x_pepo, pepo):
    x_pepo = x_pepo.copy()
    pepo = pepo.copy()
    pepo=pepo.H
    Lx = pepo.Lx
    Ly = pepo.Ly
    x_pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    x_pepo.reindex_({ f"b{i},{j}":f"m{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"b{i},{j}":f"m{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn = (x_pepo & pepo )
    tn.flatten(fuse_multibonds=True, inplace=True)
    #print(tn)
    tn.reindex_({ f"k{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    tn.view_as_(
            qtn.tensor_2d.PEPO,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            upper_ind_id='k{},{}',
            lower_ind_id='b{},{}',
    )
    return tn


# def apply_pepo_sandwich(x_pepo, pepo, infidel_, chi, opt, compress=False, bond_dim=4,canonize_distance= 2):
#     x_pepo = x_pepo.copy()
#     pepo = pepo.copy()
#     pepoH=pepo.H
#     Lx = pepo.Lx
#     Ly = pepo.Ly
#     pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
#     pepo.reindex_({ f"b{i},{j}":f"m{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
#     x_pepo.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
#     tn=(x_pepo & pepo )
#     tn.flatten(fuse_multibonds=True, inplace=True)
#     #print(tn)
#     tn.reindex_({ f"m{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
#     tn.view_as_(
#             qtn.tensor_2d.PEPO,
#             Lx=Lx, 
#             Ly=Ly,
#             site_tag_id='I{},{}',
#             x_tag_id='X{}',
#             y_tag_id='Y{}',
#             upper_ind_id='k{},{}',
#             lower_ind_id='b{},{}',
#     )
   
#     tn.reindex_({ f"b{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
#     pepoH.reindex_({ f"k{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})

#     if compress:
#         tn.compress_all(inplace=True, **{"max_bond":bond_dim, "canonize_distance": canonize_distance, "cutoff":1.e-12})
#         # peps_t = peps_view(tn)
#         # peps_t, infidel = peps_bond_reduction(peps_t, bond_dim, chi, opt, Lx, Ly, 
#         #                                         iter = 0,
#         #                                         iterate = 500,
#         #                                         layer_tags = ['KET', 'BRA'],
#         #                                         normalize = False, 
#         #                                         progbar = True, 
#         #                                         cost_f = "fidelity",
#         #                                         sequence = {"xmin", "xmax", "ymin", "ymax"},
#         #                                         rand = None,
#         #                                         bond_dim_rand = 2,
#         #                                         canonize_distance= canonize_distance,
#         #                                     )
#         # infidel_.append(infidel)
#         # tn=pepo_view(peps_t)
#         # tn.balance_bonds_()







#     tn=(pepoH & tn )
#     tn.flatten(fuse_multibonds=True, inplace=True)
#     #tn.reindex_({ f"b{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
#     #tn.reindex_({ f"m{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
#     #print(tn)
#     if compress:
#         tn.compress_all(inplace=True, **{"max_bond":bond_dim, "canonize_distance": canonize_distance, "cutoff":1.e-12})
#         # peps_t = peps_view(tn)
#         # peps_t, infidel = peps_bond_reduction(peps_t, bond_dim, chi, opt, Lx, Ly, 
#         #                                         iter = 0,
#         #                                         iterate = 500,
#         #                                         layer_tags = ['KET', 'BRA'],
#         #                                         normalize = False, 
#         #                                         progbar = True, 
#         #                                         cost_f = "fidelity",
#         #                                         sequence = {"xmin", "xmax", "ymin", "ymax"},
#         #                                         rand = None,
#         #                                         bond_dim_rand = 2,
#         #                                         canonize_distance= canonize_distance,
#         #                                     )
#         # infidel_.append(infidel)
#         # tn=pepo_view(peps_t)
#         # tn.balance_bonds_()



#     tn.view_as_(
#             qtn.tensor_2d.PEPO,
#             Lx=Lx, 
#             Ly=Ly,
#             site_tag_id='I{},{}',
#             x_tag_id='X{}',
#             y_tag_id='Y{}',
#             upper_ind_id='k{},{}',
#             lower_ind_id='b{},{}',
#     )




#    return tn, infidel_




def pepo_trans(pepo):
    #pepo = pepo.copy()
    Lx = pepo.Lx
    Ly = pepo.Ly
    pepo.reindex_({ f"b{i},{j}":f"g{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"k{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    pepo.reindex_({ f"g{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))})
    #return pepo


def mpo_trans(mpo):
    mpo = mpo.copy()
    L = mpo.L
    mpo.reindex_({ f"b{i}":f"g{i}"  for i in itertools.product(range(L))})
    mpo.reindex_({ f"k{i}":f"b{i}"  for i in itertools.product(range(L))})
    mpo.reindex_({ f"g{i}":f"k{i}"  for i in itertools.product(range(L))})
    return mpo




def cal_cir(circ, where, G_, Lx, Ly, Lz, dtype, opt):
    from autoray import do, reshape, backend_like

    def cor_1d(i, j, k):
        return i*Ly*Lz + j*Lz + k
    
    
    rehearse = False

    simplify_atol=1e-12
    simplify_equalize_norms=False
    simplify_sequence='ADCRS'
    fs_opts = {
                'seq': 'ADCRS',
                'atol': simplify_atol,
                'equalize_norms': simplify_equalize_norms,
            }
    
    rho = circ.get_rdm_lightcone_simplified(where=where, **fs_opts)
    k_inds = list(circ.ket_site_ind(i) for i in where)
    b_inds = list(circ.bra_site_ind(i) for i in where)

    
    for count, G in enumerate(G_):
        G_data = reshape(G, (2,2)  )
        output_inds = ()
        TG = qtn.Tensor(data=G_data, inds=[b_inds[count]] + [k_inds[count]])
        rho = rho & TG
    rhoG = rho
    rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
    rhoG.squeeze(fuse=True, inplace=True)
    rhoG.astype_(dtype)


    info = rhoG.contract(
        all,
        output_inds=output_inds,
        optimize=opt,
        get='path-info',
        use_cotengra=True
    )


    z_cal = rhoG.contract(all, output_inds=output_inds, use_cotengra=True,
                            optimize=info.path, 
                        #backend=backend
                        )



    return z_cal 


def l_to_backend(pepo_l, to_backend):
    #to_backend_ = get_to_backend(to_backend)

    pepo_l_ = []
    for pepo in pepo_l:
        pepo_local = []
        for pepo_ in pepo:
            pepo_.apply_to_arrays(to_backend)
            pepo_local.append(pepo_)
        pepo_l_.append(pepo_local)

    return pepo_l_

def gate_to_backend(gate_l, to_backend):
    gate_l_ff = []
    for count, gate_ in enumerate(gate_l):
        gate_l_f = []
        for gate in gate_:
            if len(gate.shape) == 4:
                gate_tn = qtn.Tensor(gate, inds=("a", "b", "c", "d"))
                gate_tn.apply_to_arrays(to_backend)
                gate_l_f.append(gate_tn.data)
            if len(gate.shape) == 2:
                gate_tn = qtn.Tensor(gate, inds=("a", "b"))
                gate_tn.apply_to_arrays(to_backend)
                gate_l_f.append(gate_tn.data)
        gate_l_ff.append(gate_l_f)
    
    return gate_l_ff


def gate_to_backend_(gate_l, to_backend):
    gate_l_ff = []
    for count, gate_ in enumerate(gate_l):
        gate_l_f = []
        for gate in gate_:
            gate = gate.array
            if len(gate.shape) == 4:
                gate_tn = qtn.Tensor(gate, inds=("a", "b", "c", "d"))
                gate_tn.apply_to_arrays(to_backend)
                gate_l_f.append(gate_tn.data)
            if len(gate.shape) == 2:
                gate_tn = qtn.Tensor(gate, inds=("a", "b"))
                gate_tn.apply_to_arrays(to_backend)
                gate_l_f.append(gate_tn.data)
        gate_l_ff.append(gate_l_f)
    
    return gate_l_ff





def cal_cir1d(circ, where, G_, opt):
    from autoray import do, reshape, backend_like

    
    dtype = "complex128"
    rehearse = False

    simplify_atol=1e-12
    simplify_equalize_norms=False
    simplify_sequence='ADCRS'
    fs_opts = {
                'seq': 'ADCRS',
                'atol': simplify_atol,
                'equalize_norms': simplify_equalize_norms,
            }
    
    rho = circ.get_rdm_lightcone_simplified(where=where, **fs_opts)
    k_inds = list(circ.ket_site_ind(i) for i in where)
    b_inds = list(circ.bra_site_ind(i) for i in where)

    
    for count, G in enumerate(G_):
        G_data = reshape(G, (2,2)  )
        output_inds = ()
        TG = qtn.Tensor(data=G_data, inds=[b_inds[count]] + [k_inds[count]])
        rho = rho & TG
    rhoG = rho
    rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
    rhoG.squeeze(fuse=True, inplace=True)
    rhoG.astype_(dtype)


    info = rhoG.contract(
        all,
        output_inds=output_inds,
        optimize=opt,
        get='path-info',
        #use_cotengra=True
    )


    z_cal = rhoG.contract(all, output_inds=output_inds, 
                          #use_cotengra=True,
                          optimize=info.path, 
                        #backend=backend
                        )



    return z_cal 






def cal_cir_approx(circ, site, O_label, Lx, Ly, Lz, dtype, opt, chi):
    from autoray import do, reshape, backend_like

    def cor_1d(i, j, k):
        return i*Ly*Lz + j*Lz + k
    x, y = site
    G = qu.pauli(O_label)
    rehearse = False

    simplify_atol=1e-12
    simplify_equalize_norms=False
    simplify_sequence='ADCRS'
    fs_opts = {
                'seq': 'ADCRS',
                'atol': simplify_atol,
                'equalize_norms': simplify_equalize_norms,
            }
    where=[cor_1d(x,y,0)]
    rho = circ.get_rdm_lightcone_simplified(where=where, **fs_opts)
    k_inds = tuple(circ.ket_site_ind(i) for i in where)
    b_inds = tuple(circ.bra_site_ind(i) for i in where)

    if isinstance(G, (list, tuple)):
        # if we have multiple expectations create an extra indexed stack
        nG = len(G)
        G_data = do('stack', G)
        G_data = reshape(G_data, (nG,) + (2,) * 2 * len(where))
        output_inds = (rand_uuid(),)
    else:
        G_data = reshape(G, (2,) * 2 * len(where))
        output_inds = ()

    TG = qtn.Tensor(data=G_data, inds=output_inds + b_inds + k_inds)

    rhoG = rho | TG

    rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
    rhoG.squeeze(fuse=True, inplace=True)
    rhoG.astype_(dtype)

    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=256,
        minimize='combo-compressed', 
        progbar=True,
        # # save paths to disk:
        # directory=True  
        directory="cash/",
        parallel=True,

    )

    tree = rhoG.contraction_tree(copt)
    
    
    rhoG.contract_compressed_(
        optimize=tree.get_path(),  # or optimize=copt
        max_bond=chi,
        equalize_norms=1.0,
        progbar=True,
    )


    mantissa, exponent = (rhoG.contract(), rhoG.exponent)
    print(mantissa, exponent)

    return mantissa * 10**(exponent) 









def pepo_gate(pepo_, U, where, cutoff = 1.e-12, dtype = "complex128"):
    pepo_ = pepo_ * 1
    if len(where) == 2:
        x, y = where
        x0, x1 = x
        y0, y1 = y
        t_a = pepo_[f"I{x0},{x1}"] * 1.
        t_b = pepo_[f"I{y0},{y1}"] * 1.
        
        inds_a = list(t_a.inds)
        inds_b = list(t_b.inds)
        
        index = list(set(inds_a).intersection(inds_b))
        if len(index) > 1:
            print("warning > len(index) > 2")
        
        index_joint = index[0]
        
        T = qtn.Tensor(data=U, inds=(f"l{x0},{x1}",f"l{y0},{y1}",f"k{x0},{x1}",f"k{y0},{y1}" ), tags=[])
        T_l, T_r = qtn.tensor_split(T, [f"l{x0},{x1}",f"k{x0},{x1}"], get = "tensors", cutoff=cutoff, bond_ind="x")
    
        inds_ = list(t_a.inds)
        if inds_[-1] == f"k{x0},{x1}":  
            inds_[-1] = f"l{x0},{x1}"
        else:
            inds_[-2] = f"l{x0},{x1}"
            
        
        t_a = t_a & T_l
        t_a = t_a.contract(all, inds_ + ["x"])
        new_index = qtn.rand_uuid()
        t_a.fuse({new_index:(index_joint, "x") }, inplace=True)
        #t_a.transpose(*inds_, inplace=True)
        t_a.reindex_({f"l{x0},{x1}":f"k{x0},{x1}"})
        pepo_[f"I{x0},{x1}"] = t_a

        
        inds_ = list(t_b.inds)
        if inds_[-1] == f"k{y0},{y1}":  
            inds_[-1] = f"l{y0},{y1}"
        else:
            inds_[-2] = f"l{y0},{y1}"

        
        
        t_b = t_b & T_r
        t_b = t_b.contract(all, inds_ + ["x"])
        t_b.fuse({new_index:(index_joint, "x") }, inplace=True)
        #t_b.transpose(*inds_, inplace=True)
        t_b.reindex_({f"l{y0},{y1}":f"k{y0},{y1}"})
        

        pepo_[f"I{y0},{y1}"] = t_b

    if len(where) == 1:
        x,  = where
        x0, x1 = x
        t_a = pepo_[f"I{x0},{x1}"] * 1.
        
        T = qtn.Tensor(data=U, inds=(f"l{x0},{x1}",f"k{x0},{x1}" ), tags=[])
            
        inds_ = list(t_a.inds)
        if inds_[-1] == f"k{x0},{x1}":  
            inds_[-1] = f"l{x0},{x1}"
        else:
            inds_[-2] = f"l{x0},{x1}"

        t_a = t_a & T
        t_a = t_a.contract(all, inds_)
        t_a.reindex_({f"l{x0},{x1}":f"k{x0},{x1}"})
        pepo_[f"I{x0},{x1}"] = t_a



    
    return pepo_


def mpo_gate(pepo_, U, where, cutoff = 1.e-12, dtype = "complex128"):
    pepo_ = pepo_ * 1
    if len(where) == 2:
        x, y = where
        t_a = pepo_[f"I{x}"] * 1.
        t_b = pepo_[f"I{y}"] * 1.
        
        inds_a = list(t_a.inds)
        inds_b = list(t_b.inds)
        
        index = list(set(inds_a).intersection(inds_b))
        if len(index) > 1:
            print("warning > len(index) > 2")
        
        if abs(x-y) != 1:
            print("abs(x-y) != 1")

        index_joint = index[0]
        
        T = qtn.Tensor(data=U, inds=(f"l{x}",f"l{y}",f"k{x}",f"k{y}" ), tags=[])
        T_l, T_r = qtn.tensor_split(T, [f"l{x}",f"k{x}"], get = "tensors", cutoff=cutoff, bond_ind="x")
    

        inds_ = list(t_a.inds)
        if inds_[-1] == f"k{x}":  
            inds_[-1] = f"l{x}"
        else:
            inds_[-2] = f"l{x}"
            
        
        t_a = t_a & T_l
        t_a = t_a.contract(all, inds_ + ["x"])
        new_index = qtn.rand_uuid()
        t_a.fuse({new_index:(index_joint, "x") }, inplace=True)
        t_a.reindex_({f"l{x}":f"k{x}"})
        pepo_[f"I{x}"] = t_a

        
        inds_ = list(t_b.inds)
        if inds_[-1] == f"k{y}":  
            inds_[-1] = f"l{y}"
        else:
            inds_[-2] = f"l{y}"

        
        
        t_b = t_b & T_r
        t_b = t_b.contract(all, inds_ + ["x"])
        t_b.fuse({new_index:(index_joint, "x") }, inplace=True)
        t_b.reindex_({f"l{y}":f"k{y}"})
        

        pepo_[f"I{y}"] = t_b

    if len(where) == 1:
        x,  = where
        t_a = pepo_[f"I{x}"] * 1.
        
        T = qtn.Tensor(data=U, inds=(f"l{x}",f"k{x}" ), tags=[])
            
        inds_ = list(t_a.inds)
        if inds_[-1] == f"k{x}":  
            inds_[-1] = f"l{x}"
        else:
            inds_[-2] = f"l{x}"

        t_a = t_a & T
        t_a = t_a.contract(all, inds_)
        t_a.reindex_({f"l{x}":f"k{x}"})
        pepo_[f"I{x}"] = t_a



    
    return pepo_


def peps_view(pepo):
    Lx=pepo.Lx
    Ly=pepo.Ly
    x_pepo = pepo * 1.
    for i,j in itertools.product(range(Lx), range(Ly)):
        x_pepo[f"I{i},{j}"].fuse({f"k{i},{j}":(f"b{i},{j}", f"k{i},{j}")  }, inplace=True)    
    
    x_pepo.view_as_(
                qtn.tensor_2d.PEPS,
                Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
                site_ind_id='k{},{}',
        )
    return x_pepo
def pepo_view(peps):
    x_peps = peps * 1.
    Lx=peps.Lx
    Ly=peps.Ly

    for i,j in itertools.product(range(Lx), range(Ly)):
        x_peps[f"I{i},{j}"].unfuse({f"k{i},{j}":(f"b{i},{j}", f"k{i},{j}")  }, shape_map ={f"k{i},{j}":(2,2)}, inplace=True)    
    
    x_peps.view_as_(
            qtn.tensor_2d.PEPO,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            upper_ind_id='k{},{}',
            lower_ind_id='b{},{}',
    )
    return x_peps

#tn.fuse_multibonds_()
def apply_peps(peps, gate_l, where_l, bond_dim, 
               canonize_distance=2, 
               cutoff=1.e-9,
               eff_ = False,
               contract = 'reduce-split',
               ):
    inf = {}
    tags_compress = []
    
    for count in range(len(gate_l)):
        info = {}
        G = gate_l[count]
        where = where_l[count]
        peps=peps.gate(G, where, 
                       contract='reduce-split', tags=["G"], 
                        propagate_tags='sites', 
                        inplace=True, info=info, 
                        long_range_use_swaps=True, 
                        long_range_path_sequence=('av', 'bh'), 
                        **{"cutoff":1.e-9, "absorb":'both'})

        if len(where) == 2:
            ij_a, ij_b = where
            *path, final = gen_long_range_swap_path(
                    ij_a, ij_b, sequence=('av', 'bh')
                )
            path += [final]
            if path:
                for cor in path:
                    x, y = cor
                    x0, x1 = x
                    y0, y1 = y
                    tags_compress.append( [f"I{x0},{x1}", f"I{y0},{y1}"])
 
    
    return peps, tags_compress





# peps, infidel = peps_bond_reduction( peps, bond_dim, chi, opt, Lx, Ly, env_cor,
#                                     iter = 0,
#                                     opt_distance =  1,    #max(Lx, Ly), distance to optimize tensors
#                                     iterate = 400,
#                                     layer_tags = ['KET', 'BRA'],
#                                     normalize = True, 
#                                     progbar = True,
#                                     max_seperation = 2, 
#                                     cost_f = "fidelity",
#                                     sequence = {"xmin", "xmax"}, #{"xmin", "xmax", "ymin", "ymax"},
#                                     rand = 0.0,
#                                     bond_dim_rand = 2,
#                                     canonize_distance= canonize_distance,
#                                 )
def bp_mps(psi, O_label, site, opt, to_backend, 
           max_iterations=256,
           progbar=True, tol_final=1.e-6,
            damping=0.01 ):
    psiG = psi.copy()
    for count, site_ in enumerate(site):
        psiG.gate_(to_backend(O_label[count]), site_, contract=True)    
    expec = psi.H & psiG
    obs = complex(
            contract_l1bp(
            expec,
            tol=tol_final,
            max_iterations=max_iterations,
            optimize=opt,
            progbar=progbar,
            damping=damping
        )
    )
    return obs



def refix_inds(tn_outer, k_inds):
    dic = {}
    psi_ = tn_outer.copy()

    pattern = r'I\d+'
    strings = psi_.tags
    site_tags = [s for s in strings if re.search(pattern, s)]
    site_tags = [ i  for i in site_tags if i.startswith("I") ]

    for tag_ in site_tags:
        tid = psi_.tag_map[tag_]
        tn = psi_.tensor_map[int(*tid)]
        tn.modify(tags=tag_)
        integer_ = re.findall(r'\d+', tag_)
        integer_ = integer_[0]
        for i in tn.inds: 
            if i in k_inds:
                dic |= {i:f"k{integer_}"}
    
    psi_.reindex_(dic)
    return psi_

def refix_inds_(tn_outer, k_inds, b_inds, inplace = False):
    dic = {}
    psi_ = tn_outer if inplace else tn_outer.copy()
    
    pattern = r'I\d+'
    strings = psi_.tags
    site_tags = [s for s in strings if re.search(pattern, s)]
    
    for tag_ in site_tags:
        tid = psi_.tag_map[tag_]
        tn = psi_.tensor_map[int(*tid)]
        tn.modify(tags=tag_)
        integer_ = re.findall(r'\d+', tag_)
        integer_ = integer_[0]
        for i in tn.inds: 
            if i in k_inds:
                dic |= {i:f"k{integer_}"}
            if i in b_inds:
                dic |= {i:f"b{integer_}"}
            
        
    psi_.reindex_(dic)
    return psi_


def evolve_circ_peps(circ, bond_dim,  
                        prgbar = True, 
                        cutoff = 1.0e-12,
                        opt = None,
                        prog_compress=True,
                        max_iterations = 256,
                        tol = 1.e-8,
                        damping=0.01,
                        steps = 20,
                        O_label =None, 
                        site = None,
                        copt=None,
                        backend = None,
                        last_step = 0,
                        site_2d = None,
                        label = None,
                        chi=None,
                        compress_= None,
                        chi_sample=None,
                        chi_bmps=None,
                        normalize=None,
                        sample_=None,
                        num_workers=None,
                        samples_per_worker=None,
                        for_each_repeat=None,
                        method_s=None,
                        store_state=None,
                        method=None,




                    ):

    psi = circ.psi
    N_l = []
    O_l = []

    to_backend = get_to_backend(backend) #"numpy-single"

    psi.apply_to_arrays(to_backend)
    Zcheck = complex(0)
    with tqdm(total=steps+ last_step,  desc="circ_peps:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:

        for r in range(steps + last_step):
            psi.retag_({f"ROUND_{r}": "OUTER"})

            psi, tn_outer = psi.partition("OUTER", inplace=True)

            k_inds = list(oset(psi.outer_inds()))


            if prog_compress:
                print("-------------l2bp-compress-------------")
            compress_l2bp(
                tn_outer,
                #site_tags=site_tags,
                max_iterations=max_iterations ,
                max_bond=bond_dim,
                cutoff=cutoff,
                cutoff_mode='rsum2',
                tol=tol,
                optimize=opt,
                progbar=prog_compress,
                inplace=True,
                damping=damping,
            )
            
            tn_outer.balance_bonds_()
            #print(complex((tn_outer.H & tn_outer).contract(all, optimize=opt)))
            
            psi_ = refix_inds(tn_outer, k_inds)
            #qu.save_to_disk(psi_, f"Store/psi2d_bnd{bond_dim}-depth{r}")

            
            # res = apply_hyperoptimized_compressed(expec, copt, chi_, output_inds=None, 
            #                               tree_gauge_distance=tree_gauge_distance, 
            #                               progbar=progbar, 
            #                               cutoff=1.e-12
            #                               )
            # main, exp=(res.contract(), res.exponent)
            # print("<X>", complex(main * 10**(exp)) )

            if prog_compress:
                print("-------------l1bp-<X>-------------")

            # obs = bp_mps(psi_, O_label, site, opt,to_backend, 
            #         max_iterations=max_iterations,
            #         progbar=prog_compress, 
            #         tol_final=tol, damping=damping )
        



    #---------------------------Reza------------------------
            if prog_compress:
                print("-------------l2bp-<N>-------------")
                #print("memmory", xyz.report_memory())
                #print("memmory-GPU", xyz.report_memory_gpu())


            bp = L2BP(psi_, optimize=opt,  damping=damping)
            bp.run(max_iterations=max_iterations, tol=tol, progbar=prog_compress)
            est_norm = complex(bp.contract())
            # print("est_norm", est_norm)
            if len(site) == 1:
                rho = bp.partial_trace(site[0], optimize=opt)
                Zcheck = complex(autoray.do("trace", rho @ to_backend(O_label[0])))


            N_l.append(abs(complex(est_norm)))
            qu.save_to_disk(N_l, f"Store/info_circpeps/peps_norm_bnd{bond_dim}-depth{steps}")
            
            #X_ = abs(complex(obs)) / abs(complex(est_norm))
            X_ = (complex(Zcheck)).real
            
            
            O_l.append( X_ )
            qu.save_to_disk(O_l, f"Store/info_circpeps/peps_O_bnd{bond_dim}-depth{steps}")



            pbar.set_postfix({"depth": steps, 
                              "norm": abs(complex(est_norm)),
                              "normim": complex(est_norm).imag, 
                              "bnd": tn_outer.max_bond(),
                              "<X>": X_,
                              #"Check": Zcheck.real,
                              #"memmory":xyz.report_memory(),
                              })
            pbar.refresh()
            pbar.update(1)

    #---------------------------------------------------------------------

            psi.add_tensor_network(tn_outer, virtual=True)
            psi.check()

    return psi, N_l, O_l




def mps_to_pepo(psi_, dic_, inds_upper, inds_lower, tag_I, Lx, Ly):
    pepo = psi_.copy()
    for i in range(psi_.L):
        x, y = dic_[i]
        pepo[f"I{i}"].add_tag(f"X{x}")
        pepo[f"I{i}"].add_tag(f"Y{y}")
    pepo.retag_(tag_I)
    pepo.reindex_(inds_upper)
    pepo.reindex_(inds_lower)
    pepo.view_as_(
            qtn.tensor_2d.PEPO,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            upper_ind_id='k{},{}',
            lower_ind_id='b{},{}',
    )

    return pepo

def evolve_circ_pepo( circ, bond_dim, backend, 
                    prgbar = True, 
                    cutoff = 1.0e-12,
                    opt = None,
                    prog_compress=True,
                    max_iterations = 256,
                    tol = 1.e-8,
                    damping=0.01,
                    steps = 20,
                    O_label =None, 
                    site = None,
                    list_basis=None,
                    last_step = -1,
                    peps = None,
                    method=None,
                    chi = None, 
                    tree_gauge_distance = 4,
                    max_separation = 1,
                    copt = None,
                    mode = "mps",
                ):
    psi = circ.psi
    N_l = []
    O_l = []
    x_l = []
    to_backend = get_to_backend(backend)  # "numpy-single"
    


    if peps:
        Lx = peps.Lx
        Ly = peps.Ly
        dic_ = {}
        for i,j in  itertools.product(range(Lx), range(Ly)):
            dic_ |= {i*Ly + j:(i,j)} 
        inds_upper = { f"k{i*Ly + j}":f"k{i},{j}" for i,j in  itertools.product(range(Lx), range(Ly))         }
        inds_lower = { f"b{i*Ly + j}":f"b{i},{j}" for i,j in  itertools.product(range(Lx), range(Ly))         }
        tag_I = { f"I{i*Ly + j}":f"I{i},{j}" for i,j in  itertools.product(range(Lx), range(Ly))         }
        peps.apply_to_arrays(to_backend)

    if last_step == -1:
        # match message sizes in BP2 and BP1 stages
        last_step = max(1, int(log2(bond_dim) / 2))
        print("last_step", last_step)

    psi_0 = qtn.MPS_computational_state(list_basis)
    
    ket = circ.psi
    bra = ket.conj()


    for count, site_ in enumerate(site):
        ket.gate_inds_(O_label[count], [f"k{site_}"], contract=True)

    ket.add_tag("KET")
    bra.add_tag("BRA")

    tn = ket & bra

    tn.apply_to_arrays(to_backend)
    psi_0.apply_to_arrays(to_backend)
 

    with tqdm(total=steps-1,  desc="circ_pepo:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:

        for r in reversed(range(last_step, steps)):
            
            tn.retag_({f"ROUND_{r}": "INNER"})
            tn, tn_inner = tn.partition("INNER", inplace=True)
            k_inds = list(tn.select("KET", which="any").outer_inds())
            b_inds = list(tn.select("BRA", which="any").outer_inds())
            
            tn_inner.equalize_norms_()

            if prog_compress:
                start_time = time.time()
                print(f"--------l2bp-compress:bnd{tn_inner.max_bond()}---------")
            
            #tn_inner.rank_simplify_(inplace = True, max_combinations=500)
            compress_l2bp(
                            tn_inner,
                            max_bond=bond_dim,
                            cutoff=cutoff,
                            cutoff_mode='rsum2',
                            tol=tol,
                            optimize=opt,
                            progbar=prog_compress,
                            inplace=True,
                            damping=damping,
                        )
            if prog_compress:
                print(f"--- %s seconds l2bp-compress ---" % (time.time() - start_time))
            
            psi_ = refix_inds_(tn_inner, k_inds, b_inds)
            x_pepo = mps_to_pepo(psi_, dic_, inds_upper, inds_lower, tag_I, Lx, Ly)
            

            if method == "hypercompress":
                if prog_compress:
                    start_time = time.time()
                    print(f"-------<O>: hypercompress_chi:{chi}-------------")

                z_appro = pepo_flat_hypercompress(x_pepo, peps, 
                                                chi, 
                                                copt,
                                                progbar=prog_compress, 
                                                tree_gauge_distance=tree_gauge_distance,
                                                opt = opt,
                                                max_bond = None,
                                                canonize_distance = 4,
                                                )
                x_l.append(abs(complex(z_appro)))
                qu.save_to_disk(x_l, f"Store/circ_pepo_O_bnd{bond_dim}-depth{steps}")
                if prog_compress:
                    print("--- %s seconds hypercompress ---" % (time.time() - start_time))

            if method == "boundray":
                if prog_compress:
                    start_time = time.time()
                    print(f"-------------<O>: boundray_chi:{chi}-------------")

                z_appro = pepo_cal(peps, x_pepo, 
                                    chi, opt,  
                                    max_separation=max_separation, 
                                    mode = mode,       #'mps',"full-bond" 
                                    progbar = prog_compress, 
                                )
                x_l.append(abs(complex(z_appro)))
                qu.save_to_disk(x_l, f"Store/circ_pepo_O_bnd{bond_dim}-depth{steps}")
                if prog_compress:
                    print("--- %s seconds boundray ---" % (time.time() - start_time))



            if prog_compress:
                start_time = time.time()
                print(f"-------------<O>: L1BP-------------")

            z_appro_bp = pepo_BP(x_pepo, peps, 
                                    progbar=prog_compress, 
                                    opt = opt,
                                    max_bond = None,
                                )
            if prog_compress:
                print("--- %s seconds <O>: L1BP ---" % (time.time() - start_time))

            if prog_compress:
                start_time = time.time()
                print(f"-------------l2bp-<N>-------------")

            bp = L2BP(psi_, optimize=opt)
            bp.run(tol=tol, max_iterations=max_iterations, progbar=prog_compress)
            _, norm_exponent = bp.contract(strip_exponent=True)
            est_norm = float(10 ** ((norm_exponent - (len(bp.local_tns) * log10(2))) / 2))
            
            
            O_l.append(abs(complex(z_appro_bp)))
            N_l.append(abs(complex(est_norm)))
            qu.save_to_disk(N_l, f"Store/circ_pepo_norm_bnd{bond_dim}-depth{steps}")
            qu.save_to_disk(O_l, f"Store/circ_pepo_Obp_bnd{bond_dim}-depth{steps}")

            if prog_compress:
                print("--- %s seconds l2bp-<N> ---" % (time.time() - start_time))

                print("-----------------------------------")
                print(xyz.report_memory())
                print("-----------------------------------")

            pbar.set_postfix({"depth": steps, 
                            "norm": abs(complex(est_norm)), "bnd":tn_inner.max_bond(),
                            "<X_bp>":abs(complex(z_appro_bp)),
                            "<X>":abs(complex(z_appro)),

                            #"memmory":xyz.report_memory(),
                            })
            pbar.refresh()
            pbar.update(1)

    #---------------------------------------------------------------------

            tn.add_tensor_network(tn_inner, check_collisions=False, virtual=True)
            tn.check()
            


    return psi, N_l, O_l, x_l






def pepo_norm(peps, opt, max_bond=10, canonize = True, 
            layer_tags=['KET', 'BRA'], 
            equalize_norms=False, 
            max_separation=1, 
            canonize_distance=0,
            sequence = None,
            progbar=False,
            ):
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    
    norm = pepsH & peps
    peps_norm = norm.contract_boundary(max_bond=max_bond, 
                                   mode='mps',  #'full-bond', 'mps'
                                   sequence = sequence, #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                   #compress_opts = {"canonize_distance":canonize_distance},
                                   canonize=canonize,
                                   final_contract_opts={"optimize": opt}, 
                                   cutoff=1e-14,
                                   progbar=progbar,
                                   layer_tags=layer_tags,
                                   max_separation=max_separation,
                                   equalize_norms = equalize_norms,
                                   #**{"equalize_norms":True}
                                  )
    return abs(peps_norm)
def pepo_init(Lx, Ly, site_, Z_, to_backend, dtype="complex128", cycle=False):

    if cycle:
        pepo= pepo_identity(Lx, Ly, dtype=dtype)
        pepo = peps_cycle(pepo, int(1))
    else:
        pepo= pepo_identity(Lx, Ly, dtype=dtype)
        
    for count, site in enumerate(site_):
        Z = Z_[count]
        t = pepo[site] 
        shape = t.shape
        t.modify(data = Z.reshape(shape))
        t.add_tag(f"O_")
        pepo.add_tag("pepo_tag", where=None, which='all')
    pepo.apply_to_arrays(to_backend)
    return  pepo

def peps_cycle(peps, bond_dim, cylinder=False):
    Ly = peps.Ly
    Lx = peps.Lx
    for j in range(Ly):
        T1 = peps[f"I{Lx-1},{j}"]
        T2 = peps[f"I{0},{j}"]
        qtn.new_bond(T1, T2, size=bond_dim, name=None, axis1=0, axis2=0)

    if not cylinder:
        for i in range(Lx):
            T1 = peps[f"I{i},{Ly-1}"]
            T2 = peps[f"I{i},{0}"]
            qtn.new_bond(T1, T2, size=bond_dim, name=None, axis1=0, axis2=0)
    return peps

def apply_2dtn(peps, G, where, bond_dim=None, contract='split', tags=["G"], 
             dtype="complex128", cutoff = 1.0e-12,
             canonize_distance = 2,
             cycle = False, to_backend=None,
             sequence=('av', 'bh', "ah", "bv"),
             ):
    swap = qu.swap(dim=2, dtype = dtype)
    swap = swap.reshape(2,2,2,2)
    dp = peps.phys_dim(*where[0])
    if to_backend:
        swap = to_backend(swap)
    else:
        swap = get_swap(
                    dp, dtype=autoray.get_dtype_name(G), backend=autoray.infer_backend(G)
                )
        
    if len(where) == 1:
        x, = where
        i, j = x
        G_ = autoray.do("transpose", G, (1,0)) 
        qtn.tensor_network_gate_inds(peps, G_, [f"k{i},{j}"], contract=True, 
                                    tags=tags, info=None, inplace=True,
                                    **{"cutoff":cutoff}
                                    )
        G_ = autoray.do("transpose", G, (1,0)) 
        G_ = autoray.do("conj", G_)
        qtn.tensor_network_gate_inds(peps, G_, [f"b{i},{j}"], contract=True, 
                                tags=tags, info=None, inplace=True,
                                **{"cutoff":cutoff}
                                )    
    # two-qubit gate:
    else: 
        x, y = where
        i, j = x
        m, n = y
            #tags_compress.append( [f"I{i},{j}", f"I{m},{n}"])


        *swaps, final = gen_long_range_swap_path(
                x, y, sequence=sequence,
            )

#################SWAP###################################################
        for pair in swaps:
                x_, y_ = pair
                i_, j_ = x_
                m_, n_ = y_
                qtn.tensor_network_gate_inds(peps, 
                                            swap, 
                                            [f"k{i_},{j_}", f"k{m_},{n_}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff, "max_bond": bond_dim}
                                            )
        
        

                qtn.tensor_network_gate_inds(peps, 
                                            swap.conj(), 
                                            [f"b{i_},{j_}", f"b{m_},{n_}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff}
                                            )
            
#######################G##########################        
        x_, y_ = final
        i_, j_ = x_
        m_, n_ = y_
        G_ = autoray.do("transpose", G, (2,3, 0,1)) 
        qtn.tensor_network_gate_inds(peps, 
                                        G_, 
                                        [f"k{i_},{j_}", f"k{m_},{n_}"], 
                                        contract=contract, 
                                        tags=tags, info=None, 
                                        inplace=True,**{"cutoff":cutoff}
                                    )

        
    
        G_ = autoray.do("transpose", G, (2,3, 0,1)) 
        G_ = autoray.do("conj", G_ )
        qtn.tensor_network_gate_inds(peps, 
                                        G_, 
                                        [f"b{i_},{j_}", f"b{m_},{n_}"], 
                                        contract=contract, 
                                        tags=tags, info=None, 
                                        inplace=True,**{"cutoff":cutoff}
                                    )

        
#################SWAP_back###################################################
        
        for pair in reversed(swaps):
                x_, y_ = pair
                i_, j_ = x_
                m_, n_ = y_
                qtn.tensor_network_gate_inds(peps, 
                                            swap, 
                                            [f"k{i_},{j_}", f"k{m_},{n_}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,**{"cutoff":cutoff}
                                            )
                
                qtn.tensor_network_gate_inds(peps, 
                                                swap.conj(), 
                                                [f"b{i_},{j_}", f"b{m_},{n_}"], 
                                                contract=contract, 
                                                tags=tags, info=None, 
                                                inplace=True,**{"cutoff":cutoff}
                                                )
      
    return peps



def apply_2dtn_(peps, G, where, bond_dim=None, 
             bra = False, contract='split', tags=["G"], 
             dtype="complex128", cutoff = 1.0e-12,
             canonize_distance = 2,
             cycle = False, to_backend=None,
             sequence=('av', 'bh', "ah", "bv"),
             ):
    swap = qu.swap(dim=2, dtype = dtype)
    swap = swap.reshape(2,2,2,2)
    dp = peps.phys_dim(*where[0])
    if to_backend:
        swap = to_backend(swap)
    else:
        swap = get_swap(
                    dp, dtype=autoray.get_dtype_name(G), backend=autoray.infer_backend(G)
                )
        
    if len(where) == 1:
        x, = where
        i, j = x 
        qtn.tensor_network_gate_inds(peps, G, [f"k{i},{j}"], contract=True, 
                                    tags=tags, info=None, inplace=True,
                                    **{"cutoff":cutoff}
                                    )
    # two-qubit gate:
    else: 
        x, y = where
        i, j = x
        m, n = y
            #tags_compress.append( [f"I{i},{j}", f"I{m},{n}"])


        *swaps, final = gen_long_range_swap_path(
                x, y, sequence=sequence,
            )

#################SWAP###################################################
        for pair in swaps:
                x_, y_ = pair
                i_, j_ = x_
                m_, n_ = y_
                qtn.tensor_network_gate_inds(peps, 
                                            swap, 
                                            [f"k{i_},{j_}", f"k{m_},{n_}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff, "max_bond": bond_dim}
                                            )
        
        
            
#######################G##########################        
        x_, y_ = final
        i_, j_ = x_
        m_, n_ = y_
        qtn.tensor_network_gate_inds(peps, 
                                        G, 
                                        [f"k{i_},{j_}", f"k{m_},{n_}"], 
                                        contract=contract, 
                                        tags=tags, info=None, 
                                        inplace=True,**{"cutoff":cutoff}
                                    )

        
#################SWAP_back###################################################
        
        for pair in reversed(swaps):
                x_, y_ = pair
                i_, j_ = x_
                m_, n_ = y_
                qtn.tensor_network_gate_inds(peps, 
                                            swap, 
                                            [f"k{i_},{j_}", f"k{m_},{n_}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,**{"cutoff":cutoff}
                                            )


    return peps
















def peps_x(peps, where_l, pauli_l, chi, opt, backend=None, mode="mps"):
    to_backend = get_to_backend(backend) #"numpy-single"

    for count, p_ in enumerate(pauli_l):
        x, y = where_l[count]
        pepsx = qtn.tensor_network_gate_inds(peps, to_backend(p_), [f"k{x},{y}"], contract=True,  inplace=False)
    pepsx.add_tag('KET')
    peps.add_tag('KET')

    pepsH = peps.conj().retag({'KET': 'BRA'})
    normx = pepsH & pepsx
    norm = pepsH & peps

    X = normx.contract_boundary( max_bond=chi, 
                                    mode=mode,  #'full-bond', 'mps'
                                    sequence = ['xmin', 'xmax', "ymin", "ymax" ], #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                    #compress_opts = {"canonize_distance":2},
                                    #around = [site,],
                                    canonize = True,
                                    final_contract_opts = {"optimize": opt}, 
                                    progbar = True,
                                    layer_tags = ["KET", "BRA"],
                                    max_separation = 1,
                                    equalize_norms = False,
                                    final_contract = True,
                                    #**{"equalize_norms":True}
                                    )
    norm = norm.contract_boundary( max_bond=chi, 
                                    mode=mode,  #'full-bond', 'mps'
                                    sequence = ['xmin', 'xmax', "ymin", "ymax" ], #{'xmin', 'xmax' }, #{'xmin', 'xmax'}, #{'ymin', 'ymax'},#{'xmin', 'xmax', 'ymin', 'ymax'},
                                    #compress_opts = {"canonize_distance":2},
                                    #around = [site,],
                                    canonize = True,
                                    final_contract_opts = {"optimize": opt}, 
                                    progbar = True,
                                    layer_tags = ["KET", "BRA"],
                                    max_separation = 1,
                                    equalize_norms = False,
                                    final_contract = True,
                                    #**{"equalize_norms":True}
                                    )

    return abs(complex(X/norm)), abs(complex(norm))








def peps_compress_reza(peps_, opt, chi, max_iterations=250, tol = 1.e-6, progbar=False, damping=0.02):
    Lx = peps_.Lx
    Ly = peps_.Ly
    bp = L2BP(peps_, optimize=opt,  damping=damping)
    bp.normalize_messages()
    bp.run(max_iterations=max_iterations, tol=tol, progbar=progbar)
    neighbors = bp.neighbors
    messages = bp.messages
    messages_squared = bp.messages_squared()
    messages_squared_inv = bp.messages_squared_inv()
    est_norm = complex(bp.contract())
    #print("0", est_norm)
    
    tn_l = []
    for i in range(Lx):
        psi_bp, mess_inv = state_ket(peps_, [(i, j) for j in range(Ly)], neighbors, messages_squared, messages_squared_inv,
                                                       optimize=opt)    
        psi_bp.rank_simplify_()
        psi_bp.compress_all_(max_bond=chi, canonize_distance=Ly, canonize=True)
        
        dic = { i: i.removesuffix("_l2bp*")   for i in psi_bp.outer_inds()} 
        psi_bp.reindex_(dic)
    
        for t in mess_inv:
            psi_bp &= t
        psi_bp.rank_simplify_()
    
        tn_l.append(psi_bp)
    peps_compressed = qtn.TensorNetwork(tn_l)
    peps_compressed.view_as_(
                qtn.tensor_2d.PEPS,
                Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
                site_ind_id='k{},{}',
        )
    peps_compressed.balance_bonds_()
    bp = L2BP(peps_compressed, optimize=opt,  damping=damping)
    bp.normalize_messages()
    bp.run(max_iterations=max_iterations, tol=tol, progbar=progbar)
    neighbors = bp.neighbors
    messages = bp.messages
    messages_squared = bp.messages_squared()
    messages_squared_inv = bp.messages_squared_inv()
    est_norm = complex(bp.contract())
    #print("x", est_norm)
    tn_l = []
    for j in range(Ly):
        psi_bp, mess_inv = state_ket(peps_compressed, [(i, j) for i in range(Lx)], neighbors, messages_squared, messages_squared_inv, optimize=opt)    
        psi_bp.rank_simplify_()
        psi_bp.compress_all_(max_bond=chi, canonize_distance=Lx, canonize=True)
        
        dic = { i: i.removesuffix("_l2bp*")      for i in psi_bp.outer_inds()} 
        psi_bp.reindex_(dic)
    
        for t in mess_inv:
            psi_bp &= t
        psi_bp.rank_simplify_()
    
        tn_l.append(psi_bp)
    
    peps_compressed = qtn.TensorNetwork(tn_l)
    peps_compressed.view_as_(
                qtn.tensor_2d.PEPS,
                Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
                site_ind_id='k{},{}',
        )
    peps_compressed.balance_bonds_()
    # bp.normalize_messages()
    # bp = L2BP(peps_compressed, optimize=opt,  damping=damping)
    # bp.run(max_iterations=max_iterations, tol=tol, progbar=progbar)
    # neighbors = bp.neighbors
    # messages = bp.messages
    # messages_squared = bp.messages_squared()
    # messages_squared_inv = bp.messages_squared_inv()
    # est_norm = complex(bp.contract())
    # print("y", est_norm)

    
    return peps_compressed


def tn_bp_gs(bp, tn, site_, opt="auto-hq"):
    tn = tn.copy()
    site_tags = bp.site_tags


    format_string = "I" + ",".join(["{}"] * len(site[0][0]))



    for site in site_:
        # Use the .format() method
        site_tag_cluster = [ format_string.format(*i) for i in site]
        
        
        
        site_tag_ = list(set(site_tags) - set(site_tag_cluster))
        
        cluster_neighbors = []
        for count in range(len(site)):
            neighbors_tag = bp.neighbors[site_tag_cluster[count]]
            neighbors_tag = [i for i in neighbors_tag if i not in site_tag_cluster]
            cluster_neighbors.append(neighbors_tag)
    
        cluster_neighbors_flat = [item for sublist in cluster_neighbors for item in sublist]
        massages = []
        for count, k in enumerate(cluster_neighbors):
            for k_ in k:
                #icoming massages from neighbors k_ to tn cluster
                t_ = bp.messages[k_, site_tag_cluster[count]]                
                t_.add_tag("M")
                massages.append(t_)
        tn_massages = qtn.TensorNetwork(massages)        

    tn_local = tn.select(site_tag_cluster, "any")
    tn_cluster = tn_local & tn_massages
    tn_local = []
    for count, i in enumerate(site_tag_):
        i_around = bp.neighbors[i]
        local_messages = []
        for j in i_around:
            local_messages.append(bp.messages[j, i])
        tn_local.append(qtn.TensorNetwork([tn[i]] +  local_messages))

    norm_cluster = complex(tn_cluster.contract(all, optimize=opt))
    norm_ = [ complex(i.contract(all, optimize=opt)) for i in tn_local] + [norm_cluster]
    norm = np.prod(norm_)
    res = {"norm":norm, "norm_":norm_, "tn_cluster":tn_cluster, "tn_local":tn_local, "site_tag_cluster":site_tag_cluster, }
    res |= {"site_tag_":site_tag_ , "tn":tn, "site_tag_neighbors":cluster_neighbors_flat, "norm_cluster":norm_cluster}

    return res


def shift_loop(loop_pair, x=0, y=0):
    # Define the shift values
    x_shift = x
    y_shift = y
    
    # Function to shift the coordinates
    def shift_coordinates(coordinate_pair, x_shift, y_shift):
        new_pair = []
        for coord in coordinate_pair:
            # Extract the numeric part from the string (e.g., 'I0,0' -> '0,0')
            prefix, coords = coord[0], coord[1:]
            x, y = map(int, coords.split(','))  # Split and convert to integers
    
            # Apply the shifts to x and y
            new_x = x + x_shift
            new_y = y + y_shift
    
            # Create the new string with shifted coordinates
            new_coord = f"{prefix}{new_x},{new_y}"
            new_pair.append(new_coord)
        return tuple(new_pair)
    
    loop_pair_ = [shift_coordinates(pair, x_shift, y_shift) for pair in loop_pair]
    return loop_pair_


def parse_vertex(vertex):
    # Parse a string like 'I1,0' to a tuple of integers (1, 0)
    return tuple(map(int, vertex[1:].split(',')))

def translate_vertex(vertex, dx, dy, Lx, Ly):
    x, y = parse_vertex(vertex)
    new_x = (x + dx) 
    new_y = (y + dy) 
    return new_x, new_y, f'I{new_x},{new_y}'

def translate_vertex_(vertex, dx, dy, Lx, Ly):
    x, y = parse_vertex(vertex)
    new_x = (x + dx) % Lx
    new_y = (y + dy) % Ly
    return f'I{new_x},{new_y}'



def translate_loop(loop, dx, dy, Lx, Ly):
    translated_loop = []
    break_ = 0
    for edge in loop:
        new_x0, new_y0, x_new_str  = translate_vertex(edge[0], dx, dy, Lx, Ly)
        new_x1, new_y1, y_new_str  = translate_vertex(edge[1], dx, dy, Lx, Ly)
        if new_x0 > Lx-1 or new_y0 > Ly-1:
            break_ +=1
        if new_x1 > Lx-1 or new_y1 > Ly-1:
            break_ +=1


        translated_edge = (x_new_str, y_new_str)

        translated_loop.append(translated_edge)
    

    if break_>0:
        return []
    else:
        return translated_loop


def translate_loop_(loop, dx, dy, Lx, Ly):
    translated_loop = []
    break_ = 0
    for edge in loop:
        x_new_str  = translate_vertex_(edge[0], dx, dy, Lx, Ly)
        y_new_str  = translate_vertex_(edge[1], dx, dy, Lx, Ly)
        translated_edge = (x_new_str, y_new_str)

        translated_loop.append(translated_edge)

    return translated_loop

def rho_bp_excited_loops_(tn, bp, pass_rho, edges_f, obs_tensor, pari_exclusive=None):
    res = req_backend(progbar=True)
    opt = res["opt"]

    res = bp_excited_loops_(tn, bp, edges_f, contract_=False, pari_exclusive=pari_exclusive)
    pass_rho |= {"tn_l": res["tn_l"],"inds_excit": res["inds_excit"],"tags_excit": res["tags_excit"],}
    res_rho = env_rho(pass_rho)
    #rho_data = res_rho["rho_data"]
    
    #flops_l = res_rho["flops_l"]
    if res_rho:
        rho_l = res_rho["rho_l"]
        obs_real_l = [ (complex((rho & obs_tensor).contract(all, optimize=opt))).real for rho in rho_l]
        obs_real = math.fsum(obs_real_l)
        obs_img = math.fsum([ (complex((rho & obs_tensor).contract(all, optimize=opt))).imag for rho in rho_l])
        obs_ = complex(obs_real, obs_img)
        obs_ *= 10**tn.exponent
        
        norm_loop_real = math.fsum([ complex(rho.trace(pass_rho['leftinds_rho'], pass_rho['rightinds_rho'])).real  for rho in rho_l])
        norm_loop_imag = math.fsum([ complex(rho.trace(pass_rho['leftinds_rho'],pass_rho['rightinds_rho'])).imag  for rho in rho_l])
        norm_loop = complex(norm_loop_real, norm_loop_imag)
        
        norm_loop *= (10**tn.exponent)

    else:
        obs_ = .0 
        norm_loop = .0
    
    return obs_, norm_loop





def loop_gen_local(bp, circum = 4, tags_cluster=[], sites=[], circum_g = None, tn_flat=None, site_tags=[], intersect=True):
    
    #all edges in tn:
    edges = [tuple(sorted(i)) for i in bp.edges.keys()]
    
    if tags_cluster:
        edges =  [ (a, b)  for (a, b) in edges if a in tags_cluster and b in tags_cluster] 
        #edges =  [ (a, b)  for (a, b) in edges if a in tags_cluster or b in tags_cluster] 
    
    max_edges = len(edges)
    #print("max: number of edges:", max_edges)
    
    
    loops_local = []
    if circum or circum==0:
        loops_local = combine_elements(edges, circum, sites=sites)
    

    loops_global = []
    if circum_g:
        if not tn_flat:
            print("provide tn to get global loops")
        tn_flat = tn_flat.copy()
        res = gen_loop(tn_flat, loop_length=circum_g, intersect=True, site_tags=site_tags)
        if  res["loop_pair"]:
            loops_global = res["loop_pair"]
        res = gen_loop(tn_flat, loop_length=circum_g, intersect=False, site_tags=site_tags)
        if  res["loop_pair"]:
            loops_global += res["loop_pair"]
        res = gen_loop_(tn_flat, loop_length=circum_g, site_tags=site_tags)
        if  res["loop_pair"]:
            loops_global += res["loop_pair"]
        
        loops_global = [tuple(i) for i in loops_global]
        #print("loops_global", len(loops_global))


    loops = loops_local+loops_global
    
    if loops and circum>0:
        loops = list(set(loops))

    return loops













def obs_bp_excited_loops_(tn, bp, tn_, bp_, edges_f, chi=40, contract_=True):

    res = bp_excited_loops_(tn_, bp_, edges_f, chi=chi, contract_=contract_)
    obs_real = math.fsum([z.real for z in res["excited"]])
    obs_image = math.fsum([z.imag for z in res["excited"]])
    obs_ = complex(obs_real, obs_image)
    
    
    res_ = bp_excited_loops_(tn, bp, edges_f, chi=chi, contract_=contract_)
    norm_real = math.fsum([z.real for z in res_["excited"]])
    norm_image = math.fsum([z.imag for z in res_["excited"]])
    norm_loop = complex(norm_real, norm_image)
    Z = obs_*10**(tn_.exponent)
    norm_loop = norm_loop*10**(tn.exponent)
    
    return Z, norm_loop



def get_peps(Lx=4, Ly=4, bnd=8, step=4, inds_z=["k1,1","k2,1"], length=4, 
             region_tag=[], 
             cycle=True, 
             method = "parallel",
            ):


    to_backend, opt_, opt = req_backend(progbar=False)
    to_backend_ = get_to_backend(to_backend)

    
    #pull out peps:
    L = Lx * Ly
    if cycle:
        res_peps = qu.load_from_disk(f"../Store/info_peps/res_peps{bnd}L{L}_d_{method}_cycle{cycle}")
    if not cycle:
        res_peps = qu.load_from_disk(f"../Store/info_peps/res_peps{bnd}L{L}_d_{method}_cycle{cycle}")


    #peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=bnd, seed=666)
    peps = res_peps["peps_l"][step]
    peps = peps.copy()
    #peps.apply_to_arrays(to_backend_)
    site_tags = peps.site_tags
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    tn = pepsH & peps

    
    # some info dic for plotting peps, or reindexing it
    fix = {f"I{i},{j}":(i,j) for i,j in itertools.product(range(Lx), range(Ly)) }
    inds_k = {f"k{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}
    inds_b = {f"b{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}



    # put all of them into a dic:
    res = {"peps":peps, "tn":tn, "Lx":Lx, "Ly":Ly, "peps_l":res_peps["peps_l"], "fix":fix}
    res |= {"site_tags":site_tags, "inds_b":inds_b, "inds_k":inds_k, "pepsH":pepsH}


    
    if inds_z:
        indx0, indx1 = inds_z
        Z = qu.pauli('Z')
        Z = to_backend_(Z)
        peps_ = qtn.tensor_network_gate_inds(peps, Z, [indx0], contract=True,  inplace=False)
        
        z_tn = pepsH & peps_
        
        peps_ = peps_.copy()
        peps_ = qtn.tensor_network_gate_inds(peps_, Z, [indx1], contract=True,  inplace=False)
        
        
        norm_tn = pepsH & peps
        zz_tn = pepsH & peps_
        
        res |= {"norm_peps":site_tags, "peps_z":z_tn, "peps_zz":zz_tn, "peps_norm":norm_tn }
    
    
    return res



def bp_cluster(tn, bp, Lx=None, Ly=None, length=2, progbar=False, chi=None, max_repeats_=2**6, parallel=True, cycle=True, contract=True):
    
    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats_,
        methods=  ('greedy-compressed', 'greedy-span', 'kahypar-agglom'),
        minimize='combo-compressed', 
        progbar=progbar,
        parallel=parallel,
        directory="cash/",
    )
    #to_backend, opt_, opt, copt = quf.req_backend(progbar=True, chi=chi)

    if cycle:
        gs_, excited_= get_pairs_mulitexcit_(Lx, Ly, length=length)
    if not cycle:
        gs_, excited_= get_pairs_mulitexcit(Lx, Ly, length=length)

    res_o = []
    tn_l = []
    for count, gs in tqdm(enumerate(gs_)):
        
        tn_appro = tn_bp_excitations(tn, bp, excited_[count], gs, loop=[])                     
        tn_l.append(tn_appro)
        tn_appro.full_simplify_(seq='R', split_method='svd', inplace=True)
        if contract:
            tnres_, _ = apply_hyperoptimized_compressed(tn_appro, copt, chi, tree_gauge_distance=4, progbar=False)
            main, exp = (tnres_.contract(), tnres_.exponent)
            excited = complex(main * 10**(exp))
            res_o.append(excited)
    
    res = {  "obs":res_o, "tn_l":tn_l}
    return res



def bp_excited_loops(tn_, bp_, loop_l, progbar=False, chi=None,max_repeats_=2**6, parallel=True, 
                     contract_=True, pari_exclusive=[], tree_gauge_distance=4, f_max = 15 , peak_max=34,
                    prgbar=True,opt_external=None,
                    ):
    
    # copt = ctg.ReusableHyperCompressedOptimizer(
    #     chi,
    #     max_repeats=max_repeats_,
    #     methods=  ('greedy-compressed', 'greedy-span', 'kahypar-agglom'),
    #     minimize='combo-compressed', 
    #     progbar=progbar,
    #     parallel=parallel,
    #     #directory="cash/",
    # )

    
    is_subset = False
    res = req_backend(progbar=progbar)
    opt = res["opt"]
    res = {}
    excit_ = []
    tn_l = []
    inds_excit = []
    inds_gs = []
    tags_excit = []
    # put all regions in below list
    pair_gs_l = []
    pair_excited_l = []
    
    for idx, loop in enumerate(loop_l):
        loop = [tuple(sorted(pair)) for pair in loop]
        pair_excited = loop
        
        if pair_excited:
            elem0, elem1 = pair_excited[0]
            if not isinstance(elem0, str):
                format_string = "I" + ",".join(["{}"] * len(elem0))
                pair_excited = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excited]
           
        all_edges = [tuple(sorted(i)) for i in bp_.edges.keys()]
        pair_gs = list(set(all_edges) - set(pair_excited))
        if pari_exclusive:
            
            if not set(pair_excited):
                # if pair_excited = empty that means "gs bp"  that we want to include
                is_subset = False
            else:
                # if pair_excited is in pari_exclusive we want to ignore to not include "double counting"
                is_subset = set(pair_excited).issubset(set(pari_exclusive))
            
            
            #remove the exclusive pairs from the both regions 
            pair_excited = list(set(pair_excited)-set(pari_exclusive))
            pair_gs = list(set(pair_gs)-set(pari_exclusive))

        if not is_subset:
            pair_gs_l.append(pair_gs)
            pair_excited_l.append(pair_excited)





    # eliminate the pairs that are common 

    # Helper function to normalize a sublist
    def normalize_sublist(sublist):
        return tuple(sorted(tuple(sorted(tup)) for tup in sublist))
    
    # Helper function to normalize a pair of sublists
    def normalize_pair(sublist1, sublist2):
        return (normalize_sublist(sublist1), normalize_sublist(sublist2))
    
    # Track unique pairs with a set
    unique_pairs = set()
    unique_list1 = []
    unique_list2 = []
    
    # Process each pair of sublists
    for sublist1, sublist2 in zip(pair_gs_l, pair_excited_l):
        normalized_pair = normalize_pair(sublist1, sublist2)
        # Add to unique lists only if the normalized pair hasn't been seen
        if normalized_pair not in unique_pairs:
            unique_pairs.add(normalized_pair)
            unique_list1.append(sublist1)
            unique_list2.append(sublist2)

    
    #print("redunction-->", len(pair_excited_l), len(unique_list1))
    pair_gs_l = unique_list1
    pair_excited_l = unique_list2


    for pair in pair_excited_l:
        pair = list(chain.from_iterable(pair))
        tags_excit.append(pair)


    flops_l = []
    peak_l = []
    with tqdm(total=len(pair_excited_l),  desc="bp:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
    
    
        for count in range(len(pair_excited_l)):
            pair_excited = pair_excited_l[count]
            pair_gs = pair_gs_l[count]
            inds_local = []
            inds_local_ = []
    
            if pair_excited:
                for pair in pair_excited:
                    inds = bp_.messages[pair].inds
                    for indx in inds:
                        inds_local_.append(indx)
            inds_local_ = list(set(inds_local_))
            
            
            if pair_gs:
                for pair in pair_gs:
                    inds = bp_.messages_[pair].inds
                    for indx in inds:
                        inds_local.append(indx)
            inds_local = list(set(inds_local))       
             
            inds_excit.append(inds_local_)
            inds_gs.append(inds_local)
            
            start_time = time.time()
            tn_appro = tn_bp_excitations(tn_, bp_, pair_excited, pair_gs, loop=[]) 
            #print("tn_bp_excitations", (time.time() - start_time))
            
    
            
            # if chi:
            #     to_backend, opt_, opt, copt = req_backend(progbar=False, chi=chi)
            # else:
            #     to_backend, opt_, opt = req_backend(progbar=False)

            
            if chi:
                res = req_backend(progbar=False, chi=chi)
                copt = res["copt"]
                backend_ = res["backend_"]
            else:
                res = req_backend(progbar=False, chi=chi)
                opt = res["opt"]
                backend_ = res["backend_"]
            
            
            
            
            
            #to_backend_ = get_to_backend(to_backend)
            #tn_appro.apply_to_arrays(to_backend_)
            
            
            tn_l.append(tn_appro.copy())

            
            start_time = time.time()        
            tn_appro.full_simplify_(seq='R', split_method='svd', inplace=True)
            #print("simplify", (time.time() - start_time))
            flops, peak = (1,1)
            c_time = 0

            if opt_external:
                opt = opt_external

            
            if chi:
                if contract_:
                    start_time = time.time()        
                    
                    tnres_, (flops, peak) = apply_hyperoptimized_compressed(tn_appro, copt, chi, 
                                                                tree_gauge_distance=tree_gauge_distance, 
                                                                progbar=progbar, f_max=f_max,
                                                                peak_max=peak_max)
                    
                    
                    if flops<1:
                        flops = 1
                    flops_l.append(np.log10(flops))
                    peak_l.append(peak)
                    
                    c_time = time.time() - start_time
                    #print("contraction", (time.time() - start_time))
                    
                    main, exp = (tnres_.contract(), tnres_.exponent)
                    excited = complex(main * 10**(exp))
                    excit_.append(excited)
    
            else:  
                if contract_:
                    excited = tn_appro.contract(all, optimize=opt)
                    excit_.append(excited)
        
            pbar.set_postfix({"flops": np.log10(flops), 
                              "peak": peak,
                              "c_time":c_time,
                              })
            pbar.refresh()
            pbar.update(1)

    res |= { "tn_l":tn_l, "excited":excit_,  "peak":peak_l, "flops":flops_l, "tags_excit":tags_excit, "inds_excit":inds_excit, "inds_gs":inds_gs }
    return res





def tn_bp_excitations(tn, bp, pair_excited=[], pair_gs=[], loop =[]):
    
    # copy the tensor network                 
    tn_appro = tn.copy()
    
    # get projectors into excited states of bp
    pr_excited = bp.projects

    # get redundancy in the tag pairs:
    pair_excit_tags = list(set(pair_excited))
    pair_gs_tags = list(set(pair_gs))

    # if format is not str 
    if pair_excit_tags:
        elem0, elem1 = pair_excit_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_excit_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excit_tags]

    if pair_gs_tags:
        elem0, elem1 = pair_gs_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_gs_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_gs_tags]


    
    if set(pair_excit_tags).intersection(pair_gs_tags):
        print("warnning: common pair in bp gs and excited state")
        print("warnning: decide about gs or excitation" )

    
    
    for pair in pair_excit_tags:
        
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        
        pr_excited_tn = pr_excited[tuple(pair_)].copy()
    
        # drop indicies in the right tensor and attach those indicies to pr_excited_tn
        tn_j = tn_appro.select(j)
        left_inds_ = pr_excited_tn.left_inds
        inds_ = pr_excited_tn.inds
        right_inds_ = [i for i in inds_ if i not in left_inds_]

        map_inds = { left_inds_[count]:right_inds_[count]      for count in range(len(right_inds_)) }
        tn_j.reindex_(map_inds)

    
        tn_appro = tn_appro & pr_excited_tn
    
    res = req_backend(progbar=False)
    opt = res["opt"]


    
    #print(tn_appro.draw("Me"))
    for pair in pair_gs_tags:
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        #select tensors with tag i and j and their common indicies
        tn_i = tn_appro.select(i)
        tn_j = tn_appro.select(j)
    
        # Get massages going to tensors with tag i (from j) and tensors with tag j (from i)
        mij = bp.messages[j,i].copy()
        mji = bp.messages[i,j].copy()
        inds_ij = list(mij.inds)
        
        #drop the bond between tensor_i and tensor_j and add massages m_ij to tensor_i and massage m_ji to tensor j
        map_inds_i = { i:qtn.rand_uuid() for i in inds_ij }
        map_inds_j = { i:qtn.rand_uuid() for i in inds_ij }
        
        mij.reindex_(map_inds_i)
        tn_i.reindex_(map_inds_i)
        
        mji.reindex_(map_inds_j)
        tn_j.reindex_(map_inds_j)
    
        # add some tags to massages
        mij.add_tag(["in", "Mg"])
        mji.add_tag(["out", "Mg"])
        
        tn_appro = tn_appro & mij & mji

    #print(tn_appro.contract(all, optimize=opt))
    return tn_appro





def bp_excited_loops_(tn_, bp_, loop_l, progbar=False, chi=None,max_repeats_=2**6, parallel=True, 
                     contract_=True, pari_exclusive=[], tree_gauge_distance=4, f_max = 15 , peak_max=34,
                    prgbar=True,opt_external="auto-hq",
                    ):
    

    pari_exclusive_flat = pari_exclusive

    tn_ = tn_.copy()
    is_subset = False
    res = {}
    excit_ = []
    tn_l = []
    inds_excit = []
    inds_gs = []
    
    # put all regions in below list
    pair_gs_l = []
    pair_excited_l = []
    
    for idx, loop in enumerate(loop_l):
        loop = [tuple(sorted(pair)) for pair in loop]
        pair_excited = loop
        pair_excited_flat = list(chain.from_iterable(pair_excited))
        
        if pair_excited:
            elem0, elem1 = pair_excited[0]
            if not isinstance(elem0, str):
                format_string = "I" + ",".join(["{}"] * len(elem0))
                pair_excited = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excited]
           
        
        
        # Ground-state massage pairs should not overlap with excited-state massage pairs.
        

        if pair_excited:
            all_edges = [tuple(sorted(i)) for i in bp_.edges.keys()]
            pair_gs = list(set(all_edges) - set(pair_excited))
            # Ground-state messages are valid only when there is a common area defined by excited-state messages
            pair_excited_flat = list(chain.from_iterable(pair_excited))
            pair_gs = [tup for tup in pair_gs if any(item in pair_excited_flat for item in tup)]
            # make sure there is no overlap with excited-state massage pairs: double-check
            pair_gs = list(set(pair_gs) - set(pair_excited))
        
        else:
            # Notice: if there is no pair_excited, include all gs-bp massages in the TN to fom ground-state TN == 1
            all_edges = [tuple(sorted(i)) for i in bp_.edges.keys()]
            pair_gs = list(set(all_edges))      

    
        if pari_exclusive:
            if not set(pair_excited):
                # if pair_excited = empty that means "gs bp"  that we want to include: proccedd 
                is_subset = False
            else:
                # if pair_excited is in pari_exclusive we want to ignore to not include "double counting": break for loop
                is_subset = set(pair_excited).issubset(set(pari_exclusive))
            
    

            
            if pair_excited:

                #remove the exclusive pairs from the both regions 
                pair_excited = list(set(pair_excited)-set(pari_exclusive))
                pair_excited_flat = list(chain.from_iterable(pair_excited))
    
               
                # Ground-state messages are valid only when there is a common area defined by excited/exclusive messages
                pair_flat = pari_exclusive_flat + pair_excited_flat
                pair_gs = list(set(all_edges) - set(pair_excited))
                pair_gs = [tup for tup in pair_gs if any(item in pair_flat for item in tup)]
                # make sure there is no common gs pair in both pair_excited or pari_exclusive
                pair_gs = list(set(pair_gs) - set(pair_excited))
                pair_gs = list(set(pair_gs) - set(pari_exclusive))
            else:
                # include all gs-bp massages, except the ones in pari_exclusive
                pair_gs = list(set(all_edges) - set(pair_excited))
                pair_gs = list(set(pair_gs) - set(pari_exclusive))


        # add excited/gs pairs into the list
        if not is_subset:
            pair_gs_l.append(pair_gs)
            pair_excited_l.append(pair_excited)







    # eliminate the pairs that are common 

    # Helper function to normalize a sublist
    def normalize_sublist(sublist):
        return tuple(sorted(tuple(sorted(tup)) for tup in sublist))
    
    # Helper function to normalize a pair of sublists
    def normalize_pair(sublist1, sublist2):
        return (normalize_sublist(sublist1), normalize_sublist(sublist2))
    
    # Track unique pairs with a set
    unique_pairs = set()
    unique_list1 = []
    unique_list2 = []
    
    # Process each pair of sublists
    for sublist1, sublist2 in zip(pair_gs_l, pair_excited_l):
        normalized_pair = normalize_pair(sublist1, sublist2)
        # Add to unique lists only if the normalized pair hasn't been seen
        if normalized_pair not in unique_pairs:
            unique_pairs.add(normalized_pair)
            unique_list1.append(sublist1)
            unique_list2.append(sublist2)

    
    #print("redunction-->", len(pair_excited_l), len(unique_list1))
    pair_gs_l = unique_list1
    pair_excited_l = unique_list2

    tags_excit = []
    
    # that is an important function that includes site tags that are defined the loop:
    for count, pair in enumerate(pair_excited_l):
        
        if pair:
            pair = list(chain.from_iterable(pair))
            tags_excit.append(pair+pari_exclusive_flat+["proj"])
        
        # In the rare case of no excitation loop, I want to include all tags
        else:
            pair = pair_gs_l[count]
            pair = list(chain.from_iterable(pair))
            if pari_exclusive:
                tags_excit.append(pair+pari_exclusive_flat)
            else:
                tags_excit.append(pair)



    flops_l = []
    peak_l = []
    with tqdm(total=len(pair_excited_l),  desc="sum-loop", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
    
    
        for count in range(len(pair_excited_l)):
            pair_excited = pair_excited_l[count]
            pair_gs = pair_gs_l[count]
            inds_local = []
            inds_local_ = []
    
            if pair_excited:
                for pair in pair_excited:
                    inds = bp_.messages[pair].inds
                    for indx in inds:
                        inds_local_.append(indx)
            inds_local_ = list(set(inds_local_))
            
            
            if pair_gs:
                for pair in pair_gs:
                    inds = bp_.messages_[pair].inds
                    for indx in inds:
                        inds_local.append(indx)
            inds_local = list(set(inds_local))       
             
            inds_excit.append(inds_local_)
            inds_gs.append(inds_local)
            
            start_time = time.time()
            tn_appro = tn_bp_excitations_(tn_, bp_, pair_excited, pair_gs, loop=[]) 
            #print("tn_bp_excitations", (time.time() - start_time))
            
    
            
            if chi:
                res = req_backend(progbar=False, chi=chi)
                copt = res["copt"]
                backend_ = res["backend_"]
            else:
                res = req_backend(progbar=False, chi=chi)
                opt = res["opt"]
                backend_ = res["backend_"]

    
            if tags_excit[count]:
                tn_appro = tn_appro.select(tags_excit[count], which="any")
                    
            tn_l.append(tn_appro.copy())
            
            start_time = time.time()        
            tn_appro.full_simplify_(seq='R', split_method='svd', inplace=True)
            #print("simplify", (time.time() - start_time)
            if opt_external:
                opt = opt_external
            
            flops, peak = (1,0)
            if contract_:
                tree = tn_appro.contraction_tree(opt)
                flops = tree.contraction_cost()
                peak = tree.peak_size(log=2)
                if flops<1:
                    flops = 1
                flops = np.log10(flops)
            
            
            c_time = 0
            if chi and flops>9.5:
                if contract_:
                    start_time = time.time()        
                    tnres_, (flops, peak) = apply_hyperoptimized_compressed(tn_appro, copt, chi, 
                                                                tree_gauge_distance=tree_gauge_distance, 
                                                                progbar=progbar, f_max=f_max,
                                                                peak_max=peak_max)
                    if flops<1:
                        flops = 1
                    flops = np.log10(flops)
                    flops_l.append(flops)
                    peak_l.append(peak)
                    
                    c_time = time.time() - start_time
                    #print("contraction", (time.time() - start_time))
                    
                    main, exp = (tnres_.contract(), tnres_.exponent)
                    excited = complex(main * 10**(exp))
                    excit_.append(excited)
    
            elif flops<=9.5:  
                if contract_:
                    excited = tn_appro.contract(all, optimize=opt)
                    excited = complex(excited)
                    excit_.append(excited)
                    flops_l.append(flops)
                    peak_l.append(peak)
        
            pbar.set_postfix({"flops": flops, 
                              "peak": peak,
                              "c_time":c_time,
                              })
            pbar.refresh()
            pbar.update(1)

    res |= { "tn_l":tn_l, "excited":excit_,  "peak":peak_l, "flops":flops_l, "tags_excit":tags_excit, "inds_excit":inds_excit, "inds_gs":inds_gs }
    return res





def tn_bp_excitations_(tn, bp, pair_excited=[], pair_gs=[], loop =[]):
    
    # copy the tensor network                 
    tn_appro = tn.copy()
    
    # get projectors into excited states of bp
    pr_excited = bp.projects

    # get redundancy in the tag pairs:
    pair_excit_tags = list(set(pair_excited))
    pair_gs_tags = list(set(pair_gs))

    # if format is not str 
    if pair_excit_tags:
        elem0, elem1 = pair_excit_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_excit_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_excit_tags]

    if pair_gs_tags:
        elem0, elem1 = pair_gs_tags[0]
        format_string = "I" + ",".join(["{}"] * len(elem0))

        if not isinstance(elem0, str):
            pair_gs_tags = [ (format_string.format(*a),format_string.format(*b))  for (a, b) in pair_gs_tags]


    
    if set(pair_excit_tags).intersection(pair_gs_tags):
        print("warnning: common pair in bp gs and excited state")
        print("warnning: decide about gs or excitation" )

    
    
    for pair in pair_excit_tags:
        
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        
        pr_excited_tn = pr_excited[tuple(pair_)].copy()
    
        # drop indicies in the right tensor and attach those indicies to pr_excited_tn
        tn_j = tn_appro.select(j)
        left_inds_ = pr_excited_tn.left_inds
        inds_ = pr_excited_tn.inds
        right_inds_ = [i for i in inds_ if i not in left_inds_]

        map_inds = { left_inds_[count]:right_inds_[count]      for count in range(len(right_inds_)) }
        tn_j.reindex_(map_inds)

    
        tn_appro = tn_appro & pr_excited_tn
    
    res = req_backend(progbar=False)
    opt = res["opt"]

    
    #print(tn_appro.draw("Me"))
    for pair in pair_gs_tags:
        pair_ = sorted(pair)
        #print("excited", pair_)
        i, j = pair_
        #select tensors with tag i and j and their common indicies
        tn_i = tn_appro.select(i)
        tn_j = tn_appro.select(j)
    
        # Get massages going to tensors with tag i (from j) and tensors with tag j (from i)
        mij = bp.messages[j,i].copy()
        mji = bp.messages[i,j].copy()
        inds_ij = list(mij.inds)
        
        #drop the bond between tensor_i and tensor_j and add massages m_ij to tensor_i and massage m_ji to tensor j
        map_inds_i = { i:qtn.rand_uuid() for i in inds_ij }
        map_inds_j = { i:qtn.rand_uuid() for i in inds_ij }
        
        mij.reindex_(map_inds_i)
        tn_i.reindex_(map_inds_i)
        
        mji.reindex_(map_inds_j)
        tn_j.reindex_(map_inds_j)
    
        # add some tags to massages
        mij.add_tag(["in", "Mg"])
        mji.add_tag(["out", "Mg"])
        
        tn_appro = tn_appro & mij & mji

    #print(tn_appro.contract(all, optimize=opt))
    return tn_appro






class L1BP(BeliefPropagationCommon):
    """Lazy 1-norm belief propagation. BP is run between groups of tensors
    defined by ``site_tags``. The message updates are lazy contractions.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region,
        which should not overlap. If the tensor network is structured, then
        these are inferred automatically.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    update : {'parallel', 'sequential'}, optional
        Whether to update all messages in parallel or sequentially.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """

    def __init__(
        self,
        tn,
        site_tags=None,
        damping=0.0,
        update="sequential",
        local_convergence=True,
        optimize="auto-hq",
        messages_=None,
        message_init_function=None,
        **contract_opts,
    ):
        self.backend = next(t.backend for t in tn)
        self.damping = damping
        self.local_convergence = local_convergence
        self.update = update
        self.optimize = optimize
        self.contract_opts = contract_opts
        self.messages_ = messages_
        self.projects = {}



        
        if site_tags is None:
            self.site_tags = tuple(tn.site_tags)
        else:
            self.site_tags = tuple(site_tags)

        (
            self.edges,
            self.neighbors,
            self.local_tns,
            self.touch_map,
        ) = create_lazy_community_edge_map(tn, site_tags)
        self.touched = oset()

        self._abs = ar.get_lib_fn(self.backend, "abs")
        self._max = ar.get_lib_fn(self.backend, "max")
        self._sum = ar.get_lib_fn(self.backend, "sum")
        _real = ar.get_lib_fn(self.backend, "real")
        _argmax = ar.get_lib_fn(self.backend, "argmax")
        _reshape = ar.get_lib_fn(self.backend, "reshape")
        self._norm = ar.get_lib_fn(self.backend, "linalg.norm")

        def _normalize(x):

            # sx = self._sum(x)
            # sphase = sx / self._abs(sx)
            # smag = self._norm(x)**0.5
            # return x / (smag * sphase)

            return x / self._sum(x)
            # return x / self._norm(x)
            # return x / self._max(x)
            # fx = _reshape(x, (-1,))
            # return x / fx[_argmax(self._abs(_real(fx)))]

        def _distance(x, y):
            return self._sum(self._abs(x - y))

        self._normalize = _normalize
        self._distance = _distance

        # for each meta bond create initial messages
        if self.messages_:
          self.messages = messages_  
        else:    
            self.messages = {}
            for pair, bix in self.edges.items():
                # compute leftwards and rightwards messages
                for i, j in (sorted(pair), sorted(pair, reverse=True)):
                    tn_i = self.local_tns[i]
                    # initial message just sums over dangling bonds
    
                    if message_init_function is None:
                        tm = tn_i.contract(
                            all,
                            output_inds=bix,
                            optimize=self.optimize,
                            drop_tags=True,
                            **self.contract_opts,
                        )
                        # normalize
                        tm.modify(apply=self._normalize)
                    else:
                        shape = tuple(tn_i.ind_size(ix) for ix in bix)
                        tm = qtn.Tensor(
                            data=message_init_function(shape),
                            inds=bix,
                        )
    
                    self.messages[i, j] = tm
        
        self.messages_ = self.messages



        
        # compute the contractions
        self.contraction_tns = {}
        for pair, bix in self.edges.items():
            # for each meta bond compute left and right contractions
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                tn_i = self.local_tns[i].copy()
                # attach incoming messages to dangling bonds
                tks = [
                    self.messages[k, i] for k in self.neighbors[i] if k != j
                ]
                # virtual so we can modify messages tensors inplace
                tn_i_to_j = qtn.TensorNetwork((tn_i, *tks), virtual=True)
                self.contraction_tns[i, j] = tn_i_to_j

    def iterate(self, tol=5e-6):
        if (not self.local_convergence) or (not self.touched):
            # assume if asked to iterate that we want to check all messages
            self.touched.update(
                pair for edge in self.edges for pair in (edge, edge[::-1])
            )

        ncheck = len(self.touched)
        nconv = 0
        max_mdiff = -1.0
        new_touched = oset()

        def _compute_m(key):
            i, j = key
            bix = self.edges[(i, j) if i < j else (j, i)]
            tn_i_to_j = self.contraction_tns[i, j]
            tm_new = tn_i_to_j.contract(
                all,
                output_inds=bix,
                optimize=self.optimize,
                **self.contract_opts,
            )
            return self._normalize(tm_new.data)

        def _update_m(key, data):
            nonlocal nconv, max_mdiff

            tm = self.messages[key]

            if callable(self.damping):
                damping_m = self.damping()
                data = (1 - damping_m) * data + damping_m * tm.data
            elif self.damping != 0.0:
                data = (1 - self.damping) * data + self.damping * tm.data

            mdiff = float(self._distance(tm.data, data))

            if mdiff > tol:
                # mark touching messages for update
                new_touched.update(self.touch_map[key])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            tm.modify(data=data)

        if self.update == "parallel":
            new_data = {}
            # compute all new messages
            while self.touched:
                key = self.touched.pop()
                new_data[key] = _compute_m(key)
            # insert all new messages
            for key, data in new_data.items():
                _update_m(key, data)

        elif self.update == "sequential":
            # compute each new message and immediately re-insert it
            while self.touched:
                key = self.touched.pop()
                data = _compute_m(key)
                _update_m(key, data)

        self.touched = new_touched
        return nconv, ncheck, max_mdiff

    def contract(self, strip_exponent=False):
        tvals = []
        for site, tn_ic in self.local_tns.items():
            if site in self.neighbors:
                tval = qtn.tensor_contract(
                    *tn_ic,
                    *(self.messages[k, site] for k in self.neighbors[site]),
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            else:
                # site exists but has no neighbors
                tval = tn_ic.contract(
                    all,
                    output_inds=(),
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            tvals.append(tval)

        mvals = []
        for i, j in self.edges:
            mval = qtn.tensor_contract(
                self.messages[i, j],
                self.messages[j, i],
                optimize=self.optimize,
                **self.contract_opts,
            )
            mvals.append(mval)

        return combine_local_contractions(
            tvals, mvals, self.backend, strip_exponent=strip_exponent
        )

    def normalize_messages(self):
        """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
        `<m_i|m_i> = <m_j|m_j>` (but in general != 1).
        """
        for i, j in self.edges:
            tmi = self.messages[i, j]
            tmi.add_tag(j)
            tmj = self.messages[j, i]
            tmj.add_tag(i)
            
            nij = (tmi @ tmj)**0.5
            nii = (tmi @ tmi)**0.25
            njj = (tmj @ tmj)**0.25
            #print(nii,nij, njj)

            tmi /= (nij * nii / njj)
            tmj /= (nij * njj / nii)

    def cal_projects(self):
        pr_excited = {}
        for pair, bix in self.edges.items():
                pair_ = sorted(pair) 
                i, j = pair_
                tmi = self.messages[j, i].copy()
                tmj = self.messages[i, j].copy()
                
                #left indicies are similar to bp massages (the bond connect i and j)
                left_inds = list(tmi.inds)

                
                right_inds = list(tmj.inds)
                tmj = tmj.transpose(*right_inds)
                #right indicies are chosen randomly
                right_inds = [ qtn.rand_uuid() for indx in right_inds]
                
                tmi = tmi.transpose(*left_inds)
                
                
            
                # make projector into excited bp manifold
                p_excited_ij = projector(tmi, tmj)
            
                #Store tensors
                pr_excited[i, j] =  qtn.Tensor(p_excited_ij, left_inds=left_inds, inds=left_inds+right_inds, tags=["proj"])
        
        
        
        self.projects = pr_excited
        


def contract_l1bp(
    tn,
    max_iterations=1000,
    tol=5e-6,
    site_tags=None,
    damping=0.0,
    update="sequential",
    local_convergence=True,
    optimize="auto-hq",
    strip_exponent=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Estimate the contraction of ``tn`` using lazy 1-norm belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to contract.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region. If
        the tensor network is structured, then these are inferred
        automatically.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    update : {'parallel', 'sequential'}, optional
        Whether to update all messages in parallel or sequentially.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    progbar : bool, optional
        Whether to show a progress bar.
    strip_exponent : bool, optional
        Whether to strip the exponent from the final result. If ``True``
        then the returned result is ``(mantissa, exponent)``.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """
    bp = L1BP(
        tn,
        site_tags=site_tags,
        damping=damping,
        local_convergence=local_convergence,
        update=update,
        optimize=optimize,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.contract(
        strip_exponent=strip_exponent,
    )

    
def state_ket(
        tn,
        site,
        neighbors,
        messages_squared,
        messages_squared_inv,
        normalized=True,
        optimize="auto-hq",
        normalize = False, 
    ):
        site=qtn.oset(site)
        site=list(site)


        #qtn.oset()
        site_tag = ["I{},{}".format(*i) for i in site]
        ket_site_ind = ["k{},{}".format(*i)  for i in site]
        bra_site_ind = {i:i.replace("k", "b") for i in ket_site_ind}
    
        ks = []
        for count in range(len(site)):
            neighbors_tag = neighbors[site_tag[count]]
            neighbors_tag = [i for i in neighbors_tag if i not in site_tag]
            ks.append(neighbors_tag)


        tn_rho_i = []
        for count in range(len(site)):
    
            
            tn_ = tn[site_tag[count]]
            if isinstance(tn_, qtn.Tensor):
                tn_rho_i.append(tn_.copy())
            else:
                for t in tn_:
                    tn_rho_i.append(t.copy())
                

        for i in tn_rho_i:
            i.add_tag('KET')


        t_messages_squared_inv = []
        for count, k in enumerate(ks):
            for k_ in k:
                t_ = messages_squared_inv[k_, site_tag[count]]                
                t_.add_tag("messages_inv")
                t_.add_tag(k_.replace("I", "out"))
                t_.add_tag(site_tag[count].replace("I", "in"))
                t_messages_squared_inv.append(t_)

                
                t = messages_squared[k_, site_tag[count]]                
                t.add_tag(k_.replace("I", "out"))
                t.add_tag(site_tag[count].replace("I", "in"))
                t.add_tag("messages")
                tn_rho_i[count] &= t 

        
        rho = qtn.TensorNetwork(tn_rho_i) 

        return rho, t_messages_squared_inv





# class L2BP(BeliefPropagationCommon):
#     """Lazy (as in multiple uncontracted tensors per site) 2-norm (as in for
#     wavefunctions and operators) belief propagation.

#     Parameters
#     ----------
#     tn : TensorNetwork
#         The tensor network to form the 2-norm of and run BP on.
#     site_tags : sequence of str, optional
#         The tags identifying the sites in ``tn``, each tag forms a region,
#         which should not overlap. If the tensor network is structured, then
#         these are inferred automatically.
#     damping : float, optional
#         The damping parameter to use, defaults to no damping.
#     update : {'parallel', 'sequential'}, optional
#         Whether to update all messages in parallel or sequentially.
#     local_convergence : bool, optional
#         Whether to allow messages to locally converge - i.e. if all their
#         input messages have converged then stop updating them.
#     optimize : str or PathOptimizer, optional
#         The path optimizer to use when contracting the messages.
#     contract_opts
#         Other options supplied to ``cotengra.array_contract``.
#     """

#     def __init__(
#         self,
#         tn,
#         site_tags=None,
#         damping=0.0,
#         update="sequential",
#         local_convergence=True,
#         optimize="auto-hq",
#         messages_=None,
#         **contract_opts,
#     ):
#         self.backend = next(t.backend for t in tn)
#         self.damping = damping
#         self.local_convergence = local_convergence
#         self.update = update
#         self.optimize = optimize
#         self.contract_opts = contract_opts

#         if site_tags is None:
#             self.site_tags = tuple(tn.site_tags)
#         else:
#             self.site_tags = tuple(site_tags)

#         (
#             self.edges,
#             self.neighbors,
#             self.local_tns,
#             self.touch_map,
#         ) = create_lazy_community_edge_map(tn, site_tags)
#         self.touched = oset()

#         _abs = ar.get_lib_fn(self.backend, "abs")
#         _sum = ar.get_lib_fn(self.backend, "sum")
#         _transpose = ar.get_lib_fn(self.backend, "transpose")
#         _conj = ar.get_lib_fn(self.backend, "conj")

#         def _normalize(x):
#             return x / _sum(x)

#         def _symmetrize(x):
#             N = ar.ndim(x)
#             perm = (*range(N // 2, N), *range(0, N // 2))
#             return x + _conj(_transpose(x, perm))

#         def _distance(x, y):
#             return _sum(_abs(x - y))

#         self._normalize = _normalize
#         self._symmetrize = _symmetrize
#         self._distance = _distance

#         # initialize messages

#         self.messages = {}

#         for pair, bix in self.edges.items():
#             #print(pair, bix)
#             cix = tuple(ix + "_l2bp*" for ix in bix)
#             remapper = dict(zip(bix, cix))
#             output_inds = cix + bix

#             # compute leftwards and righwards messages
#             for i, j in (sorted(pair), sorted(pair, reverse=True)):
#                 tn_i = self.local_tns[i]
#                 tn_i2 = tn_i & tn_i.conj().reindex_(remapper)
#                 tm = tn_i2.contract(
#                     all,
#                     output_inds=output_inds,
#                     optimize=self.optimize,
#                     drop_tags=True,
#                     **self.contract_opts,
#                 )
#                 tm.modify(apply=self._symmetrize)
#                 tm.modify(apply=self._normalize)
                
#                 if messages_:
#                     #print("messages_")
#                     t_ = messages_[i, j]
#                     inds = tm.inds
#                     inds_ = t_.inds
#                     t_.reindex_({inds_[i]:inds[i] for i in range(len(inds))})

#                     for inds_ in inds:
#                         new_bond_dim = tm.ind_size(inds_)
#                         t_.expand_ind(inds_, new_bond_dim, rand_strength=0.0)
#                     if t_.data.shape != tm.data.shape:
#                         print("warnning", t_.data.shape, tm.data.shape)                
#                     tm.modify(data=t_.data)
                

#                 self.messages[i, j] = tm

#         # initialize contractions
#         self.contraction_tns = {}
#         for pair, bix in self.edges.items():
#             for i, j in (sorted(pair), sorted(pair, reverse=True)):
#                 # form the ket side and messages
#                 tn_i_left = self.local_tns[i]
#                 # get other incident nodes which aren't j
#                 ks = [k for k in self.neighbors[i] if k != j]
#                 tks = [self.messages[k, i] for k in ks]

#                 # form the 'bra' side
#                 tn_i_right = tn_i_left.conj()
#                 # get the bonds that attach the bra to messages
#                 outer_bix = {
#                     ix for k in ks for ix in self.edges[tuple(sorted((k, i)))]
#                 }
#                 # need to reindex to join message bonds, and create bra outputs
#                 remapper = {}
#                 for ix in tn_i_right.ind_map:
#                     if ix in bix:
#                         # bra outputs
#                         remapper[ix] = ix + "_l2bp**"
#                     elif ix in outer_bix:
#                         # messages connected
#                         remapper[ix] = ix + "_l2bp*"
#                     # remaining indices are either internal and will be mangled
#                     # or global outer indices and will be contracted directly

#                 tn_i_right.reindex_(remapper)

#                 self.contraction_tns[i, j] = qtn.TensorNetwork(
#                     (tn_i_left, *tks, tn_i_right), virtual=True
#                 )

#     def iterate(self, tol=5e-6):
#         if (not self.local_convergence) or (not self.touched):
#             # assume if asked to iterate that we want to check all messages
#             self.touched.update(
#                 pair for edge in self.edges for pair in (edge, edge[::-1])
#             )

#         ncheck = len(self.touched)
#         nconv = 0
#         max_mdiff = -1.0
#         new_touched = oset()

#         def _compute_m(key):
#             i, j = key
#             bix = self.edges[(i, j) if i < j else (j, i)]
#             cix = tuple(ix + "_l2bp**" for ix in bix)
#             output_inds = cix + bix

#             tn_i_to_j = self.contraction_tns[i, j]

#             tm_new = tn_i_to_j.contract(
#                 all,
#                 output_inds=output_inds,
#                 drop_tags=True,
#                 optimize=self.optimize,
#                 **self.contract_opts,
#             )
#             tm_new.modify(apply=self._symmetrize)
#             tm_new.modify(apply=self._normalize)
#             return tm_new.data

#         def _update_m(key, data):
#             nonlocal nconv, max_mdiff

#             tm = self.messages[key]

#             if self.damping > 0.0:
#                 data = (1 - self.damping) * data + self.damping * tm.data

#             try:
#                 mdiff = float(self._distance(tm.data, data))
#             except (TypeError, ValueError):
#                 # handle e.g. lazy arrays
#                 mdiff = float("inf")

#             if mdiff > tol:
#                 # mark touching messages for update
#                 new_touched.update(self.touch_map[key])
#             else:
#                 nconv += 1

#             max_mdiff = max(max_mdiff, mdiff)
#             tm.modify(data=data)

#         if self.update == "parallel":
#             new_data = {}
#             # compute all new messages
#             while self.touched:
#                 key = self.touched.pop()
#                 new_data[key] = _compute_m(key)
#             # insert all new messages
#             for key, data in new_data.items():
#                 _update_m(key, data)

#         elif self.update == "sequential":
#             # compute each new message and immediately re-insert it
#             while self.touched:
#                 key = self.touched.pop()
#                 data = _compute_m(key)
#                 _update_m(key, data)

#         self.touched = new_touched

#         return nconv, ncheck, max_mdiff

#     def normalize_messages(self):
#         """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
#         `<m_i|m_i> = <m_j|m_j>` (but in general != 1).
#         """
#         for i, j in self.edges:
#             tmi = self.messages[i, j]
#             tmj = self.messages[j, i]
#             nij = (tmi @ tmj)**0.5
#             nii = (tmi @ tmi)**0.25
#             njj = (tmj @ tmj)**0.25
#             tmi /= (nij * nii / njj)
#             tmj /= (nij * njj / nii)

#     def contract(self, strip_exponent=False):
#         """Estimate the contraction of the norm squared using the current
#         messages.
#         """
#         tvals = []
#         for i, ket in self.local_tns.items():
#             # we allow missing keys here for tensors which are just
#             # disconnected but still appear in local_tns
#             ks = self.neighbors.get(i, ())
#             bix = [ix for k in ks for ix in self.edges[tuple(sorted((k, i)))]]
#             bra = ket.H.reindex_({ix: ix + "_l2bp*" for ix in bix})
#             tni = qtn.TensorNetwork(
#                 (
#                     ket,
#                     *(self.messages[k, i] for k in ks),
#                     bra,
#                 )
#             )
#             tvals.append(
#                 tni.contract(all, optimize=self.optimize, **self.contract_opts)
#             )

#         mvals = []
#         for i, j in self.edges:
#             mvals.append(
#                 (self.messages[i, j] & self.messages[j, i]).contract(
#                     all,
#                     optimize=self.optimize,
#                     **self.contract_opts,
#                 )
#             )

#         return combine_local_contractions(
#             tvals, mvals, self.backend, strip_exponent=strip_exponent
#         )

#     def partial_trace(
#         self,
#         site,
#         normalized=True,
#         optimize="auto-hq",
#     ):
#         site=qtn.oset(site)
#         site=list(site)

#         example_tn = next(tn for tn in self.local_tns.values())

#         #qtn.oset()
#         site_tag = [example_tn.site_tag(i) for i in site]
#         ket_site_ind = [example_tn.site_ind(i)  for i in site]
#         bra_site_ind = {i:i.replace("k", "b") for i in ket_site_ind}
    
#         ks = []
#         for count in range(len(site)):
#             neighbors_tag = self.neighbors[site_tag[count]]
#             neighbors_tag = [i for i in neighbors_tag if i not in site_tag]
#             ks.append(neighbors_tag)

#         #tn_rho_i=self.tn.select(site_tag, which="any")
#         tn_rho_i = [self.local_tns[site_tag[count]].copy() for count in range(len(site))] 
#         tn_rho_i = list(itertools.chain(*tn_rho_i))
#         for i in tn_rho_i:
#             i.add_tag('KET')

#         tn_bra_is = [i.H for i in tn_rho_i]
#         for i in tn_bra_is:
#             i.retag_({'KET': 'BRA'})

#         for count, k in enumerate(ks):
#             for k_ in k:
#                 t = self.messages[k_, site_tag[count]]
#                 t.add_tag("M")
#                 tn_rho_i[count] &= t 

#         outer_bix = []
#         for count, k in enumerate(ks):
#             for k_ in k:
#                 index = self.edges[   tuple(sorted((k_, site_tag[count])))]
#                 ind, = index
#                 outer_bix.append( ind  )
#             #outer_bix.append( ix  for ix in self.edges[tuple(sorted((k_, site_tag[count])))] )
    
#         index_map = { ix:ix + "_l2bp*" for ix in outer_bix}
#         tn_bra = qtn.TensorNetwork(tn_bra_is)
#         tn_bra.reindex_(index_map)
#         tn_bra.reindex_(bra_site_ind)
        
#         rho = qtn.TensorNetwork(tn_rho_i) & tn_bra

#         return rho, outer_bix

#     def messages_squared(self):
#         messages = self.messages
#         messages_squared = {}
#         for i in messages:
#             tn = messages[i]
#             tn = tn.copy()
#             inds = list(tn.inds)
#             ndim = len(inds)//2
#             tn_dense = tn.to_dense(inds[:ndim],inds[ndim:] )
            
#             s2, W = ar.do("linalg.eigh", tn_dense)
#             W = ar.dag(W)
#             s2 = ar.do("clip", s2, s2[-1] * 1e-12, None)
#             s = ar.do("sqrt", s2)

#             s2 = ar.do("diag", s2)
#             s2 = ar.astype(s2, W.dtype)
#             #print( ar.dag(W) @ s2 @ W  - tn_dense)
            
#             s = ar.do("diag", s)
#             s = ar.astype(s, W.dtype)
            
#             tn_dense_squared = ar.dag(W) @ s  @ W 
#             tn_dense_squared = ar.do("reshape", tn_dense_squared, tn.data.shape)
#             tn.modify(data=tn_dense_squared)
#             messages_squared |= {i:tn}
#         return messages_squared
#     def messages_squared_inv(self):
#         messages = self.messages
#         messages_squared = {}
#         for i in messages:
#             tn = messages[i]
#             tn = tn.copy()
#             inds = list(tn.inds)
#             ndim = len(inds)//2
#             tn_dense = tn.to_dense(inds[:ndim],inds[ndim:] )
            
#             s2, W = ar.do("linalg.eigh", tn_dense)
#             W = ar.dag(W)
#             s2 = ar.do("clip", s2, s2[-1] * 1e-12, None)
#             s = ar.do("sqrt", s2)
#             s_inv = 1/s


#             s_inv = ar.do("diag", s_inv)
#             s_inv = ar.astype(s_inv, W.dtype)
            
#             #print( ar.dag(W) @ s2 @ W  - tn_dense)
            
        
            
#             tn_dense_squared = ar.dag(W) @ s_inv  @ W 
#             tn_dense_squared = ar.do("reshape", tn_dense_squared, tn.data.shape)
#             tn.modify(data=tn_dense_squared)
#             messages_squared |= {i:tn}
#         return messages_squared


#     def compress(
#         self,
#         tn,
#         max_bond=None,
#         cutoff=5e-6,
#         cutoff_mode="rsum2",
#         renorm=0,
#         lazy=False,
#     ):
#         """Compress the state ``tn``, assumed to matched this L2BP instance,
#         using the messages stored.
#         """
#         for (i, j), bix in self.edges.items():
#             tml = self.messages[i, j]
#             tmr = self.messages[j, i]

#             bix_sizes = [tml.ind_size(ix) for ix in bix]
#             dm = math.prod(bix_sizes)

#             ml = ar.reshape(tml.data, (dm, dm))
#             dl = self.local_tns[i].outer_size() // dm
#             Rl = qtn.decomp.squared_op_to_reduced_factor(
#                 ml, dl, dm, right=True
#             )

#             mr = ar.reshape(tmr.data, (dm, dm)).T
#             dr = self.local_tns[j].outer_size() // dm
#             Rr = qtn.decomp.squared_op_to_reduced_factor(
#                 mr, dm, dr, right=False
#             )

#             Pl, Pr = qtn.decomp.compute_oblique_projectors(
#                 Rl,
#                 Rr,
#                 cutoff_mode=cutoff_mode,
#                 renorm=renorm,
#                 max_bond=max_bond,
#                 cutoff=cutoff,
#             )

#             Pl = ar.do("reshape", Pl, (*bix_sizes, -1))
#             Pr = ar.do("reshape", Pr, (-1, *bix_sizes))

#             ltn = tn.select(i)
#             rtn = tn.select(j)

#             new_lix = [qtn.rand_uuid() for _ in bix]
#             new_rix = [qtn.rand_uuid() for _ in bix]
#             new_bix = [qtn.rand_uuid()]
#             ltn.reindex_(dict(zip(bix, new_lix)))
#             rtn.reindex_(dict(zip(bix, new_rix)))

#             # ... and insert the new projectors in place
#             tn |= qtn.Tensor(Pl, inds=new_lix + new_bix, tags=[i,"proj"])
#             tn |= qtn.Tensor(Pr, inds=new_bix + new_rix, tags=[j,"proj"])

#         if not lazy:
#             for st in self.site_tags:
#                 try:
#                     tn.contract_tags_(
#                         st, optimize=self.optimize, **self.contract_opts
#                     )
#                 except KeyError:
#                     pass

#         return tn


def contract_l2bp(
    tn,
    site_tags=None,
    damping=0.0,
    update="sequential",
    local_convergence=True,
    optimize="auto-hq",
    max_iterations=1000,
    tol=5e-6,
    strip_exponent=False,
    info=None,
    progbar=False,
    messages_=None,
    **contract_opts,
):
    """Estimate the norm squared of ``tn`` using lazy belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to estimate the norm squared of.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    update : {'parallel', 'sequential'}, optional
        Whether to update all messages in parallel or sequentially.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The contraction strategy to use.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
    strip_exponent : bool, optional
        Whether to strip the exponent from the final result. If ``True``
        then the returned result is ``(mantissa, exponent)``.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    progbar : bool, optional
        Whether to show a progress bar.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """
    bp = L2BP(
        tn,
        site_tags=site_tags,
        damping=damping,
        update=update,
        local_convergence=local_convergence,
        optimize=optimize,
        messages_ = messages_
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.contract(strip_exponent=strip_exponent)


def compress_l2bp(
    tn,
    max_bond,
    cutoff=0.0,
    cutoff_mode="rsum2",
    max_iterations=1000,
    tol=5e-6,
    site_tags=None,
    damping=0.0,
    update="sequential",
    local_convergence=True,
    optimize="auto-hq",
    lazy=False,
    inplace=False,
    info=None,
    progbar=False,
    messages_=None,
    normalize_messages = False,
    **contract_opts,
):
    """Compress ``tn`` using lazy belief propagation, producing a tensor
    network with a single tensor per site.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to form the 2-norm of, run BP on and then compress.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The cutoff to use when compressing.
    cutoff_mode : int, optional
        The cutoff mode to use when compressing.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region. If
        the tensor network is structured, then these are inferred
        automatically.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    update : {'parallel', 'sequential'}, optional
        Whether to update all messages in parallel or sequentially.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    lazy : bool, optional
        Whether to perform the compression lazily, i.e. to leave the computed
        compression projectors uncontracted.
    inplace : bool, optional
        Whether to perform the compression inplace.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    progbar : bool, optional
        Whether to show a progress bar.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.

    Returns
    -------
    TensorNetwork
    """
    tnc = tn if inplace else tn.copy()

    bp = L2BP(
        tnc,
        site_tags=site_tags,
        damping=damping,
        update=update,
        local_convergence=local_convergence,
        optimize=optimize,
        messages_ = messages_,
        **contract_opts,
    )
    if normalize_messages:
        bp.normalize_messages()
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    bp.compress(
        tnc,
        max_bond=max_bond,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        lazy=lazy,
    )
    return bp.messages 


def rho_bp_o(bp, site_t, inds_b, to_backend, z_inds=["k2,2", "k2,3"]):
    opt = opt_contraction_path(optlib='nevergrad', max_time='equil:256',
                               target_size=2**34, alpha = "flops",)

    inds_b = inds_b.copy()
    rho, inds = bp.partial_trace(site_t, optimize=opt)
    Z = to_backend(qu.pauli("Z"))

    rho_ = rho.reindex(inds_b)    
    rhoz=qtn.tensor_network_gate_inds(rho, Z, [z_inds[0]], contract=True,  inplace=False)
    rhozz=qtn.tensor_network_gate_inds(rhoz, Z, [z_inds[1]], contract=True,  inplace=False)
    rhoz.reindex(inds_b, inplace=True)
    rhozz.reindex(inds_b, inplace=True)
    rho_.rank_simplify_()
    rho_.fuse_multibonds_()
    rhoz.rank_simplify_()
    rhoz.fuse_multibonds_()
    rhozz.rank_simplify_()
    rhozz.fuse_multibonds_()
    
    z = rhoz.contract(all, optimize=opt)
    zz = rhozz.contract(all, optimize=opt)
    norm = rho_.contract(all, optimize=opt)
    
    z = complex(z/norm)
    zz = complex(zz/norm)

    
    return z, zz

def rho_bp_o_hyper(bp, site_t, inds_b, to_backend, z_inds=["k2,2", "k2,3"], chi_=40):
    copt = ctg.ReusableHyperCompressedOptimizer(
            chi_,
            max_repeats=2**8,
            minimize='combo-compressed', 
            #max_time="rate:1e8",
            progbar=True,
            parallel=True,
            #directory="cash/",
            )
    opt = opt_contraction_path( max_time='equil:256',
                               target_size=2**34, alpha = "flops",)

    inds_b = inds_b.copy()
    rho, inds = bp.partial_trace(site_t, optimize=opt)
    Z = to_backend(qu.pauli("Z"))

    rho_ = rho.reindex(inds_b)    
    rhoz=qtn.tensor_network_gate_inds(rho, Z, [z_inds[0]], contract=True,  inplace=False)
    rhozz=qtn.tensor_network_gate_inds(rhoz, Z, [z_inds[1]], contract=True,  inplace=False)
    rhoz.reindex(inds_b, inplace=True)
    rhozz.reindex(inds_b, inplace=True)
    rho_.rank_simplify_()
    rho_.fuse_multibonds_()
    rhoz.rank_simplify_()
    rhoz.fuse_multibonds_()
    rhozz.rank_simplify_()
    rhozz.fuse_multibonds_()


    res_X = apply_hyperoptimized_compressed(rhoz, copt, chi_,
                                        tree_gauge_distance=4, 
                                        progbar=True, 
                                        cutoff=1.e-12
                                     )
    main, exp=(res_X.contract(), res_X.exponent)
    z = (main) * 10**(exp)
    res_X = apply_hyperoptimized_compressed(rhozz, copt, chi_,
                                        tree_gauge_distance=4, 
                                        progbar=True, 
                                        cutoff=1.e-12
                                     )
    main, exp=(res_X.contract(), res_X.exponent)
    zz = (main) * 10**(exp)

    res_X = apply_hyperoptimized_compressed(rho_, copt, chi_,
                                        tree_gauge_distance=4, 
                                        progbar=True, 
                                        cutoff=1.e-12
                                     )
    main, exp=(res_X.contract(), res_X.exponent)
    norm = (main) * 10**(exp)
    
    z = complex(z/norm)
    zz = complex(zz/norm)

    
    return z, zz

def rho_bp(bp, site_t, inds_b, output_inds=["k2,2", "k2,3", "b2,2", "b2,3"], contract = True, absorb_guage=False, fuse = True):
    inds_b = inds_b.copy()
    opt = opt_contraction_path(optlib='nevergrad', max_time='equil:256',
                               target_size=2**34, alpha = "flops",)


    keys_to_remove = [i for i in output_inds if i.startswith("b")]
    # Remove keys from my_dict using del statement
    for key in keys_to_remove:
        if key in inds_b:
            del inds_b[key]

    rho, inds = bp.partial_trace(site_t, optimize=opt)
    rho.reindex_(inds_b)
    
    if absorb_guage:
        for ind in inds:
            rho.contract_ind(ind)
        if fuse:
            rho.fuse_multibonds_()
    
    if contract:
        rho = rho.contract(all, output_inds=output_inds, optimize=opt)
    
    
    return qtn.TensorNetwork([rho])

def rho_o(rho, to_backend, z_inds=["k2,2"], pauli_="Z"):
    opt = opt_contraction_path( max_time='equil:256',
                               target_size=2**34, alpha = "flops",)

    Z = to_backend(pauli_)
    dic = { i:i.replace("k", "b")  for i in z_inds}
    rho_ = rho.reindex(dic, inplace=False)

    rhoz=qtn.tensor_network_gate_inds(rho, Z, [z_inds[0]], contract=True,  inplace=False)
    #rhozz=qtn.tensor_network_gate_inds(rhoz, Z, [z_inds[1]], contract=True,  inplace=False)
    rhoz.reindex(dic, inplace=True)
    #rhozz.reindex(dic, inplace=True)
    norm = rho_.contract(all, optimize=opt)
    z = rhoz.contract(all, optimize=opt)
    #zz = rhozz.contract(all, optimize=opt)
    
    z = complex(z/norm)
    #zz = complex(zz/norm)
    
    return z#, zz
    
def rho_o_hyper(rho, to_backend, pauli_=None,output_inds=["k2,2", "k2,3", "b2,2", "b2,3"], z_inds=["k2,2", "k2,3"], chi_=40):
    copt = ctg.ReusableHyperCompressedOptimizer(
            chi_,
            max_repeats=2**8,
            minimize='combo-compressed', 
            #max_time="rate:1e8",
            progbar=False,
            parallel=True,
            #directory="cash/",
            )
    
    
    rho_, (f, peak_) = apply_hyperoptimized_compressed(rho, copt, chi_, output_inds=output_inds,
                                        tree_gauge_distance=4, 
                                        progbar=False, 
                                        cutoff=1.e-12
                                     )

    for t in rho_:
        t.transpose_(*output_inds)
        rho_d = t.data
        #print(rho_d)
        rho_d = (rho_d + ar.dag(rho_d))*0.5
        #print(rho_d)
        rho_d = ar.do("reshape", rho_d, (2,2))
        t.transpose_(*output_inds)
        t.modify(data=rho_d)

    z = rho_o(rho_, to_backend, pauli_=pauli_, z_inds=z_inds)
    return z 


def rho_o_appr(rho, opt, to_backend, Lx, Ly, z_inds=["k2,2", "k2,3"], chi_=40, mode="mps", sequence = ["ymin", "ymax"]):
    Z = to_backend(qu.pauli("Z"))

    dic = { i:i.replace("k", "b")  for i in z_inds}
    rho_ = rho.reindex(dic, inplace=False)
    rho_.fuse_multibonds_()
    rhoz=qtn.tensor_network_gate_inds(rho, Z, [z_inds[0]], contract=True,  inplace=False)
    rhozz=qtn.tensor_network_gate_inds(rhoz, Z, [z_inds[1]], contract=True,  inplace=False)
    
    rhoz.reindex_(dic)
    rhoz.fuse_multibonds_()
    
    rhozz.reindex_(dic)
    rhozz.fuse_multibonds_()

    rho_.view_as_(qtn.tensor_2d.TensorNetwork2D, Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
    )
    rhoz.view_as_(qtn.tensor_2d.TensorNetwork2D, Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
    
    )
    rhozz.view_as_(qtn.tensor_2d.TensorNetwork2D, Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
    )



    zz = rhozz.contract_boundary(max_bond=chi_, mode=mode,  
                                         progbar=True,
                                        final_contract_opts={"optimize": opt}, 
                                       cutoff=1e-14,
                                       layer_tags=['KET', 'BRA'],
                                       max_separation=0,
                                        sequence = sequence,
                                       #equalize_norms = equalize_norms,
    
                               )
    z = rhoz.contract_boundary(max_bond=chi_, mode=mode,  
                                progbar=True,
                                        final_contract_opts={"optimize": opt}, 
                                       cutoff=1e-14,
                                       layer_tags=['KET', 'BRA'],
                                       max_separation=0,
                                       sequence = sequence,
    
                               )
    
    norm = rho_.contract_boundary(max_bond=chi_, mode=mode,  
                                progbar=True,
                                        final_contract_opts={"optimize": opt}, 
                                       cutoff=1e-14,
                                       layer_tags=['KET', 'BRA'],
                                       max_separation=0,
                                       sequence = sequence,
    
                               )

    z = complex(z/norm)
    zz = complex(zz/norm)
    
    return z, zz



def evolve_peps( peps, pepo_l, x_pepo, bond_dim,  
                 prgbar = True, 
                 cutoff = 1.0e-12,
                 chi_sample = None,
                 opt = None,
                 prog_compress=True,
                 mode= "mps",
                 max_separation=1,
                 method = "L1BP",
                 max_iterations = 256,
                 tol_final = 1.e-6,
                 damping=0.02,    
                 tol=1.e-6,
                 backend = None,
                 site_2d = None,
                 label = None,
                 chi_bmps = None,
                 normalize=False,
                 num_workers = 4,
                 samples_per_worker = 10,
                 for_each_repeat=1,
                 sample_ = False,
                 method_s = "mps",
                 store_state = True,
                 compress_ = "bp",
                 iter_dmrg = 4,
                ):
    

    if num_workers:
        ray.init(num_cpus=num_workers)


    print("bond_dim", bond_dim)
    print("chi_sample", chi_sample, "chi_bmps", chi_bmps, "mode", mode)
    print("sample_", sample_)
    x_bmps = 0
    norm_peps = 0
    x_isample = 0
    error = 0
    x_bp_normalized = 0

    peps = peps.copy()
    backend, opt, opt_ = req_backend()
    to_backend = get_to_backend(backend) #"numpy-single"
    Lx = peps.Lx
    Ly = peps.Ly
    L = Lx*Ly
    N_l = []
    O_l = []
    X_l = []
    Xbp_l = []
    D_kl_ = []
    Xbmps_l_ = []
    Xbp_second = []
    Xbp_first = []
    Xbp_second_appr = []
    disprob_ = []
    normbmps_l = []
    Zcheck = 0
    infidel_l = []
    infidelO_l = []
    mps_dic = { f"k{j*Lx + i}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}

    Error_l = []
    inds_b = { f"b{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}
    inds_k = { f"k{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}
    #x_pepo.apply_to_arrays(to_backend)
    #peps.apply_to_arrays(to_backend)
    pepo_l_ = []
#--------------Fixed pepo-------------------------------
    pepo_l_ = []
    for pepo in pepo_l:
        pepo_local = []
        for pepo_ in pepo:
            #pepo_trans(pepo_)
            #pepo_ = pepo_.H
            #pepo_.apply_to_arrays(to_backend)
            pepo_local.append(pepo_)
        pepo_l_.append(pepo_local)
 
    # initialize messages
    bp = L2BP(peps & pepo_l_[0][0], optimize=opt,  damping=damping, messages_=None)
    bp.run(max_iterations=120, tol=1.e-6, progbar=False)
    messages_=bp.messages    

        
    with tqdm(total=len(pepo_l),  desc="peps:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
        for steps, pepo in enumerate(pepo_l_):
            
            for pepo_ in pepo:
            
                peps.equalize_norms_()
                peps.balance_bonds_()

                if compress_ == "bp":
                    if prog_compress:
                        print(f"-------------l2bp-compress{peps.max_bond()}-------------")
                
                    peps = peps & pepo_    
                      
                    messages_ = compress_l2bp(peps, site_tags=peps.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol_final, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            #messages_ = messages_,
                                            normalize_messages=True,
                                            update='sequential', #'parallel'
                                            )
                    
                    
                    peps.reindex_(inds_b)

                if compress_ == "qr":

                    peps = peps & pepo_      
                    peps.flatten(fuse_multibonds=True, inplace=True)
                    if prog_compress:
                        print(f"-------------qr-compress{peps.max_bond()}-------------")

                    # peps.compress_all_(max_bond=bond_dim, 
                    #                    canonize_distance=8, canonize=True,
                    #                    #**{"reduced":True}
                    #                    )

                    peps = peps_compress_reza(peps, opt, bond_dim, 
                                              max_iterations=max_iterations, tol = tol,
                                              progbar=False)

                    peps.reindex_(inds_b)

                if compress_ == "dmrg":
                    peps = peps_normalize(peps, opt, chi=chi_bmps)
                    peps_fix = peps & pepo_      
                    peps_fix.flatten(fuse_multibonds=True, inplace=True)
                    peps = peps_fix.compress_all(inplace=False, **{"max_bond":bond_dim, "canonize_distance":4, "cutoff":1e-14})
                    peps.reindex_(inds_b)
                    peps_fix.reindex_(inds_b)
                    if prog_compress:
                        print(f"-------------dmrg-compress{peps.max_bond()}-------------")
                    
                    _ , infidel = dis_peps(peps, peps_fix, opt, 
                                        chi=chi_bmps, 
                                        )
                    infidel_l.append(infidel)
                    print("infidelqr", infidel)
                    qu.save_to_disk(infidel_l, f"Store/info_peps/infidelqr_L{L}bnd{bond_dim}")
                    infidel_ = infidel * 1.
                    #if peps_fix.max_bond() > bond_dim:
                    peps, infidel_ = peps_fit(peps, peps_fix, chi_bmps, opt,
                                                iter=iter_dmrg,
                                                )
                    print("infideldmrg", infidel)
                    infidelO_l.append(infidel_)
                    qu.save_to_disk(infidelO_l, f"Store/info_peps/infidel_L{L}bnd{bond_dim}")

                peps.view_as_(
                            qtn.tensor_2d.PEPS,
                            Lx=Lx, 
                            Ly=Ly,
                            site_tag_id='I{},{}',
                            x_tag_id='X{}',
                            y_tag_id='Y{}',
                            site_ind_id='k{},{}',
                            )

            # if prog_compress:
            #     print("-------------l1bp-<X>-------------")



    #---------------------------Reza------------------------
            if prog_compress:
                print("-------------l2bp-<N>-------------")
                #print("memmory", xyz.report_memory())
                #print("memmory-GPU", xyz.report_memory_gpu())

            
            if store_state:
                qu.save_to_disk(peps, f"Store/info_peps/state_peps/peps_bnd{bond_dim}_{steps}")

            
            bp = L2BP(peps, optimize=opt,  damping=damping)
            bp.run(max_iterations=max_iterations, tol=tol, progbar=prog_compress)
            est_norm = complex(bp.contract())
            N_l.append(abs(complex(est_norm)))
            qu.save_to_disk(N_l, f"Store/info_peps/normbp_{label}_L{L}bnd{bond_dim}")


            site_t = [(2,2), (2,3)]  
            rho = rho_bp(bp, site_t, inds_b, contract = True, absorb_guage=True, fuse = True)
            Zcheck, ZZcheck = rho_o(rho, to_backend)
            print("xbp", Zcheck.real, ZZcheck.real)
            Xbp_l.append(  (Zcheck.real, ZZcheck.real)  )
            qu.save_to_disk(Xbp_l, f"Store/info_peps/xbp_{label}_L{L}bnd{bond_dim}")
            # #print("--------------ext: width~3---------------")
            site_t = [(lx,ly) for lx in range(1,4,1) for ly in range(1,4,1)]
            rho=rho_bp(bp, site_t, inds_b, contract = False, absorb_guage=True, fuse = True)
            Zcheck, ZZcheck = rho_o_hyper(rho, to_backend, chi_=66)
            print("hyper", Zcheck.real, ZZcheck.real)
            Xbp_second_appr.append(  (Zcheck.real, ZZcheck.real)  )
            qu.save_to_disk(Xbp_second_appr, f"Store/info_peps/xbpsa_{label}_L{L}bnd{bond_dim}")

            # #print("--------------ext: width~3---------------")
            # site_t = [(lx,ly) for lx in range(1,4,1) for ly in range(0,Ly,1)]
            # rho=rho_bp(bp, site_t, inds_b, contract = False, absorb_guage=True, fuse = True)
            # Zcheck, ZZcheck = rho_o_hyper(rho, to_backend, chi_=66)
            # print("hyper", Zcheck.real, ZZcheck.real)
            # Xbp_first.append(  (Zcheck.real, ZZcheck.real)  )
            # qu.save_to_disk(Xbp_first, f"Store/info_peps/xbpsaP_{label}_L{L}bnd{bond_dim}")


            if "boundray" in method:
                if prog_compress:
                    print(f"-------------<O>: boundray_chi:{chi_bmps}-------------")
                Xbmps_l_ = []
                for x_pepo_ in x_pepo:
                    Zcheck, norm_peps = pepo_cal(peps, x_pepo_, 
                                            chi_bmps, opt,  
                                            max_separation=max_separation, 
                                            mode = mode,       #'mps',"full-bond" 
                                            progbar = prog_compress, 
                                            normalize=True,    
                                            Falt = False, 
                                            )
                    norm_peps = abs(complex(norm_peps))
                    Xbmps_l_.append(complex(Zcheck).real)
                print("norm_peps", norm_peps)
                print("z,z2,z4", tuple(Xbmps_l_))
                Xbmps_l.append( tuple(Xbmps_l_) )
                normbmps_l.append(norm_peps)
                qu.save_to_disk(Xbmps_l, f"Store/info_peps/x_{label}_L{L}bnd{bond_dim}")
                qu.save_to_disk(normbmps_l, f"Store/info_peps/norm_{label}_L{L}bnd{bond_dim}")

            

            # if normalize:
            #     peps = peps * (est_norm**-0.5)
            #     peps.balance_bonds_()


            if sample_:
                info_sample = peps_sample(peps, steps,bond_dim,
                                          num_workers = num_workers,
                                          samples_per_worker=samples_per_worker,
                                          chi=chi_sample, 
                                          where_l= site_2d,  
                                          backend=backend, 
                                          method = method_s,
                                        )
                x_isample, x2_isample, disprob, D_kl = info_sample
                print(x_isample, x2_isample)
                X_l.append( (x_isample, x2_isample) )
                disprob_.append(disprob)
                D_kl_.append(D_kl)
                #Error_l.append(error)
                qu.save_to_disk(X_l, f"Store/info_peps/xsample_{label}_bnd{bond_dim}")
                qu.save_to_disk(disprob_, f"Store/info_peps/normpeps_{label}_bnd{bond_dim}")
                qu.save_to_disk(D_kl_, f"Store/info_peps/dkl_{label}_bnd{bond_dim}")

                #qu.save_to_disk(Error_l, f"Store/info_peps/error_chi{chi_sample}_sample{t_sample}_{label}_bnd{bond_dim}")
            
            # if peps_dire and bond_dim <= 12:
            #     x_bmps, norm_bmps = peps_x(peps, site_2d, O_label, chi_bmps, opt, backend=backend)
            #     Xbmps_l.append( x_bmps )
            #     normbmps_l.append(norm_bmps)
            #     qu.save_to_disk(Xbmps_l, f"Store/info_peps/xbmps_chi{chi_bmps}_{label}_L{L}bnd{bond_dim}")
            #     qu.save_to_disk(normbmps_l, f"Store/info_peps/nbmps_chi{chi_bmps}_{label}_L{L}bnd{bond_dim}")


            
            
            pbar.set_postfix({"depth": steps, 
                              "norm_b": complex(est_norm).real,
                              "bnd":peps.max_bond(),
                              })
            pbar.refresh()
            pbar.update(1)

    #---------------------------------------------------------------------


    return peps, N_l, O_l, X_l, Xbp_l
    


def normalize_bp(tn, bp, loop_pair, site_tags):
    res = bp_excited_loops(tn, bp, loop_pair, chi=100, contract_=False, prgbar=False)
    tn_l = res["tn_l"]
    tags = list(site_tags)
    tn_ = tn_l[0]
    norm_l = []
    for tag in tags:
        tn_local_ = tn_.select(tag, which="all")
        tn_local = tn.select(tag, which="all")
    
        norm_ = tn_local_.contract(all)
        norm_l.append(np.log10(complex(norm_)))
        tn_local /= norm_ 
        #tn_local.draw()


    tn.exponent = sum(norm_l)



def simulate_peps( peps, pepo_l, bond_dim,  
                 prgbar = True, 
                 cutoff = 1.0e-12,
                 chi_sample = None,
                 opt = None,
                 prog_compress=False,
                 mode= "mps",
                 max_separation=1,
                 method = "L1BP",
                 max_iterations = 256,
                 tol_final = 1.e-6,
                 damping=0.01,    
                 tol=1.e-6,
                 backend = None,
                 site_2d = None,
                 label = None,
                 chi_bmps = None,
                 normalize=False,
                 num_workers = None,
                 samples_per_worker = 10,
                 for_each_repeat=1,
                 sample_ = False,
                 method_s = "mps",
                 store_state = True,
                 compress_ = "bp",
                 iter_dmrg = 0,
                 cluster = None,
                 obs = None,
                 gate_l = [],
                 where_l = [],
                 norm_cal = True,
                 trans=False,
                 canonize_distance = 2,
                 update = 'sequential',
                                 ):
    

    if num_workers:
        ray.init(num_cpus=num_workers)

    x_bmps = 0
    norm_peps = 0
    x_isample = 0
    error = 0
    x_bp_normalized = 0

    peps = peps.copy()
    backend, opt, opt_ = req_backend()
    to_backend = get_to_backend(backend) #"numpy-single"
    Lx = peps.Lx
    Ly = peps.Ly
    L = Lx*Ly
    N_l = []
    O_l = []
    X_l = []
    Xbp_l = []
    D_kl_ = []
    Xbmps_l_ = []
    Xbp_second = []
    Xbp_first = []
    Xbp_second_appr = []
    disprob_ = []
    normbmps_l = []
    Zcheck = 0
    infidel_l = []
    infidelO_l = []
    F_l = []
    val = 0
    mps_dic = { f"k{j*Lx + i}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}

    Error_l = []
    inds_b = { f"b{i},{j}":f"k{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}
    inds_k = { f"k{i},{j}":f"b{i},{j}"  for i,j in itertools.product(range(Lx), range(Ly))}
    #x_pepo.apply_to_arrays(to_backend)
    #peps.apply_to_arrays(to_backend)
    pepo_l_ = []
#--------------Fixed pepo-------------------------------
    pepo_l_ = []
    for pepo in pepo_l:
        pepo_local = []
        for pepo_ in pepo:
            pepo_ = pepo_.copy()
            if trans:
                pepo_trans(pepo_)
            #pepo_ = pepo_.H
            #pepo_.apply_to_arrays(to_backend)
            pepo_local.append(pepo_)
        pepo_l_.append(pepo_local)
 
    # initialize messages
    bp = L2BP(peps & pepo_l_[0][0], optimize=opt,  damping=damping, messages_=None)
    bp.run(max_iterations=120, tol=1.e-6, progbar=False)
    messages_=bp.messages    

    peps_l = []
    with tqdm(total=len(pepo_l),  desc="peps:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
        for steps, pepo in enumerate(pepo_l_):
            peps_l.append(peps)
            for pepo_ in pepo:
            
                peps.equalize_norms_()
                peps.balance_bonds_()
                peps = peps & pepo_    
                      
                messages_ = compress_l2bp(peps, site_tags=peps.site_tags,
                                            cutoff_mode='rsum2',
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            tol=tol_final,
                                            max_iterations=max_iterations,
                                            optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            messages_ = messages_,
                                            #normalize_messages=True,
                                            update=update, #'parallel'
                                            )
                    
                    
                peps.reindex_(inds_b)

                peps.view_as_(
                            qtn.tensor_2d.PEPS,
                            Lx=Lx, 
                            Ly=Ly,
                            site_tag_id='I{},{}',
                            x_tag_id='X{}',
                            y_tag_id='Y{}',
                            site_ind_id='k{},{}',
                            )


    #---------------------------Reza------------------------
            if prog_compress:
                print("-------------l2bp-<N>-------------")
                #print("memmory", xyz.report_memory())
                #print("memmory-GPU", xyz.report_memory_gpu())

            

            if norm_cal:
                    bp = L2BP(peps, optimize=opt,  damping=damping)
                    bp.run(max_iterations=max_iterations, tol=tol, progbar=prog_compress)
                    norm_1 = complex(bp.contract())
                    val = abs(complex(norm_1/1.)**2)

                    if val>1.2:
                        val = 0.5
                    elif val<0:
                        val = 0.5
            
            N_l.append(val)
            f_ = np.prod(N_l)
            f_ = abs(complex(f_))
            F_l.append(f_)
                
            if site_2d:
                x_, y_ = site_2d

                site_t = [(x_, y_)]  
                output_inds=[f"k{x_},{y_}",f"b{x_},{y_}"]
                rho = rho_bp(bp, site_t, inds_b,output_inds=output_inds, contract = True, absorb_guage=True, fuse = True)
                Zcheck= rho_o(rho, to_backend, pauli_=obs, z_inds=[f"k{x_},{y_}"])
            
                if Zcheck.real>=-1. and Zcheck.real<=1: 
                    Xbp_l.append(  Zcheck.real  )
                else:
                    Xbp_l.append(  0  )
                
                

                res_l = []
                for cluster_ in cluster:
                    rho = rho_bp(bp, cluster_, inds_b, output_inds=output_inds, contract = False, absorb_guage=True, fuse = False)
                    
                    Zcheck_ = rho_o_hyper(rho, to_backend, pauli_=obs, z_inds=[f"k{x_},{y_}"], output_inds=output_inds, chi_=66)
                    if Zcheck_.real<-1. or Zcheck_.real>1: 
                        Zcheck_ = 0

                    res_l.append(Zcheck_.real)
                    

                if res_l:
                    Xbp_second_appr.append( sum(res_l)/len(res_l) )



                        
            
            pbar.set_postfix({ "obs":complex(Zcheck).real,
                              "error":1 - F_l[-1],
                              "bnd":peps.max_bond(),
                              })
            pbar.refresh()
            pbar.update(1)

    #---------------------------------------------------------------------
    res_info = {"peps_l":peps_l, "fidel":F_l, "obs":Xbp_l, "obs_cluster":Xbp_second_appr}

    return res_info


















#@profile
def evolve_pepo(x_pepo, pepo_l, bond_dim,  
                 prgbar = True, 
                 cutoff = 1.0e-10,
                 peps = None,
                 chi = None,
                 copt = None,
                 prog_compress=True,
                 tree_gauge_distance = 4,
                 mode= "mps",
                 max_separation=0,
                 method = "L1BP",
                 max_iterations = 256,
                 tol = 5e-7,
                 damping=0.01,                      
                 O_label = None,
                 site = None,
                 label = None,
                 eff_ = False,
                 compress_ = "bp",
                ):
     #"numpy-single"
    backend, opt, opt_ = req_backend()

    Lx = x_pepo.Lx
    Ly = x_pepo.Ly
    L = Lx*Ly
    count = 0
    z_appro = 0
    O_l = []
    O_l_bp = []
    N_l = []
    #x_pepo.apply_to_arrays(to_backend)

        

    with tqdm(total=len(pepo_l),  desc="pepo:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
        for count, pepo in enumerate(pepo_l):
            x_pepo.equalize_norms_()
            for pepo_ in pepo:
                if count==0:
                    
                    x_pepo = apply_pepo_sandwich_flat(x_pepo, pepo_)
                    
                    
                    if compress_ == "bp":
                        if prog_compress:
                            print(f"-------------l2bp-compress: bnd{x_pepo.max_bond()}-------------")

                    
                        compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )
                    elif compress_ == "qr":
                        x_pepo.flatten(fuse_multibonds=True, inplace=True)
                        if prog_compress:
                            print(f"-------------qr-compress: bnd{x_pepo.max_bond()}-------------")
                        x_pepo.compress_all_(max_bond=bond_dim, 
                                       canonize_distance=4, canonize=True,
                                       #**{"reduced":True}
                                       )


                elif eff_:
                    x_pepo = apply_pepo_sandwich_1(x_pepo, pepo_)                
                    
                    
                    if compress_ == "bp":
                        if prog_compress:
                            print(f"-------------l2bp-compress-1: bnd{x_pepo.max_bond()}-------------")

                        compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )
                    
                    
                    elif compress_ == "qr":
                        x_pepo.flatten(fuse_multibonds=True, inplace=True)
                        if prog_compress:
                            print(f"-------------qr-compress-1: bnd{x_pepo.max_bond()}-------------")
                        x_pepo.compress_all_(max_bond=bond_dim, 
                                       canonize_distance=4, canonize=True,
                                       #**{"reduced":True}
                                       )
                    
                    
                    x_pepo = apply_pepo_sandwich_2(x_pepo, pepo_)                
                    if compress_ == "bp":
                        if prog_compress:
                            print(f"-------------l2bp-compress-2: bnd{x_pepo.max_bond()}-------------")

                        compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )

                    elif compress_ == "qr":
                        x_pepo.flatten(fuse_multibonds=True, inplace=True)
                        if prog_compress:
                            print(f"-------------qr-compress-2: bnd{x_pepo.max_bond()}-------------")
                        x_pepo.compress_all_(max_bond=bond_dim, 
                                       canonize_distance=4, canonize=True,
                                       #**{"reduced":True}
                                       )

                
                else:
                    x_pepo = apply_pepo_sandwich(x_pepo, pepo_)                
                    #x_pepo.balance_bonds_()
                    if compress_ == "bp":
                        if prog_compress:
                            print(f"-------------l2bp-compress: bnd{x_pepo.max_bond()}-------------")

                        compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )
                    elif compress_ == "qr":
                        x_pepo.flatten(fuse_multibonds=True, inplace=True)
                        if prog_compress:
                            print(f"-------------qr-compress: bnd{x_pepo.max_bond()}-------------")
                        x_pepo.compress_all_(max_bond=bond_dim, 
                                       canonize_distance=4, canonize=True,
                                       #**{"reduced":True}
                                       )

            
            
            x_pepo.view_as_(
                    qtn.tensor_2d.PEPO,
                    Lx=Lx, 
                    Ly=Ly,
                    site_tag_id='I{},{}',
                    x_tag_id='X{}',
                    y_tag_id='Y{}',
                    upper_ind_id='k{},{}',
                    lower_ind_id='b{},{}',
            )

            if prog_compress:
                print("-------------l2bp-<N>-------------")

            bp = L2BP(x_pepo, optimize=opt, site_tags=x_pepo.site_tags, damping=damping)
            bp.run(tol=tol, max_iterations=max_iterations, progbar=True)
            mantissa, norm_exponent = bp.contract(strip_exponent=True)
            
            
            
            est_norm = float(10 ** ((norm_exponent - (len(bp.local_tns) * log10(2))) / 2))
            est_norm = (abs(complex(mantissa))**-.5) * est_norm
            N_l.append( abs(est_norm) )
            qu.save_to_disk(N_l, f"Store/info_pepo/norm_{label}_L{L}bnd{bond_dim}")

            

            if method == "hypercompress":
                if prog_compress:
                    print(f"-------<O>: hypercompress_chi:{chi}-------------")

                z_appro = pepo_flat_hypercompress(x_pepo, peps, 
                                                    chi, 
                                                    copt,
                                                    progbar=prog_compress, 
                                                    tree_gauge_distance=tree_gauge_distance,
                                                    opt = opt,
                                                    max_bond = None,
                                                    canonize_distance = 4,

                                                )
            elif method == "boundray":
                if prog_compress:
                    print(f"-------------<O>: boundray_chi:{chi}-------------")

                z_appro = pepo_cal(peps, x_pepo, 
                                    chi, opt,  
                                    max_separation=max_separation, 
                                    mode = mode,       #'mps',"full-bond" 
                                    progbar = prog_compress, 
                                    Falt = True,
                                    normalize = False,

                                )

            #elif method == "L1BP":
            if prog_compress:
                print(f"-------------<O>: L1BP-------------")

            z_appro_bp = pepo_BP(x_pepo, peps, 
                                    progbar=prog_compress, 
                                    opt = opt,
                                    max_bond = None,
                                    )


                    
            O_l_bp.append((complex(z_appro_bp)).real)
            O_l.append((complex(z_appro)).real)
            qu.save_to_disk(O_l, f"Store/info_pepo/x_{label}_L{L}bnd{bond_dim}")
            qu.save_to_disk(O_l_bp, f"Store/info_pepo/xbp_{label}_L{L}bnd{bond_dim}")

            pbar.set_postfix({"depth": len(O_l), 'O': abs(complex(z_appro)),
                                'O_bp': abs(complex(z_appro_bp)), 
                              "norm": abs(complex(est_norm)), "bnd":x_pepo.max_bond()})
            pbar.refresh()
            pbar.update(1)

    
    x_pepo.view_as_(
                qtn.tensor_2d.PEPO,
                Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
                upper_ind_id='k{},{}',
                lower_ind_id='b{},{}',
        )

    return x_pepo, N_l, O_l, O_l_bp



def trace_2d(pepo_):
    pepo = pepo_*1.
    for count, t in enumerate(pepo):
        l = t.inds
        k_ind = [i for i in l if i.startswith("k")]
        b_ind = [i for i in l if i.startswith("b")]
        t.trace(k_ind, b_ind, preserve_tensor=False, inplace=True)
    pepo.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=pepo.Lx,
        Ly=pepo.Ly,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return pepo

def pepo_run(where_l):
    act = False
    for where in where_l:
        if len(where) == 2:
            act = True
            break
    return  act     

def apply_pepo_gate(x_pepo, gate_l, where_l, cutoff=1.e-9):
    for count in range(len(gate_l)):
        info = {}
        G = gate_l[count]
        where = where_l[count]
        print(where)
        x_pepo = gate_f(x_pepo, G, list(where), lable = "upper", 
                            contract='reduce-split', tags=["G"], propagate_tags='sites', inplace=True, info=info, 
                            long_range_use_swaps=True, long_range_path_sequence=('av', 'bh'), **{"cutoff":1.e-9, "absorb":'both'}
                            )
        GH = G.H
        x_pepo = gate_f(x_pepo, GH, list(where), lable = "lower", 
                            contract='reduce-split', tags=["G"], propagate_tags='sites', inplace=True, info=info, 
                            long_range_use_swaps=True, long_range_path_sequence=('av', 'bh'), **{"cutoff":1.e-9, "absorb":'both'}
                            )

    return x_pepo

def evolve_pepo_gate(x_pepo, gate_l, where_l, bond_dim, canonize_distance, depth_r = 2, prgbar = True, cutoff = 1.0e-12):
    Lx = x_pepo.Lx
    Ly = x_pepo.Ly
    L = Lx * Ly
    infidel_ = []
    infidelity = 0
    O_appr = []
    count = 0
    gate_l_= [gate_l[i:i + depth_r] for i in range(0, len(gate_l), depth_r)]
    where_l_= [where_l[i:i + depth_r] for i in range(0, len(where_l), depth_r)]

    with tqdm(total=len(gate_l_),  desc="pepo:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
        for count in range(len(gate_l_)):
            compress_ = pepo_run(where_l_[count])
            x_pepo = apply_pepo_gate(x_pepo, gate_l_[count], where_l_[count], 
                                cutoff=cutoff,
                                )
            if compress_:
                #x_pepo.balance_bonds_()
                x_pepo.compress_all(inplace=True, **{"max_bond":bond_dim, "canonize_distance": canonize_distance, "cutoff":cutoff})

            pbar.refresh()
            pbar.update(1)

    return x_pepo, infidel_



#@profile
def evolve_dpepo(x_pepo, pepo_l, bond_dim,  
                 prgbar = True, 
                 cutoff = 1.0e-12,
                 peps = None,
                 chi = None,
                 opt = None,
                 copt = None,
                 prog_compress=True,
                 tree_gauge_distance = 4,
                 mode= "mps",
                 max_separation=0,
                 method = "L1BP",
                 max_iterations = 256,
                 tol = 1.e-8,
                 backend=None,
                 damping=0.01,                      
                 O_label = None,
                 site = None,
                 label = None,
                 eff_ = False
                ):
     #"numpy-single"

    Lx = x_pepo.Lx
    Ly = x_pepo.Ly
    count = 0
    z_appro = 0
    z_appro_bp = 0
    O_l = []
    O_l_bp = []
    N_l = []
    N_l_ = []
    res_ = []
    #x_pepo.apply_to_arrays(to_backend)

        

    with tqdm(total=len(pepo_l),  desc="dpepo:", leave=True, position=0, 
            colour='MAGENTA', disable = not prgbar) as pbar:
        for count, pepo in enumerate(pepo_l):
            x_pepo.equalize_norms_()
            x_pepo.balance_bonds_()
            for pepo_ in pepo:
                if count==0:
                    x_pepo = apply_pepo_sandwich_flat(x_pepo, pepo_)
                    print(f"-------------l2bp-compress-1: bnd{x_pepo.max_bond()}-------------")
                    compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )

                elif eff_:
                    x_pepo = apply_pepo_sandwich_1(x_pepo, pepo_)                
                    if prog_compress:
                        print(f"-------------l2bp-compress-1: bnd{x_pepo.max_bond()}-------------")
                    compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )
                    x_pepo = apply_pepo_sandwich_2(x_pepo, pepo_)                
                    if prog_compress:
                        print(f"-------------l2bp-compress-2: bnd{x_pepo.max_bond()}-------------")
                    compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )
                
                
                else:
                    x_pepo = apply_pepo_sandwich(x_pepo, pepo_)                
                    #x_pepo.balance_bonds_()
                    if prog_compress:
                        print(f"-------------l2bp-compress: bnd{x_pepo.max_bond()}-------------")

                    compress_l2bp(x_pepo, site_tags=x_pepo.site_tags,
                                            max_bond=bond_dim, 
                                            cutoff=cutoff,
                                            cutoff_mode='rsum2',
                                            max_iterations=max_iterations,
                                            tol=tol, optimize=opt,
                                            progbar=prog_compress,
                                            inplace=True,
                                            damping=damping,
                                            local_convergence=True,
                                            )

            
            
            x_pepo.view_as_(
                    qtn.tensor_2d.PEPO,
                    Lx=Lx, 
                    Ly=Ly,
                    site_tag_id='I{},{}',
                    x_tag_id='X{}',
                    y_tag_id='Y{}',
                    upper_ind_id='k{},{}',
                    lower_ind_id='b{},{}',
            )

            if prog_compress:
                print("-------------l2bp-<N>-------------")


            x_pepo_trace = trace_2d(x_pepo)
            
            bp = L1BP(x_pepo_trace, optimize=opt, site_tags=x_pepo.site_tags, damping=damping)
            bp.run(tol=tol, max_iterations=max_iterations, progbar=True)
            mantissa, norm_exponent = bp.contract(strip_exponent=True)
            
            est_norm = abs(complex(mantissa * 10**norm_exponent))
            N_l.append( abs(est_norm) )
            qu.save_to_disk(N_l, f"Store/info_dpepo/Normbp_{label}_{bond_dim}_{backend}")

            

            if method == "hypercompress":
                if prog_compress:
                    print(f"-------<O>: hypercompress_chi:{chi}-------------")

                z_appro = pepo_flat_hypercompress(x_pepo, peps, 
                                                    chi, 
                                                    copt,
                                                    progbar=prog_compress, 
                                                    tree_gauge_distance=tree_gauge_distance,
                                                    opt = opt,
                                                    max_bond = None,
                                                    canonize_distance = 4,

                                                )
            if  "boundray" in method:
                if prog_compress:
                    print(f"-------------<O>: boundray_chi:{chi}-------------")
                res_ = []
                for peps_ in peps:
                    z_appro, est_norm = dpepo_cal(x_pepo, peps_, 
                                    chi, opt,  
                                    max_separation=max_separation, 
                                    mode = mode,       #'mps',"full-bond" 
                                    progbar = prog_compress, 
                                    Falt = True,
                                    normalize=True,

                                )
                    res_.append((complex(z_appro)).real)
                print(tuple(res_), est_norm)
                O_l.append(tuple(res_))
                qu.save_to_disk(O_l, f"Store/info_dpepo/X_{label}_{bond_dim}_{backend}")

            #elif method == "L1BP":

            if prog_compress:
                print(f"-------------<O>: L1BP-------------")
            if "bp" in method:
                res_ = []
                for peps_ in peps:
                    z_appro_bp = dpepo_BP(x_pepo, peps_, 
                                            progbar=prog_compress, 
                                            opt = opt,
                                            max_bond = None,
                                            )
                    res_.append((complex(z_appro_bp)).real/est_norm)

                O_l_bp.append(tuple(res_))
                qu.save_to_disk(O_l_bp, f"Store/info_dpepo/Xbp_{label}_{bond_dim}_{backend}")

            N_l_.append( abs(est_norm) )
            qu.save_to_disk(N_l_, f"Store/info_dpepo/Norm_{label}_{bond_dim}_{backend}")
                    

            pbar.set_postfix({"depth": len(O_l), 'O': complex(z_appro).real,
                                'O_bp': complex(z_appro_bp).real, 
                              "norm": abs(complex(est_norm)), "bnd":x_pepo.max_bond()})
            pbar.refresh()
            pbar.update(1)

    
    x_pepo.view_as_(
                qtn.tensor_2d.PEPO,
                Lx=Lx, 
                Ly=Ly,
                site_tag_id='I{},{}',
                x_tag_id='X{}',
                y_tag_id='Y{}',
                upper_ind_id='k{},{}',
                lower_ind_id='b{},{}',
        )

    return x_pepo, N_l_, O_l, O_l_bp



















def canon_(peps, tags):
    Lx = peps.Lx                  #127
    Ly = peps.Ly
    peps_g = peps.canonize_around(tags, which='any',
                            max_distance=Lx * Ly,
                            absorb='right',
                            gauge_links = True , #SU
                            link_absorb='both',
                            equalize_norms=False, 
                            inplace=False)
    return peps_g


def peps_qr_exp(peps, site, O_label, opt):
    Lx = peps.Lx                  #127
    Ly = peps.Ly
    L = Lx * Ly
    X = O_label[0]
    x, y = site[0]
    tags = [f"I{x},{j}" for j in range(Ly)]
    peps = canon_(peps, tags)
    tn = peps.select(tags, which='any')
    norm = (tn & tn.H).contract(all, optimize = opt)
    tn_X = tn.gate(X, where=(x,y))
    X_ = (tn_X & tn.H).contract(all, optimize = opt)
    print("local-mps-0", X_ / norm)



    tags = [f"I{x},{y}"] 
    peps = canon_(peps, tags)
    tn = peps.select(tags, which='any')
    norm = (tn & tn.H).contract(all, optimize = opt)
    tn_X = tn.gate(X, where=(x,y))
    X_ = (tn_X & tn.H).contract(all, optimize = opt)
    print("local-0", X_ / norm)
 
    tags = [f"I{x},{y}"] + [f"I{x},{y+1}"] + [f"I{x},{y-1}"] +  [f"I{x+1},{y}"] +   [f"I{x-1},{y}"] 
    peps = canon_(peps, tags)
    tn = peps.select(tags, which='any')
    norm = (tn & tn.H).contract(all, optimize = opt)
    tn_X = tn.gate(X, where=(x,y))
    X_ = (tn_X & tn.H).contract(all, optimize = opt)
    print("local-1", X_ / norm)



    tags = [f"I{x},{j}" for j in range(Ly)]
    tags += [f"I{i},{y}" for i in range(Lx)]
    peps = canon_(peps, tags)
    tn = peps.select(tags, which='any')
    norm = (tn & tn.H).contract(all, optimize = opt)
    tn_X = tn.gate(X, where=(x,y))
    X_ = (tn_X & tn.H).contract(all, optimize = opt)
    print("local-mps-1", X_ / norm)


    # tags = [f"I{x},{y}"] + [f"I{x},{y+1}"] + [f"I{x},{y-1}"] +  [f"I{x+1},{y}"] +   [f"I{x-1},{y}"] 
    # tags += [f"I{x+1},{y+1}"] + [f"I{x-1},{y-1}"] +  [f"I{x+1},{y-1}"] +  [f"I{x-1},{y+1}"]
    # peps = canon_(peps, tags)
    # tn = peps.select(tags, which='any')
    # norm = (tn & tn.H).contract(all, optimize = opt)
    # tn_X = tn.gate(X, where=(x,y))
    # X_ = (tn_X & tn.H).contract(all, optimize = opt)
    # print("local-2", X_ / norm)

    # tags = [f"I{x},{j}" for j in range(Ly)] + [f"I{x+1},{j}" for j in range(Ly)] +[f"I{x-1},{j}" for j in range(Ly)] 
    # peps=canon_(peps, tags)
    # tn = peps.select(tags, which='any')
    # norm = (tn & tn.H).contract(all, optimize = opt)
    # tn_X = tn.gate(X, where=(x,y))
    # X_ = (tn_X & tn.H).contract(all, optimize = opt)
    # print("local-mps-1", X_ / norm)

    return X_ / norm


def peps_env(peps, site, O_label, opt, x_0 = 2, x_1 = 4, 
             max_bond = 40, max_bond_ctg = 40,
             mode = 'full-bond',
             ):
    copt = ctg.ReusableHyperCompressedOptimizer(
        max_bond_ctg,
        max_repeats=2**9,
        methods=  ('greedy-compressed', 'greedy-span', 'kahypar-agglom'),
        minimize='combo-compressed', 
        progbar=True,
        parallel="ray",
        # # save paths to disk:
        # directory=True  
        #methods = 'labels',
        #optlib="nevergrad",         # 'nevergrad', 'baytune', 'chocolate','random'
        #methods = 'labels',
        directory="cash/",

    )


    Lx = peps.Lx                  #127
    Ly = peps.Ly
    L = Lx * Ly
    X = O_label[0]
    x, y = site[0]

    
    print("mode", mode)
    print("chi", max_bond)
    print("max_bond_ctg", max_bond_ctg) 
    print("x_0, x_1", x_0, x_1)

    tn_left, tn_right, peps_middle = peps_mps_x(peps, opt, x_0=x_0, x_1=x_1, max_bond=max_bond, mode=mode)
    peps_ket = peps_middle.select(["KET"], which="any")
    peps_bra = peps_middle.select(["BRA"], which="any")
    tn = (peps_middle & tn_left & tn_right)
    norm_X = apply_hyperoptimized_compressed(tn, copt, max_bond_ctg, output_inds=None, tree_gauge_distance=2, progbar=True, cutoff=1.e-12, equalize_norms=1.0)
    main_, exp_=(norm_X.contract(), norm_X.exponent)
    x, y = site[0]
    peps_ketX = apply_2d(peps_ket, O_label[0], [(x,y),], tags="G")
    tn = (peps_ketX & peps_bra& tn_left & tn_right)
    res_X = apply_hyperoptimized_compressed(tn, copt, max_bond_ctg, output_inds=None, tree_gauge_distance=2, progbar=True, cutoff=1.e-12, equalize_norms=1.0)
    main, exp=(res_X.contract(), res_X.exponent)
    result = (main/main_) * 10**(exp - exp_)
    print("result", result)
    tid = tn._get_tids_from_tags(["G"], which='all')
    tn_ = tn.copy()
    X = tn_.pop_tensor(*tid)
    output_inds = X.inds
    rho = apply_hyperoptimized_compressed(tn_, copt, max_bond_ctg, output_inds=output_inds, tree_gauge_distance=2, progbar=True, cutoff=1.e-12, equalize_norms=1.0)

    for t in rho:
        A = t.data
        AH = qu.dag(t.data)
        A_ = (A + AH) * 0.5

    rho = qtn.Tensor(data=A_, inds=rho["KET"].inds, tags=None)

    X = (rho & X).contract(all, optimize=opt)
    norm_ = rho.trace(output_inds[0], output_inds[1], preserve_tensor=False, inplace=False)
    result = X / norm_
    print("result", result)
    return result



#mix_mpo

# mpo_l_lightcone, circ = quf.trotter_gates_IBM_lightcone(site, Lx, Ly, Lz, delta_t, J, h, 
#                                       cycle = "open", # periodic
#                                       dtype=dtype,
#                                       cutoff = cutoff,
#                                       max_bond_mpo = max_bond_mpo,
#                                       style = "left",
#                                       depth_=depth_
#                             )

# qu.save_to_disk(gate_dic, "gate_dic")
# gate_dic = qu.load_from_disk("gate_dic")
# qu.save_to_disk(mpo_list, "mpo_list")
# mpo_list = qu.load_from_disk("mpo_list")
# qu.save_to_disk(gate_dic_lightcone, "gate_dic_lightcone")
# gate_dic_lightcone = qu.load_from_disk("gate_dic_lightcone")


def apply_hyperoptimized_compressed(tn, copt, max_bond, output_inds=None, tree_gauge_distance=4, progbar=False, cutoff=1.e-12, equalize_norms=1.0, f_max = 14 , peak_max=33):

    tn_ = tn.copy()
    tn_.full_simplify_(seq='R', split_method='svd', inplace=True)

    tree = tn_.contraction_tree(copt)
    flops = tree.contraction_cost()
    peak = tree.peak_size(log=2)


    

    contract_pass = True
    
    if flops >1.e-8 :
        if np.log10(flops) > f_max or peak >peak_max:
            contract_pass = False 
        
    if not contract_pass:
        print("warning: over-resources", np.log10(flops), peak)
        return None, (1.e-5, 1.e-5)
    
    if contract_pass:
        tn_.contract_compressed_(
            optimize=tree,
            output_inds=output_inds,
            max_bond=max_bond,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
        )
    return tn_, (flops, peak)



def gate_to_mpo(gate_, where_, L, SQ="absorb", dtype="complex128", cutoff = 1.0e-12, max_bond_mpo=256, style="left"):
    mpo_l = []
    if SQ == "absorb":
        for count, where in enumerate(where_):
            mpo = mpo_from_gate(gate_[count], where, L,  dtype = dtype, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
            if virtual_bond_max(mpo) == 1 and count > 0:
                #mpo_f = mpo_l_lightcone[len(mpo_l_lightcone) - 1].apply(mpo)
                mpo_f = mpo.apply(mpo_l[len(mpo_l) - 1])
                mpo_f.compress(style, max_bond=max_bond_mpo, cutoff=cutoff)
                mpo_l[len(mpo_l) - 1] = mpo_f 
            else:
                mpo.compress(style, max_bond=max_bond_mpo, cutoff=cutoff)
                mpo_l.append(mpo)
    
    else:
        for count, where in tqdm(enumerate(where_)):
            mpo = mpo_from_gate(gate_[count], where, L,  dtype = dtype, cutoff=cutoff, max_bond_mpo=max_bond_mpo, style= style)
            if virtual_bond_max(mpo) > 1:
                mpo.compress(style, max_bond=max_bond_mpo, cutoff=cutoff)
            mpo_l.append(mpo)
        
    return mpo_l

def block_mps(l, L_mps, dtype="complex128", cycle=False):
    tensors = [qtn.Tensor() for _ in range(L_mps)]
    count = 0
    tags = []
    for i in range(L_mps):
        for i_ in range(l[i]):
            tensors[i].new_ind(f'k{count}', size=2)
            tensors[i].add_tag(f"I{i}")
            tags.append(f"I{i}")
            count += 1
    if cycle:
        for i in range(L_mps):
            tensors[i].new_bond(tensors[(i + 1)%L_mps ], size=7)
    else:
        for i in range(L_mps-1):
            tensors[i].new_bond(tensors[(i + 1) ], size=7)
    
    
    mps = qtn.TensorNetwork(tensors)
    mps.astype_(dtype)
    mps.randomize_(dtype=dtype, seed=2, inplace=True)
    return mps, tags


def btn_to_mps(mps_, L_mps, cycle=False, inplace = False):
    mps = mps_ if inplace else mps_.copy()
    inds_unfuse = []
    for i in range(L_mps):
        inds_l = []
        for indx in mps[f"I{i}"].inds:
            if indx.startswith('k'):
              inds_l.append(indx)
    
        mps[f"I{i}"].fuse({f"k{i}":inds_l}, inplace=True)
        inds_unfuse.append(inds_l)
    mps.view_as_(qtn.MatrixProductState, L = L_mps, site_tag_id='I{}',site_ind_id='k{}',cyclic=cycle)
    return mps, inds_unfuse


def mps_to_btn(mps_, inds_unfuse, inplace = False):
    mps = mps_ if inplace else mps_.copy()
    for i in range(mps.L):
        dim = (2,)*len(inds_unfuse[i])
        mps[f"I{i}"].unfuse({f"k{i}":tuple(inds_unfuse[i])}, shape_map={f"k{i}":dim} , inplace=True)
    return mps


def mps_to_bmps(p_, l, opt, cycle=False, inplace = False):
    L_mps = len(l)

    inds_unfuse= []
    count_ = 0
    for count, elemnts in enumerate(l):
        local =[]
        for _ in range(elemnts):
            local.append(f"k{count_}")
            count_+=1
        inds_unfuse.append(local)
    #print(inds_unfuse)
    p = p_ if inplace else p_.copy()
    count_ = 0
    t_l = []
    for count, size in enumerate(l):
        tags = []
        for _ in range(l[count]):
            tags.append(f"I{count_}")
            count_ += 1
        t = p.select(tags, which="any").contract(all, optimize=opt)
        t.modify(tags=f"I{count}")
        t.fuse({f"k{count}":tuple(inds_unfuse[count])}, inplace=True)
        t_l.append(t)
    mps = qtn.TensorNetwork(t_l)
    mps.view_as_(qtn.MatrixProductState, L = L_mps, site_tag_id='I{}',site_ind_id='k{}', cyclic=cycle)
    return mps, inds_unfuse


def mpo_to_bmpo(p_, l, inds_unfuse, L_mps, opt, inplace = False):
    p = p_ if inplace else p_.copy()
    count_ = 0
    t_l = []
    for count, size in enumerate(l):
        tags = []
        for _ in range(l[count]):
            tags.append(f"I{count_}")
            count_ += 1
        t = p.select(tags, which="any").contract(all, optimize=opt)
        t.modify(tags=f"I{count}")
        inds = list(inds_unfuse[count])
        t.fuse({f"k{count}":tuple(inds)}, inplace=True)
        inds_ = [i.replace("k", "b") for i in inds]
        t.fuse({f"b{count}":tuple(inds_)}, inplace=True)
        t_l.append(t)
    mps = qtn.TensorNetwork(t_l)
    mps.view_as_(qtn.MatrixProductOperator, L = L_mps, site_tag_id='I{}',upper_ind_id='k{}',lower_ind_id='b{}',cyclic=False)
    return mps


def mpo_to_btn(mps_, inds_unfuse, inplace = False):
    mps = mps_ if inplace else mps_.copy()
    for i in range(mps.L):
        inds = list(inds_unfuse[i])
        dim = (2,)*len(inds)
        mps[f"I{i}"].unfuse({f"k{i}":tuple(inds)}, 
                            shape_map={f"k{i}":dim} , inplace=True)
        
        inds_ = [i.replace("k", "b") for i in inds]
        mps[f"I{i}"].unfuse({f"b{i}":tuple(inds_)}, 
                            shape_map={f"b{i}":dim} , inplace=True)
    return mps

def btn_to_mpo(mps_, L_mps, cycle=False, inplace = False):
    mps = mps_ if inplace else mps_.copy()
    inds_unfuse = []
    for i in range(L_mps):
        inds_l = []
        for indx in mps[f"I{i}"].inds:
            if indx.startswith('k'):
              inds_l.append(indx)

        inds_l_ = [i.replace("k", "b") for i in inds_l]
    
        mps[f"I{i}"].fuse({f"k{i}":inds_l}, inplace=True)
        mps[f"I{i}"].fuse({f"b{i}":inds_l_}, inplace=True)
        inds_unfuse.append(inds_l)
    mps.view_as_(qtn.MatrixProductState, L = L_mps, site_tag_id='I{}',site_ind_id='k{}',cyclic=cycle)
    return mps, inds_unfuse


def Init_bmps(l, L_mps, list_basis, opt, dtype="complex64", cycle=False, theta = 0):
    #btn, tags = block_mps(l, L_mps, dtype=dtype, cycle=cycle)
    #mps, inds_unfuse = btn_to_mps(btn, L_mps, cycle=cycle)
    #btn=mps_to_btn(mps, inds_unfuse, L_mps)

    
    p_0_ = qtn.MPS_computational_state([0]*sum(l))

    
    
    for t in p_0_:
        vec = np.array([math.cos(theta), math.sin(theta)]) 
        shape = t.shape
        t.modify(data = vec.reshape(shape))


    mps, inds_unfuse = mps_to_bmps(p_0_, l, opt,cycle=cycle)
    return mps, inds_unfuse


def iregular_bnd(p, bnd_l, rand_strength=0.20):
    for count, bndsize in enumerate(bnd_l):
        inds = p.bond(count, count+1)
        bndsize_ = p.bond_size(count, count+1)
        if bndsize > bndsize_: 
            p.expand_bond_dimension_(bndsize, 
                                     rand_strength = rand_strength, 
                                     inds_to_expand=inds, 
                                     inplace=True
                                     )    
    return  p

def block_match(site_blocks, x, y):
    check_ = False
    block_ = None
    for count, block in enumerate(site_blocks):
        if (x in block) and (y in block):                      
            check_ = True
            block_ = block
            break
    return check_, block_



def apply_(bpsi, gate_, where_, contract = False, tags="G"):
        
    for count, G in enumerate(gate_):
        where = where_[count]
        
        if len(where) == 2:
            x, y = where
            #G = np.transpose(G, (2,3,0,1))
            qtn.tensor_network_gate_inds(bpsi, G, [f"k{x}", f"k{y}"], 
                                        contract=contract, 
                                        tags=tags, info=None, inplace=True
                                        )
        if len(where) == 1:
            x, = where
            #print(G, bpsi[0])
            qtn.tensor_network_gate_inds(bpsi, G, [f"k{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True
                                     )



def apply_mpo(bpsi, gate_, where_, contract = False, tags="G"):
        
    for count, G in enumerate(gate_):
        where = where_[count]
        
        if len(where) == 2:
            x, y = where
            #G = np.transpose(G, (2,3,0,1))
            qtn.tensor_network_gate_inds(bpsi, G, [f"k{x}", f"k{y}"], 
                                        contract=contract, 
                                        tags=tags, info=None, inplace=True
                                        )
            qtn.tensor_network_gate_inds(bpsi, G.H, [f"b{x}", f"b{y}"], 
                                        contract=contract, 
                                        tags=tags, info=None, inplace=True
                                        )

        if len(where) == 1:
            x, = where
            qtn.tensor_network_gate_inds(bpsi, G, [f"k{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True
                                     )
            qtn.tensor_network_gate_inds(bpsi, G.H, [f"b{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True
                                     )

def apply_mpo_swap(bpsi, gate_, where_, contract = 'split', 
                    tags="G", dtype="complex128", cutoff = 1.0e-12,
                    max_bond = None, to_backend=None, 
                    ):
    swap = qu.swap(dim=2, dtype = dtype)
    swap = swap.reshape(2,2,2,2)
    if not to_backend:
        swap = get_swap(
            2, dtype=autoray.get_dtype_name(gate_[0]), backend=autoray.infer_backend(gate_[0])
        )
    if to_backend:
        swap = to_backend(swap)
    

    for count_, G in enumerate(gate_):
        where = where_[count_]
        
        if len(where) == 2:
            x, y = where
            start, end = where
            count = 0
            if start > end:
                end, start = where
                count += 1

            for i in range(start, end-1):
                qtn.tensor_network_gate_inds(bpsi, 
                                             swap, 
                                             [f"k{i}", f"k{i+1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff}
                                            )
                qtn.tensor_network_gate_inds(bpsi, 
                                             swap.conj(), 
                                             [f"b{i}", f"b{i+1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            if count == 0:
                G_ = autoray.do("transpose", G, (2,3, 0,1))
                qtn.tensor_network_gate_inds(bpsi, G_, 
                                             [f"k{end-1}", f"k{end}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )
                G_ = autoray.do("transpose", G, (2,3, 0,1)) 
                G_ = autoray.do("conj", G_ )

                qtn.tensor_network_gate_inds(bpsi, G_, 
                                             [f"b{end-1}", f"b{end}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            else:
                G_ = autoray.do("transpose", G, (2,3, 0,1)) 
                qtn.tensor_network_gate_inds(bpsi, G_, 
                                             [f"k{end}", f"k{end-1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )
                G_ = autoray.do("transpose", G, (2,3, 0,1)) 
                G_ = autoray.do("conj", G_ )
                qtn.tensor_network_gate_inds(bpsi, G_, 
                                             [f"b{end}", f"b{end-1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )
            if max_bond:
                bpsi.compress( form = "left", max_bond=max_bond, cutoff=cutoff )

            for i in reversed(range(start+1, end)):
                qtn.tensor_network_gate_inds(bpsi, swap, 
                                            [f"k{i-1}", f"k{i}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )
                qtn.tensor_network_gate_inds(bpsi, swap.conj(), 
                                            [f"b{i-1}", f"b{i}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            if max_bond:
                bpsi.compress( form = "left", max_bond=max_bond, cutoff=cutoff, **{"method": "svd"} )

        if len(where) == 1:
            x, = where
            G_ = autoray.do("transpose", G, (1,0)) 
            qtn.tensor_network_gate_inds(bpsi, G_, [f"k{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True
                                     )
            G_ = autoray.do("transpose", G, (1,0)) 
            G_ = autoray.do("conj", G_ )

            qtn.tensor_network_gate_inds(bpsi, G_, [f"b{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True
                                     )


def apply_mpo_swap_ket(bpsi, gate_, where_, contract = 'split', 
                    tags="G", dtype="complex128", cutoff = 1.0e-12,
                    max_bond = None,
                    ):
    swap = qu.swap(dim=2, dtype = dtype)
    swap = swap.reshape(2,2,2,2)
    swap = get_swap(
        2, dtype=autoray.get_dtype_name(gate_[0]), backend=autoray.infer_backend(gate_[0])
    )

    for count_, G in enumerate(gate_):
        where = where_[count_]
        
        if len(where) == 2:
            x, y = where
            start, end = where
            count = 0
            if start > end:
                end, start = where
                count += 1

            for i in range(start, end-1):
                qtn.tensor_network_gate_inds(bpsi, 
                                             swap, 
                                             [f"k{i}", f"k{i+1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            if count == 0:
                qtn.tensor_network_gate_inds(bpsi, G, 
                                             [f"k{end-1}", f"k{end}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            else:
                qtn.tensor_network_gate_inds(bpsi, G, 
                                             [f"k{end}", f"k{end-1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            for i in reversed(range(start+1, end)):
                qtn.tensor_network_gate_inds(bpsi, swap, 
                                            [f"k{i-1}", f"k{i}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )


        if len(where) == 1:
            x, = where
            qtn.tensor_network_gate_inds(bpsi, G, [f"k{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True
                                     )




def apply_mpo_swap_bra(bpsi, gate_, where_, contract = 'split', 
                    tags="G", dtype="complex128", cutoff = 1.0e-12,
                    max_bond = None,
                    ):
    swap = qu.swap(dim=2, dtype = dtype)
    swap = swap.reshape(2,2,2,2)
    swap = get_swap(
        2, dtype=autoray.get_dtype_name(gate_[0]), backend=autoray.infer_backend(gate_[0])
    )

    for count_, G in enumerate(gate_):
        where = where_[count_]
        
        if len(where) == 2:
            x, y = where
            start, end = where
            count = 0
            if start > end:
                end, start = where
                count += 1

            for i in range(start, end-1):
                qtn.tensor_network_gate_inds(bpsi, 
                                             swap.conj(), 
                                             [f"b{i}", f"b{i+1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, 
                                            inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            if count == 0:
                G_ = autoray.do("transpose", G, (2,3, 0,1)) 
                G_ = autoray.do("conj", G_ )

                qtn.tensor_network_gate_inds(bpsi, G_, 
                                             [f"b{end-1}", f"b{end}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )

            else:
                G_ = autoray.do("transpose", G, (2,3, 0,1)) 
                G_ = autoray.do("conj", G_ )
                qtn.tensor_network_gate_inds(bpsi, G_, 
                                             [f"b{end}", f"b{end-1}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )
            for i in reversed(range(start+1, end)):
                qtn.tensor_network_gate_inds(bpsi, swap.conj(), 
                                            [f"b{i-1}", f"b{i}"], 
                                            contract=contract, 
                                            tags=tags, info=None, inplace=True,
                                            **{"cutoff":cutoff}
                                            )


        if len(where) == 1:
            x, = where
            #G_ = np.transpose(G, (1,0))
            #G_ = G_.conj()
            G_ = autoray.do("transpose", G, (1,0)) 
            G_ = autoray.do("conj", G_ )

            qtn.tensor_network_gate_inds(bpsi, G_, [f"b{x}"], contract=True, 
                                        tags=tags, info=None, inplace=True
                                     )










def block_match(site_blocks, x, y):
    check_ = False
    block_ = None
    for count, block in enumerate(site_blocks):
        if (x in block) and (y in block):                      
            check_ = True
            block_ = block
            break
    return check_, block_



def gate_support(where, site_blocks):
    site_list = []
    for count, where_ in enumerate(where):
        if len(where_) == 2:
            x, y = where_
        if len(where_) == 1:
            x, = where_
            y = x
        for count_, block_ in enumerate(site_blocks):
            if x in block_:
                site_list.append(count_)
            if y in block_:
                site_list.append(count_)

    return min(site_list), max(site_list)



def dmrg_run_(where_, site_blocks):
    
    dmrg_run = False
    n_tqgate = 0
    for count in range(len(where_)):
        where = where_[count]
        if len(where) == 2:
            n_tqgate += 1
            x, y = where
            In_block, blk_ = block_match(site_blocks, x, y)                    
            
            if not In_block:
                dmrg_run = True


    return dmrg_run, n_tqgate



def tq_gates(where_):
    
    dmrg_run = False
    n_tqgate = 0
    for count in range(len(where_)):
        where = where_[count]
        if len(where) == 2:
            n_tqgate += 1

    return n_tqgate






def apply_2d(bpsi, G, where, tags="G"):
    bpsi = bpsi.copy()
    if len(where) == 2:
        x, y = where
        x0, y0 = x
        x1, y1 = y
        k0rand = qtn.rand_uuid() 
        k1rand = qtn.rand_uuid() 
        tn = qtn.Tensor(G, inds=(k0rand, k1rand, f"k{x0},{y0}",f"k{x1},{y1}"), tags=tags) 
        bpsi.reindex_({f"k{x0},{y0}":k0rand, f"k{x1},{y1}":k1rand})
        bpsi = bpsi & tn
    if len(where) == 1:
        x, = where
        x0, y0 = x
        k0rand = qtn.rand_uuid() 
        tn = qtn.Tensor(G, inds=(k0rand, f"k{x0},{y0}"), tags=tags) 
        bpsi.reindex_({f"k{x0},{y0}":k0rand})
        bpsi = bpsi & tn
    return bpsi


# sum_z = []
# sum_x = []
# psi.normalize()
# for i in range(L):
#     for j in range(0,i,1):
#         pA = psi.gate(qu.pauli('X'), i, contract=True, inplace=False)
#         pA = pA.gate_(qu.pauli('X'), j, contract=True, inplace=True)
#         x_ = (pA.H & psi).contract(all, optimize=opt)
        
#         pA = psi.gate(qu.s, i, contract=True, inplace=False)
#         pA = pA.gate_(qu.pauli('Z'), j, contract=True, inplace=True)
#         z_ = (pA.H & psi).contract(all, optimize=opt)
#         sum_z.append(z_)
#         sum_x.append(x_)
#         #print(i,j, x_, z_)

# print(  ((2* sum(sum_x)) / L**2) + 1./L, (2* sum(sum_z)+L) / L**2) 


# sum_z = []
# sum_x = []

# for i in range(L):
#     for j in range(L):
#         pA = psi.gate(qu.pauli('X'), i, contract=True, inplace=False)
#         pA = pA.gate_(qu.pauli('X'), j, contract=True, inplace=True)
#         x_ = (pA.H & psi).contract(all, optimize=opt)
        
#         pA = psi.gate(qu.pauli('Z'), i, contract=True, inplace=False)
#         pA = pA.gate_(qu.pauli('Z'), j, contract=True, inplace=True)
#         z_ = (pA.H & psi).contract(all, optimize=opt)
#         sum_z.append(z_)
#         sum_x.append(x_)
#         #print(i,j, x_, z_)



# print(sum(sum_x) / L**2, sum(sum_z) / L**2 , (sum(sum_x) / L**2) + (sum(sum_z) / L**2)) 

def peps_mps_x(peps, opt, x_0=2, x_1=4, max_bond=12, mode='full-bond'):
    if x_0 < 2:
        print("warning", "x_0 < 2")
    if x_1 >= peps.Lx -2 :
        print("warning", "x_1 >= Lx - 2")
    
    peps = peps.copy()
    peps.add_tag('KET')
    pepsH = peps.conj().retag({'KET': 'BRA'})
    norm = pepsH & peps

    
    around=[(x_0, j) for j in range(peps.Ly)]
    around += [(x_1, j) for j in range(peps.Ly)]

    norm.contract_boundary_(max_bond=max_bond, mode=mode, 
                            layer_tags=['KET', 'BRA'], 
                            around = around,
                            progbar=True,
                            cutoff=1e-12,
                            final_contract=True,
                            final_contract_opts = opt
                           )

    tags = [f"X{i}" for i in range(x_0,x_1+1,1)]
    peps_cut=norm.select(tags, which="any")
    
    norm_ = norm.select(["KET", "BRA"], which="all")

    mps_left = norm_.select(f"X{0}", which="any")
    mps_right = norm_.select(f"X{peps.Lx - 1}", which="any")
    mps_left.drop_tags(tags=["KET", "BRA"])
    mps_right.drop_tags(tags=["KET", "BRA"])


    return mps_left, mps_right, peps_cut









#     gate_pepo = []
# cor_l = []
# pepo_l = []
# for count in range(depth_):
     
#     gate_dic_t = gate_dic_l[count]
#     #print("len(gate_dic_t)", len(gate_dic_t))
#     #gate_dic_local = quf.split_dictionary(gate_dic_t, 1)
#     gate_dic_local = quf.split_dictionary(gate_dic_t,  
#                                           len(gate_dic_t),
#                                          )
    
#     for gate_dic in gate_dic_local:
#         pepo = quf.pepo_identity(Lx, Ly, dtype=dtype)
#         cor_local = []
#         gate_pepo_ = []
#         for where in gate_dic:
#             #print(len(where), where)
#             gate_pepo_.append(1)
#             if len(where) ==2:
#                 cor_local.append(where)
#             pepo = quf.gate_f(pepo, gate_dic[where], 
#                                 where, 
#                                 contract="split", 
#                                 tags=None, 
#                                 long_range_use_swaps=True,
#                                 long_range_path_sequence= ('av', 'bh'), # ('av', 'ah'),  #('v', 'h'), random, ('av', 'bv', 'ah', 'bh')
#                                 propagate_tags='sites',
#                                 inplace=False,
#                                 **{"cutoff":cutoff}
#                                 )
#         print(pepo.max_bond())
#         gate_pepo.append(len(gate_pepo_))
#         cor_l.append(cor_local)
#         pepo_l.append(pepo)


def pepo_flat_hypercompress(x_pepo, peps, chi, copt, 
                            progbar=False, tree_gauge_distance=4,
                            opt=None,
                            max_bond = None,
                            canonize_distance = 4,
                            ):

    Lx = peps.Lx
    Ly = peps.Ly
    pepsH = peps.H
    pepsH.reindex_({f"k{i},{j}":f"b{i},{j}" for (i, j)  in itertools.product(range(peps.Lx), range(peps.Ly))})
    tn = ( pepsH & x_pepo & peps)
    tn.flatten(fuse_multibonds=True, inplace=True)

    if max_bond:
        tn.compress_all(inplace=True, **{"max_bond":20, "canonize_distance":canonize_distance, "cutoff":1e-14})

    #if opt:
    try:
        res, info = apply_hyperoptimized_compressed(tn, copt, chi, output_inds=None, 
                                                tree_gauge_distance=tree_gauge_distance, 
                                                progbar=progbar, 
                                                cutoff=1.e-12
                                             )
        main, exp=(res.contract(), res.exponent)
        return main * 10**(exp)
    except:
        return tn.contract(all, optimize='auto-hq')


def pepo_BP(x_pepo, peps, progbar=False,
                            opt=None,
                            max_bond = None,
                            damping = 0.05,
                            Flat=True,
                            max_iterations = 250,
                            tol=1e-06,
                            ):

    x_pepo = x_pepo * 1.
    peps = peps * 1.
    Lx = peps.Lx
    Ly = peps.Ly
    pepsH = peps.H
    pepsH.reindex_({f"k{i},{j}":f"b{i},{j}" for (i, j)  in itertools.product(range(peps.Lx), range(peps.Ly))})
    
    tn = ( pepsH & x_pepo & peps)
    
    if Flat:
        tn.flatten(fuse_multibonds=True, inplace=True)

    if max_bond:
        tn.compress_all(inplace=True, **{"max_bond":20, "canonize_distance":canonize_distance, "cutoff":1e-14})

    (mantissa, exponent) = contract_l1bp(tn, max_iterations=max_iterations, 
                                        tol=tol, 
                                        site_tags=peps.site_tags, 
                                        damping=damping, 
                                        local_convergence=True, 
                                        update='parallel', 
                                        optimize=opt, 
                                        strip_exponent=True, 
                                        info=None,
                                        progbar=progbar,
                                        )    #tn.squeeze(fuse=True, include=None, exclude=None, inplace=True)
    
    return mantissa * 10**(exponent)







def dpepo_BP(x_pepo, peps, progbar=False,
                            opt=None,
                            max_bond = None,
                            damping = 0.06,
                            Flat=True,
                            max_iterations = 600,
                            tol=1e-06,
                            ):

    x_pepo = x_pepo * 1.
    peps = peps * 1.
    
    tn = ( x_pepo & peps)
    
    if Flat:
        tn.flatten(fuse_multibonds=True, inplace=True)
    #tn.balance_bonds_()
    if max_bond:
        tn.compress_all(inplace=True, **{"max_bond":20, "canonize_distance":canonize_distance, "cutoff":1e-14})

    (mantissa, exponent) = contract_l1bp(tn, max_iterations=max_iterations, 
                                        tol=tol, 
                                        site_tags=peps.site_tags, 
                                        damping=damping, 
                                        local_convergence=True, 
                                        update='parallel', 
                                        optimize=opt, 
                                        strip_exponent=True, 
                                        info=None,
                                        progbar=progbar,
                                        )    #tn.squeeze(fuse=True, include=None, exclude=None, inplace=True)
    
    return mantissa * 10**(exponent)



def gate_pepo(gate_, where_, Lx, Ly, cycle, cutoff=1.e-12, 
                depth_r=None, tags=[], pepo_=None, 
                dtype="complex128", bnd=32,
                sequence = ('av', 'bh', "ah", "bv"),contract='split',
                trans=True,
                ):
    
    #print("where_chunks", where_chunks)
    pepo_l = []
    bonds = []
    if depth_r:
        gate_red = [gate_[i:i + depth_r] for i in range(0, len(gate_), depth_r)]
        where_red= [where_[i:i + depth_r] for i in range(0, len(where_), depth_r)]
        pepo_local = []
        for count in range(len(where_red)):
            if pepo_:
                pepo = pepo_.copy()     
            else:
                pepo = pepo_identity(Lx, Ly, dtype=dtype)
                pepo.add_tag(tags)
                if cycle:
                    pepo = peps_cycle(pepo, int(1))

            gate = gate_red[count]
            where = where_red[count]
            
            for i_ in range(len(gate)):
                apply_2dtn(pepo, gate[i_], where[i_], bnd, 
                            bra = False, contract=contract, tags=[], 
                            dtype=dtype, cutoff = cutoff,
                            cycle = cycle,
                            sequence = sequence, #"random"
                            )
            if trans:
                pepo_trans(pepo)
            pepo_local.append(pepo)
            bonds.append(pepo.max_bond())
        pepo_l.append(pepo_local)
    else:
        if pepo_:
            pepo = pepo_.copy()     
        else:
            pepo = pepo_identity(Lx, Ly, dtype=dtype)
            pepo.add_tag(tags)
            if cycle:
                pepo = peps_cycle(pepo, int(1))

        for count, where in tqdm(enumerate(where_)):
            G = gate_[count]

            apply_2dtn(pepo, G, where, bnd, 
                        bra = False, contract=contract, tags=[], 
                        dtype=dtype, cutoff =cutoff,
                        cycle = cycle,
                        sequence = sequence, #"random"
                        )
        bonds.append(pepo.max_bond())
        if trans:
            pepo_trans(pepo)
        #pepo = pepo.H
        pepo_l.append([pepo])

    return pepo_l, bonds





def gate_pepo_(gate_, where_, Lx, Ly, cycle, cutoff=1.e-13, 
                segment_sizes = [], tags=[], pepo_=None, 
                dtype="complex128", bnd=32,
                sequence = ('av', 'bh', "ah", "bv"),contract='split',
                trans=True, max_iterations=650, tol=1.e-9, opt="auto-hq",
                ):

    if segment_sizes:
        total_length = len(gate_)
        cum_sizes = np.cumsum(segment_sizes)
        if cum_sizes[-1] > total_length:
            # truncate the last segment
            segment_sizes[-1] = total_length - (cum_sizes[-2] if len(cum_sizes) > 1 else 0)
        
        # Compute start/end indices
        bounds = np.cumsum([0] + segment_sizes)
        segments = list(zip(bounds[:-1], bounds[1:]))
        
        # Slice the lists
        gate_red = [gate_[i:j] for i, j in segments]
        where_red = [where_[i:j] for i, j in segments]




    # print([ len(i) for i in where_red], sum([ len(i) for i in where_red]))
    #print("where_chunks", where_chunks)
    sequence = list(sequence)
    pepo_l = []
    bonds = []
    if segment_sizes :
        pepo_local = []
        for count in range(len(where_red)):
            if pepo_:
                pepo = pepo_.copy()     
            else:
                pepo = pepo_identity(Lx, Ly, dtype=dtype)
                pepo.add_tag(tags)
                if cycle:
                    pepo = peps_cycle(pepo, int(1))

            gate = gate_red[count]
            where = where_red[count]
            
            for i_ in range(len(gate)):
                random.shuffle(list(sequence))
                apply_2dtn(pepo, gate[i_], where[i_], bnd, 
                            bra = False, contract=contract, tags=[], 
                            dtype=dtype, cutoff = cutoff,
                            cycle = cycle,
                            sequence = sequence, #"random"
                            )
                if pepo.max_bond() > bnd:

                    bp = L2BP(
                        pepo,
                        local_convergence=True,
                        optimize=opt,
                        site_tags=pepo.site_tags,
                        damping=0.0,
                        update='parallel',
                    )
                    bp.run(tol=tol, max_iterations=max_iterations, progbar=True, diis=True)
                    pepo = bp.compress(
                        pepo,
                        max_bond=bnd,
                        cutoff=cutoff,
                        cutoff_mode="rsum2",
                        renorm=0,
                        lazy=False
                    )

            
            
            if trans:
                pepo_trans(pepo)
            pepo_local.append(pepo)
            bonds.append(pepo.max_bond())
        pepo_l.append(pepo_local)
    else:
        if pepo_:
            pepo = pepo_.copy()     
        else:
            pepo = pepo_identity(Lx, Ly, dtype=dtype)
            pepo.add_tag(tags)
            if cycle:
                pepo = peps_cycle(pepo, int(1))

        for count, where in tqdm(enumerate(where_)):
            random.shuffle(list(sequence))
            G = gate_[count]

            apply_2dtn(pepo, G, where, bnd, 
                        bra = False, contract=contract, tags=[], 
                        dtype=dtype, cutoff =cutoff,
                        cycle = cycle,
                        sequence = sequence, #"random"
                        )
            
            if pepo.max_bond() > bnd:
                compress_l2bp(pepo, site_tags=pepo.site_tags,
                    max_bond=bnd, 
                    cutoff=cutoff,
                    cutoff_mode='rsum2',
                    max_iterations=max_iterations,
                    tol=tol,
                    progbar=False,
                    inplace=True,
                    damping=0.01,
                    local_convergence=True,
                    )



        bonds.append(pepo.max_bond())
        if trans:
            pepo_trans(pepo)
        pepo_l.append([pepo])

    return pepo_l, bonds
















def gate_chunk(gate_l, where_l, gate_round):
    where_chunks = []
    gate_chunks = []
    for count, i in enumerate(gate_round):
        start = sum(gate_round[:count])
        end = start +  gate_round[count]
        where_chunks.append(where_l[start:end])
        gate_chunks.append(gate_l[start:end])
    
    return gate_chunks, where_chunks



def get_to_backend(backend):
    if "torch" in backend:
        import torch

        if "cpu" in backend:
            device = "cpu"
        elif "gpu" in backend:
            device = "cuda"
        elif "mps" in backend:
            device = "mps"
        else:
            # default to gpu
            device = "cuda"

        if "double" in backend:
            dtype = torch.complex128
        elif "float64" in backend:
            dtype = torch.float64
        elif "float32" in backend:
            dtype = torch.float32

        else:
            # default to single precision
            dtype = torch.complex64

        def to_backend(x):
            return torch.tensor(x, dtype=dtype, device=device)

    elif "cupy" in backend:
        import cupy

        if "double" in backend:
            dtype = "complex128"
        elif "float64" in backend:
            dtype = "float64"
        elif "float32" in backend:
            dtype = "float32"
        else:
            dtype = "complex64"


        def to_backend(x):
            return cupy.asarray(x, dtype=dtype)

    elif "tensorflow" in backend:
        import tensorflow as tf

        if "cpu" in backend:
            device = "/cpu:0"
        elif "gpu" in backend:
            device = "/gpu:0"
        else:
            # default to gpu
            device = "/gpu:0"

        if "double" in backend:
            dtype = tf.complex128
        else:
            # default to single precision
            dtype = tf.complex64

        def to_backend(x):
            with tf.device(device):
                return tf.constant(x, dtype=dtype)

    else:  # assume numpy
        if "double" in backend:
            dtype = "complex128"
        else:
            dtype = "complex64"

        def to_backend(x):
            return x.astype(dtype)

    return to_backend


def torch_to_gpu(x):
    return x.cuda()


def torch_to_cpu(x):
    return x.cpu()



def contract_hypercompress(x_pepo, peps, chi, copt, 
                            progbar=False, tree_gauge_distance=4,
                            opt=None,
                            max_bond = None,
                            canonize_distance = 4,
                            tags = None,
                            ):

    L = len(peps.tensor_map)
    all_inds = list(x_pepo.outer_inds())
    
    pepsH = peps.H
    pepsH.reindex_( {i:i.replace("k", "b") for i  in all_inds if i.startswith('k')})
    tn = ( pepsH & x_pepo & peps)
    #tn.flatten(fuse_multibonds=True, inplace=True)
    tn.rank_simplify(inplace=True)
    tn.fuse_multibonds(inplace=True)
    #tn.squeeze(fuse=True, include=None, exclude=None, inplace=True)


    if max_bond:
        tn.compress_all(inplace=True, **{"max_bond":20, "canonize_distance":canonize_distance, "cutoff":1e-14})

    res = apply_hyperoptimized_compressed(tn, copt, chi, output_inds=None, 
                                      tree_gauge_distance=tree_gauge_distance, 
                                      progbar=progbar, 
                                      cutoff=1.e-12
                                      )
    main, exp=(res.contract(), res.exponent)
    return main * 10**(exp)


def contract_BP(x_pepo, peps, progbar=False,
                            opt=None,
                            ):
    peps=peps.copy()
    L = len(peps.tensor_map)
    all_inds = list(x_pepo.outer_inds())
    
    pepsH = peps.H
    pepsH.reindex_( {i:i.replace("k", "b") for i  in all_inds if i.startswith('k')})
    site_tags = [i.replace("k", "I") for i  in all_inds if i.startswith('k')]
    tn = ( pepsH & x_pepo & peps)
    #tn.flatten(fuse_multibonds=True, inplace=True)
    #tn.rank_simplify(inplace=True)
    #tn.fuse_multibonds(inplace=True)
    #print(tn)
    #tn.squeeze(fuse=True, include=None, exclude=None, inplace=True)

    (mantissa, exponent) = contract_l1bp(tn, max_iterations=1000, 
                                        tol=5e-06, 
                                        #site_tags=site_tags, 
                                        damping=0.0, 
                                        local_convergence=True, 
                                        update='parallel', 
                                        optimize=opt, 
                                        strip_exponent=True, 
                                        info=None,
                                        progbar=progbar,
                                        )    #tn.squeeze(fuse=True, include=None, exclude=None, inplace=True)
    
    return mantissa * 10**(exponent), site_tags



def logsumexp(x): 
    #x = np.array(l_)
    c = x.max()
    return c + np.log10(np.sum(10**(x - c)))
def sumexp(x):
    #x = np.array(l_) 
    x = 10**x
    return x




@ray.remote
def generate_samples(peps, mps,inds_fuse, chi=10, sample=10, 
                     max_iterations=20, tol=0.01, 
                     log_=True, backend=None, to_backend=None,
                     damping=0.01,
                     Lx=4, Ly=4,
                     equalize_norms = True,
                     where_l = None,
                     ):
    #os.environ["NUMBA_NUM_THREADS"] = "1"
    #to_backend = get_to_backend(backend)
    #import warnings

    #warnings.filterwarnings("ignore", message="Belief propagation did not converge*", category=UserWarning)
    #warnings.filterwarnings("ignore", message="'NUMBA_NUM_THREADS' has been set elsewhere*", category=UserWarning)
    #warnings.filterwarnings("ignore", message="ComplexWarning: Casting complex*", category=UserWarning)
    #warnings.filterwarnings("ignore", category="ComplexWarning")

 

    # Get the value of OMP_NUM_THREADS
    # numba_num_threads = os.getenv("NUMBA_NUM_THREADS")
    
    # if numba_num_threads is not None:
    #     print("numba_num_threads is set to:", numba_num_threads)
    
    # else:
    #     print("numba_num_threads is not set.")

    def z_config(config, where_l):

        sum_ = 0+0j
        for where in where_l:
            x, y = where
            if config[f"k{x},{y}"] == 0:
                sum_ += 1.+0j
            elif config[f"k{x},{y}"] == 1:
                sum_ += -1+0j   
        
        sum_ = sum_/len(where_l)
        return sum_

    def z_config_(config, where_l):
        sum_ = 1+0j
        for where in where_l:
            x, y = where
            if config[f"k{x},{y}"] == 0:
                sum_ *= 1.+0j
            elif config[f"k{x},{y}"] == 1:
                sum_ *= -1+0j   
        
        return sum_



    def block_k(k, inds_fuse):
        for count, i in enumerate(inds_fuse): 
            if k in i:
                return count

    def tn_config_1d(mps, config, inds_fuse, to_backend):
        tn = mps.copy()
        
        count = 0
        tn_list = []
        for i in config:
            block_i= block_k(i, inds_fuse)
            matches = re.findall(r'k(\d+)', i)
            matches = matches[0]
            x = matches
            x = int(x)
            if config[i] == 0:
                t_ = qtn.Tensor(data = np.array([1,0]), inds= [i], tags = "G")
            else:
                t_ = qtn.Tensor(data = np.array([0,1]), inds= [i], tags = "G")
            
            t_.apply_to_arrays(to_backend)
            tn_list.append(t_)
            tn = (tn & t_)
            tn = tn.contract(tags=["G", f"I{block_i}"])
            tn.drop_tags(tags=["G"])
            count += 1
        
        tn.view_as_(qtn.tensor_1d.TensorNetwork1DFlat, L = len(inds_fuse), site_tag_id='I{}')

        return tn

    def sample_mps_eff(psi, inds_fuse, to_backend=None, dic_2d=None):
        Lmps = len(inds_fuse)
        
        t_0 = qtn.Tensor(data = np.array([1,0]), inds= [f"k{0}"], tags = None)
        t_1 = qtn.Tensor(data = np.array([0,1]), inds= [f"k{0}"], tags=None)
        t_0.apply_to_arrays(to_backend)
        t_1.apply_to_arrays(to_backend)

        t = psi[f"I{0}"]
        sample = []
        sum = 0
        omega = 0
        for cor in range(Lmps):
    #---------------------- start: inner block -----------------------------------
            for cor_ in range(len(inds_fuse[cor])):
                cor_ = sum + cor_
                tH=t.H
                tH.reindex_({f"k{cor_}":f"b{cor_}"})
                rho = (tH & t)^all
                rho_ = autoray.do("to_numpy", rho.data).astype("float64")
                rho_diag = np.diag(rho_)
                rho_diag =  rho_diag * (1/rho_diag.sum())
                samples = np.random.choice([0, 1], 1, p=rho_diag)
                unique, counts = np.unique(samples, return_counts=True)
                #omega  *= rho_diag[unique[0]]
                omega += autoray.do("log10", rho_diag[unique[0]])
                sample.append(unique[0])
                if unique[0] == 0:
                    t_0.modify(inds=[f"k{cor_}"])
                    t = (t_0 & t )
                    t = t.contract(all)
                else:
                    t_1.modify(inds=[f"k{cor_}"])
                    t = (t_1 & t ) 
                    t = t.contract(all)
            
            sum += len(inds_fuse[cor])
    #---------------------- end: inner block -----------------------------------
            if cor < Lmps-1:
                t = t & psi[f"I{cor+1}"]


        
        dic_2d_ = {}
        if dic_2d:
            Lx = dic_2d["Lx"]
            Ly = dic_2d["Ly"]
            dic_ = {}
            for i in range(Lx):
                for j in range(Ly):
                        dic_ |= { f"{j*Lx + i}":(i,j)} 

            for count, i in enumerate(sample):
                x, y = dic_[f"{count}"]
                dic_2d_ |= {f"k{x},{y}":i}

        dic_1d = {}
        for count, i in enumerate(sample):
            dic_1d |= {f"k{count}":i}

        tn_config = None
        #tn_config = tn_config_1d(psi, dic_1d, inds_fuse, to_backend)
        
        return dic_1d, dic_2d_, omega, tn_config



    def tn_config_(peps, config, to_backend):
        tn = peps.copy()
        
        for i in config:
            matches = re.findall(r'k(\d+),(\d+)', i)
            matches = matches[0]
            x, y = matches
            x = int(x)
            y = int(y)
            if config[i] == 0:
                t_ = qtn.Tensor(data = np.array([1,0]), inds= [i], tags = "G")
            else:
                t_ = qtn.Tensor(data = np.array([0,1]), inds= [i], tags = "G")
            
            t_.apply_to_arrays(to_backend)
            tn = (tn & t_)
            tn = tn.contract(tags=["G", f"I{x},{y}"])
            tn.drop_tags(tags=["G"])
        tn.view_as_(
            qtn.tensor_2d.TensorNetwork2DFlat,
            Lx=peps.Lx, 
            Ly=peps.Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
                    )
    
        return tn   


    def get_optimizer_exact(
        target_size=2**27,
        minimize="combo",
        max_time="rate:1e8",
        directory=True,
        progbar=False,
        **kwargs,
    ):
        import cotengra as ctg
    
        if "parallel" not in kwargs:
            if "OMP_NUM_THREADS" in os.environ:
                parallel = int(os.environ["OMP_NUM_THREADS"])
            else:
                import multiprocessing as mp
                parallel = mp.cpu_count()
    
            if parallel == 1:
                parallel = False
    
            kwargs["parallel"] = parallel
    
        if target_size is not None:
            kwargs["slicing_reconf_opts"] = dict(target_size=target_size)
        else:
            kwargs["reconf_opts"] = {}
    
    
        return ctg.ReusableHyperOptimizer(
            progbar=progbar,
            minimize=minimize,
            max_time=max_time,
            directory="cash/",
            **kwargs,
        )
    target_size=2**28
    opt = get_optimizer_exact(target_size=target_size, 
                              parallel=False, 
                              #progbar=True, 
                              #max_repeats=
                              )



    copt = ctg.ReusableHyperCompressedOptimizer(
            chi,
            max_repeats=2**6,
            minimize='combo-compressed', 
            max_time="rate:1e8",
            progbar=False,
            parallel=True,
            directory="cash/",
            )


    def apply_hyperoptimized_compressed(tn, copt, max_bond, 
                                        output_inds=None, tree_gauge_distance=4,
                                          progbar=False, cutoff=1.e-12, 
                                          equalize_norms=1.0):
        tree = tn.contraction_tree(copt)
        tn.contract_compressed_(
            optimize=copt,
            output_inds=output_inds,
            max_bond=max_bond,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=1.0,
            cutoff=1e-10,
            progbar=progbar,
        )
        return tn

    def cal_log(z_sum, main, exponent, omega_log10):
        probz_log10 = autoray.do("log10", z_sum) + autoray.do("log10", main) + autoray.do("log10", main_conj) + exponent + autoray.do("conj", exponent)
        return probz_log10


    x_log = np.array([])
    x2_log = np.array([])
    xxh_log = np.array([])
    xxv_log = np.array([])

    weight_log = np.array([]) 
    dic_2d = {"Lx":Lx, "Ly":Ly}

    for _ in tqdm(range(sample)):
    #for _ in range(sample):

        
        
        if mps:
            start_time = time.time()
            config_1d, config, omega_log10, tn_config = sample_mps_eff(mps, inds_fuse, to_backend=to_backend, dic_2d=dic_2d)
            
            #tn_config_x = tn_config_(pepsx, config, to_backend)
            tn_config = tn_config_(peps, config, to_backend)
            omega = 10**omega_log10 
            #print("mps", (time.time() - start_time))

        else:
            config, tn_config , omega  = sample_d2bp(peps, max_iterations=max_iterations, 
                                                        tol=tol,
                                                        damping=damping, 
                                                        #optimize=opt, 
                                                        progbar=False,
                                                    )
        
            #tn_config_x = tn_config_(pepsx, config, to_backend)
            omega_log10 = autoray.do("log10", omega)

        # timing
        start_time = time.time()

        where_l = [ (i,j)  for i,j in itertools.product(range(peps.Lx), range(peps.Ly)) ]
        z2_sum = z_config(config, where_l)
        z2_sum = z2_sum**2

        z_sum = z_config_(config, [(3,3)])
        zzh_sum = z_config_(config, [(3,3), (4,3)])
        zzv_sum = z_config_(config, [(3,3), (3,4)])

        
        tn_config_h = tn_config.H

        tn_config_h.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)

        # tn_config_h.contract_boundary_( max_bond=chi, final_contract = True, 
        #                                final_contract_opts={"optimize":opt}, 
        #                                max_separation = 1, 
        #                                sequence = ['xmin', 'xmax', "ymin", "ymax" ], 
        #                                equalize_norms=equalize_norms, progbar=True,
        #                                )    
        # tn_config_h.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)
        # tn_config_h = apply_hyperoptimized_compressed(tn_config_h, copt, chi, 
        #                                         #output_inds=output_inds,
        #                                         tree_gauge_distance=2, 
        #                                         progbar=False, 
        #                                         cutoff=1.e-12
        #                                     )

        
        tn_config_h.contract_ctmrg(max_bond=chi, cutoff=1e-10, canonize=True,   mode='projector', max_separation=1, equalize_norms=equalize_norms, final_contract=True, final_contract_opts={"optimize": opt}, progbar=False, inplace=True)

        
        main,  exponent = (tn_config_h.contract(), tn_config_h.exponent)
        main_conj = autoray.do("conj", main)

        #start_time = time.time()

        #print("contract_time", (time.time() - start_time))

        if log_:
            probz_log10 = cal_log(z_sum, main, exponent, omega_log10)
            probz2_log10 = cal_log(z2_sum, main, exponent, omega_log10)
            probzzv_log10 = cal_log(zzh_sum, main, exponent, omega_log10)
            probzzh_log10 = cal_log(zzv_sum, main, exponent, omega_log10)
            
            prob_log10 = autoray.do("log10", main) + autoray.do("log10", main_conj) + exponent + autoray.do("conj", exponent)

            x_log = np.append(x_log, probz_log10 - omega_log10)
            x2_log = np.append(x2_log, probz2_log10 - omega_log10)
            xxh_log = np.append(xxh_log, probz2_log10 - omega_log10)
            xxv_log = np.append(xxv_log, probz2_log10 - omega_log10)

            weight_log = np.append(weight_log, prob_log10 - omega_log10)

        else:
            prb_ = (main*(10**exponent)) * (autoray.do("conj", main*(10**exponent)))
            x_log = np.append(x_log, (z_sum*prb_)/omega)
            x2_log = np.append(x2_log,  (z2_sum*prb_)/omega)
            #x4_log = np.append(x4_log,  (z4_sum*prb_)/omega)
            weight_log = np.append(weight_log, prb_/ omega)



    return (x_log, x2_log, xxh_log, xxv_log, weight_log)


def peps_sample(peps, mps,
                chi=10, 
                where_l= [(0,0)], 
                num_workers = 20,
                samples_per_worker =10,
                for_each_repeat=1,
                backend="numpy-cpu-double",
                method="mps",
                log_ = True,
                equalize_norms = 1.,
                ):

    
    #os.environ["OMP_NUM_THREADS"] = "1"
    to_backend = get_to_backend(backend)
    Lx = peps.Lx
    Ly = peps.Ly
    L = Lx*Ly
    inds_fuse = []
    for i in range(L):
        inds_fuse.append([f"k{i}"])

    for_each_repeat = for_each_repeat
    num_workers = num_workers
    samples_per_worker = samples_per_worker
    num_samples = num_workers * samples_per_worker  # Total number of samples
    t_sample = num_samples * for_each_repeat

    dic_imp = {
                "sample": samples_per_worker, 
                "log_" : log_,
                "max_iterations" : 120,
                "tol" : 8.e-3,
                "chi": chi,
                "backend": backend,
                "to_backend": to_backend,
                "damping":0.01,
                "Lx": Lx, 
                "Ly": Ly,
                "equalize_norms": equalize_norms,
                "where_l": where_l,
               }
    print(f"---------------------Sample-method={method}-------------------------------")
    print("where_l", where_l)
    print("sample_total", t_sample, "chi", chi, "equalize_norms", equalize_norms)
    print("sample_info:", "norm", dic_imp["equalize_norms"], "log", dic_imp["log_"])
    samples_ = []
    for _ in range(for_each_repeat):
        sample_futures = [generate_samples.remote(peps, mps, inds_fuse, **dic_imp) for _ in range(num_workers)]    
        samples = ray.get(sample_futures)
        #samples = [generate_samples(peps, mps, inds_fuse, **dic_imp) for _ in range(num_workers)]    
        samples_.append(samples)

    x_f_l = np.array([])
    x2_f_l = np.array([])
    xxh_f_l = np.array([])
    xxv_f_l = np.array([])

    wight_f_l = np.array([])
    

    for samples in samples_:
        for info_ in samples:
            xs, xs2, xxh, xxv,  wights= info_
            x_f_l = np.concatenate((x_f_l, xs))
            x2_f_l = np.concatenate((x2_f_l, xs2))
            xxh_f_l = np.concatenate((xxh_f_l, xxh))
            xxv_f_l = np.concatenate((xxv_f_l, xxv))
            wight_f_l = np.concatenate((wight_f_l, wights))

    # x_top = logsumexp(x_f_l)
    # w_top = logsumexp(wight_f_l)
    if log_:
        x_top_ = sumexp(x_f_l)
        w_top_ = sumexp(wight_f_l)
        x2_top_ = sumexp(x2_f_l)
        xxh_top_ = sumexp(xxh_f_l)
        xxv_top_ = sumexp(xxv_f_l)
    else:
        x_top_ = x_f_l
        w_top_ = wight_f_l
        x2_top_ = x2_f_l
        #x4_top_ = x4_f_l

    #print(len(x_f_l),  10**(x_top-w_top), sum(x_top_)/sum(w_top_) )
    #X = 10**(x_top-w_top)
    
    X_ = sum(x_top_)/sum(w_top_)
    X_2 = sum(x2_top_)/sum(w_top_)
    XX_h = sum(xxh_top_)/sum(w_top_)
    XX_v = sum(xxv_top_)/sum(w_top_)

    X_noisy, error_noisy = bootstrap(x_top_, w_top_, B_sample=500)
    X2_noisy, error2_noisy = bootstrap(x2_top_, w_top_, B_sample=500)
    XXh_noisy, error3_noisy = bootstrap(xxh_top_, w_top_, B_sample=500)
    XXv_noisy, error4_noisy = bootstrap(xxv_top_, w_top_, B_sample=500)

    print("x_noisy", (X_.real, error_noisy))
    print("x2_noisy", (X_2.real, error2_noisy))
    print("xxh_noisy", (XXh_noisy.real, error3_noisy))
    print("xxv_noisy", (XXv_noisy.real, error4_noisy))

    #disprob = np.mean(w_top_).real
    norm_peps, norm_peps_error = bootstrap_(w_top_, B_sample=500)
    disprob = (norm_peps.real, norm_peps_error.real)
    print("peps_norm ~ ", disprob)
    
    D_kl, D_kl_error = bootstrap_(wight_f_l, B_sample=500)
    D_kl = autoray.do("log10", norm_peps.real) - D_kl.real
    D_kl_ = (D_kl.real, D_kl_error.real)
    print("D_kl ~ ", D_kl_)




    #bnd_max = peps.max_bond()
    #qu.save_to_disk((x_top_,x2_top_,w_top_, wight_f_l), f"Store/info_peps/sample_l{label}_bnd{bond_dim}_d{depth}")
    
    res = {"weights":w_top_, "ampl_z":x_top_, "ampl_z2":x2_top_, "ampl_zzh":xxh_top_, "ampl_zzv":xxv_top_, "wights_log":wight_f_l}
    
    res |= {"zzh":XXh_noisy.real, "error_zh":error3_noisy}
    res |= {"zzv":XXv_noisy.real, "error_zv":error3_noisy}

    res |= {"z2":X2_noisy.real, "error_z2":error2_noisy}

    res |= {"z":X_noisy.real, "error_z":error_noisy, "D_kl":D_kl.real, "error_D_kl":D_kl_error.real}
    return res


def bootstrap(data_l, data_l_, B_sample=100):
    #print(data_l,sample(data_l,len(data_l)))
    avg_boot=[]
    for i in range(B_sample):
        indices = np.random.choice(len(data_l), size=len(data_l), replace=True)
        repla_data = [data_l[i] for i in indices]
        repla_data_ = [data_l_[i] for i in indices]
        avg_boot.append( sum(repla_data) / sum(repla_data_) ) 

    avg_r = sum(avg_boot) / len(avg_boot)
    #print(avg_r,avg_boot)
    var_r = sum((x-avg_r)**2 for x in avg_boot) / ( len(avg_boot)-1 )
 
    return avg_r, np.std(avg_boot)

def bootstrap_(data_l, B_sample=100):
    avg_boot=[]
    for i in range(B_sample):
        repla_data=np.random.choice(data_l, size=len(data_l), replace=True)
        avg_boot.append( sum(repla_data)/len(repla_data)  ) 

    avg_r = sum(avg_boot) / len(avg_boot)
    #print(avg_r,avg_boot)
    var_r = sum((x-avg_r)**2 for x in avg_boot) / ( len(avg_boot)-1 )
 
    return avg_r, np.std(avg_boot)



def Mps_cluster(bnd=4, lx=4, ly=5, depth_=20):
    bnd = bnd
    lx = lx
    ly = ly
    b_ = [ly] * lx
    depth_ = depth_
    depth_r = 2
    threads = None
    label = "PBC"
    to_backend, opt_, opt = req_backend(threads=threads)
    psi, block_l, inds_fuse, bnds = mps_prepare(to_backend, bnd=bnd, b=b_, theta = - 2*math.pi/48)
    circ, gate_l, where_l, info = prepare_gates(to_backend, 
                                            depth_=depth_, opt=opt, 
                                            depth_r=depth_r,
                                            lx=lx, 
                                            ly=ly,
                                            label = label,
                                            cycle_gates = True,
                                            exact_cal = False
                                            )


    peps, mpo_l, pepo_l, x_pepo, label, O_label, site, site_2d = info



    #info passed to mps simulation:
    dic = {
        "smart_canon" : True,
        "prgbar" : True, 
        "to_backend" : to_backend,
        "bnds" : bnds,
        "block_l" : block_l,
        }

    dic_dmrg = {
        "depth_r": depth_r, 
        "label": "PBC", #sycamore
        "fidel_cal" : True,
        "svd_init": False,
        "opt_":opt_,
        "O_label": O_label,
        "site": site,
        "store_state": True,
        "mpo_l": mpo_l,
            }
    # how many gates we want to pass to mps simulation:


    (psi, inds_fuse), info, X_l = Mps_dmrg(psi, opt_, gate_l, 
                                            where_l,
                                            inds_fuse, **dic, **dic_dmrg
                                            )
    psi.normalize()
    print(psi.show())
    print("X_l", X_l)
