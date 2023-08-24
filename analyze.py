import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from collections import Counter
from collections import OrderedDict
from multiprocessing import Pool
from moses.metrics import metrics
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from Utils.metric import get_property_fn, error_fn, obabel_logP
from Utils.smiles import get_smi, get_mol
import seaborn as sns
from Utils.plot import kde_plot


def mapper(fn, iterable, n_jobs=1):
    if n_jobs == 1:
        res = list(map(fn, iterable))
    else:
        with Pool(n_jobs) as pool:
            res = pool.map(fn, iterable)
    return res


def get_valid(smiles_list, method, n_jobs):
    mols = np.array(mapper(get_mol[method], smiles_list, n_jobs))
    is_valid = (mols != None)
    valid_mol = np.array(mols)[is_valid]
    valid_smi = np.array(mapper(get_smi[method], valid_mol))
    return valid_smi, valid_mol, is_valid

        
def get_properties(valid_mol, property_fn, n_jobs=1):
    props = OrderedDict()        
    for p, fn in property_fn.items():
        props[p] = mapper(fn, valid_mol, n_jobs)
    return pd.DataFrame(props)
                 

def mode(data):
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 10000)
    pdf = kde(x)
    return x[np.argmax(pdf)]


def compute_prop_stat(ref, gen_data, target_logP, property_fn,
                      logp_fn, prop_folder):
    # compute statistics

    stat = OrderedDict()

    # stat = { 'validity': [], 'uniqueness': [], 'novelty': [], 'internal_diversity': [],
    #          'MSD': [], 'MAD': [], 'MIN': [], 'MAX': [], 'SD': [] }

    for file_name, gen_smi in gen_data.items():
        tmp_stat = OrderedDict()
        print(f'#processing {file_name}')
        
        rd_mols = np.array(mapper(get_mol['rdkit'], gen_smi, n_jobs))
        ob_mols = np.array(mapper(get_mol['obabel'], gen_smi, n_jobs=1))
        
        rd_valid_smi = []
        rd_valid_mol = []
        ob_valid_smi = []
        ob_valid_mol = []
        
        for i in range(len(gen_smi)):
            if rd_mols[i] is not None and ob_mols[i] is not None:
                rd_valid_smi.append(gen_smi[i])
                ob_valid_smi.append(gen_smi[i])
                rd_valid_mol.append(rd_mols[i])
                ob_valid_mol.append(ob_mols[i])
    
        # assert len(rd_valid_smi) == len(ob_valid_smi)

        print('get properties...')

        logp = get_properties(ob_valid_mol, logp_fn, n_jobs=1)
        prop = get_properties(rd_valid_mol, property_fn, n_jobs)
        prop = pd.concat([logp, prop], axis=1)
        prop.insert(0, 'SMILES', rd_valid_smi)
        
        prop.to_csv(os.path.join(prop_folder, f'{file_name}.csv'), index=False)
        
        print('get statistics...')
        
        tmp_stat['validity'] = len(rd_valid_smi) / len(gen_smi)
        tmp_stat['uniqueness'] = len(set(rd_valid_smi)) / len(rd_valid_smi)
        tmp_stat['novelty'] = metrics.novelty(rd_valid_smi, ref, n_jobs)
        tmp_stat['internal_diversity'] = metrics.internal_diversity(rd_valid_smi, n_jobs)
        
        for p in ['logP'] + list(property_fn.keys()):
            tmp_stat[f'{p}_avg'] = prop[p].mean()
            tmp_stat[f'{p}_mode'] = mode(prop[p])
        
        error = error_fn(logp['logP'].tolist(), [target_logP]*len(logp))
        for e in error:
            tmp_stat[e] = error[e]
        
        for k, v in tmp_stat.items():
            if k not in stat:
                stat[k] = []
            stat[k].append(v)
        print(stat)

    stat = pd.DataFrame(stat, index=gen_data.keys())
    return stat
    
    
def prepare_ga_data(data_folder, method_list, n_test_time):
    ga_gen_data = {}
        
    for method in method_list:
        for i in range(n_test_time):
            df = pd.read_csv(os.path.join(data_folder, f'{method}{i}.txt'),
                             sep='\t', index_col=[0])
            ga_gen_data[f'{method}{i}'] = df['Canonical_SMILES'].tolist()
    return ga_gen_data


def prepare_tf_data(data_folder, method_list, n_test_time):
    tf_gen_data = {}
    
    for method in method_list:
        for i in range(n_test_time):
            df = pd.read_csv(os.path.join(data_folder, f'{method}_{i}.csv'))
            tf_gen_data[f'{method}_{i}'] = df['SMILES'].tolist()
    return tf_gen_data
            

def compute_src_prop_stat(src, n_jobs):
    rd_mol = mapper(get_mol['rdkit'], src, n_jobs=n_jobs)
    ob_mol = mapper(get_mol['obabel'], src, n_jobs=1)
    
    logp = get_properties(ob_mol, logp_fn, n_jobs=1)
    prop = get_properties(rd_mol, property_fn, n_jobs)
    prop = pd.concat([logp, prop], axis=1)
    prop.insert(0, 'SMILES', src)
    
    stat = OrderedDict()
    for p in ['logP'] + list(property_fn.keys()):
        stat[f'{p}_avg'] = [prop[p].mean()]
        stat[f'{p}_mode'] = [mode(prop[p])]
    stat = pd.DataFrame(stat)
    return prop, stat


if __name__ == "__main__":
    n_jobs = 8
    
    # Tasks
    
    compute_src_prop_stat = False
    compute_ga_prop_stat = False
    compute_tf_prop_stat = True
    plot_ga_prop = False
    plot_tf_prop = False
    
    # n_test_time = 0
    
    # for i, epoch in enumerate([36, 43, 39, 38, 40]):
    #     for step in [1,2,3,4]:
    #         df = pd.read_csv(f'/home/chaoting/ML/Molecular-Optimization/experiments/'
    #                          f'formal_evaluation_transformer_32_3_8_512_2048_0.1_{i+1}/'
    #                          f'GA_sample_logP3.5/evaluation_{epoch}/step{step}/prediction.csv')
    #         smi = df[f'Source_Mol_{step}'].tolist()
    #         df = pd.DataFrame({ 'SMILES': smi })
    #         df.to_csv(f'Results/Gen/tf-{step}step_{i}.csv', index=False)
    
    # target
    
    target_logP = 3.5
    
    # methods
    
    ga_methods = ["samegen-MC", "Roulette", "parent-child-MC"]
    tf_methods = ["tf-1step", "tf-2step", "tf-3step", "tf-4step"]

    # our interested properties

    interested_props = ['tPSA', 'QED', 'SAS', 'NP', 'MW', 'NATOM']
    
    print('define data/save folder...')
    
    data_folder = os.path.join('./Results', 'Gen')
    prop_folder = os.path.join('./Results', 'Prop')
    stat_folder = os.path.join('./Results', 'Stat')
    fig_folder = os.path.join('./Results', 'Figure')
    
    print('define property functions...')
    
    logp_fn = { 'logP': obabel_logP }
    property_fn = get_property_fn(interested_props)
        
    if compute_src_prop_stat:
        print('compute source properties...')
    
        ga_src = pd.read_csv(os.path.join('Data', 'ga-src.csv'))
        prop, stat = compute_src_prop_stat(ga_src['SMILES'], n_jobs)
        prop.to_csv(os.path.join(prop_folder, 'ga-src.csv'), index=False)
        stat.to_csv(os.path.join(stat_folder, 'ga-src.csv'), index=False)

        tf_src = pd.read_csv(os.path.join('Data', 'tf-src.csv'))
        prop, stat = compute_src_prop_stat(tf_src['SMILES'], n_jobs)
        prop.to_csv(os.path.join(prop_folder, 'tf-src.csv'), index=False)
        stat.to_csv(os.path.join(stat_folder, 'tf-src.csv'), index=False)
        
    if compute_ga_prop_stat:
        print('compute ga statistics...')
        
        ga_data = prepare_ga_data(data_folder, ga_methods, n_test_time=4)
        stat = compute_prop_stat(ga_src, ga_data, target_logP, property_fn,
                                 logp_fn, prop_folder)
        stat.to_csv(os.path.join(stat_folder, 'ga.csv'))
    
    if compute_tf_prop_stat:
        print('compute tf statistics...')
        
        train = pd.read_csv('./Data/tf-train.csv')
        train = train['Source_Mol']

        tf_data = prepare_tf_data(data_folder, tf_methods, n_test_time=5)
        stat = compute_prop_stat(train, tf_data, target_logP, property_fn,
                                 logp_fn, prop_folder)
        stat.to_csv(os.path.join(stat_folder, 'tf.csv'))
    
    # figure settings
        
    prop_xlimit = {
        'logP' : (-2, 8),
        'tPSA' : (0, 300),
        'QED'  : (0, 1),
        'SAS'  : (0, 10),
        'NP'   : (-5, 5),
        'MW'   : (0, 1000),
        'NATOM': (0, 70)
    }

    if plot_ga_prop:
        print('plot ga figures...')
        
        src_prop = pd.read_csv(os.path.join(prop_folder, 'ga-src.csv'))
        
        for prop in ['logP'] + interested_props:
            stat_plot = {}
            stat_plot['source'] = src_prop[prop]
            
            for method in ga_methods:
                df = pd.read_csv(os.path.join(prop_folder, f'{method}0.csv'))
                stat_plot[method] = df[prop]
            
            stat_plot = pd.DataFrame(stat_plot)
            kde_plot(stat_plot, os.path.join(fig_folder, f'ga-{prop}.png'),
                     xlabel=prop, ylabel='Density', xlimit=prop_xlimit[prop])
    
    if plot_tf_prop:
        print('plot tf figures...')
        
        src_prop = pd.read_csv(os.path.join(prop_folder, 'tf-src.csv'))
            
        for prop in ['logP'] + interested_props:
            stat_plot = {}
            stat_plot['source'] = src_prop[prop]
            
            for method in tf_methods:
                df = pd.read_csv(os.path.join(prop_folder, f'{method}_0.csv'))
                stat_plot[method] = df[prop]
            
            stat_plot = pd.DataFrame(stat_plot)
            kde_plot(stat_plot, os.path.join(fig_folder, f'tf-{prop}.png'),
                     xlabel=prop, ylabel='Density', xlimit=prop_xlimit[prop])