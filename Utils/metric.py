import numpy as np
from rdkit import RDLogger, rdBase
from rdkit.Chem.rdchem import AtomValenceException
from rdkit.Chem import Descriptors
from openbabel import openbabel, pybel
from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer
from Utils.smiles import get_rdkit_mol


def disable_rdkit_logging():
    """Disable RDKit whiny logging"""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    rdBase.DisableLog('rdApp.error')
disable_rdkit_logging()


def obabel_logP(mol):
    pybelmol = pybel.Molecule(mol)
    descvalues = pybelmol.calcdesc()
    return descvalues['logP']


def logP(mol):
    return Descriptors.MolLogP(mol)


def tPSA(mol):
    return Descriptors.TPSA(mol)


def QED(mol):
    try:
        return Descriptors.qed(mol)
    except AtomValenceException:
        return np.nan


def SAS(mol):
    return sascorer.calculateScore(mol)


def NP(mol):
    return npscorer.scoreMol(mol)


def MW(mol):
    return Descriptors.MolWt(mol)


def NATOM(smi_or_mol, mol_method='obabel'):
    # if mol_method == 'obabel':
    #     mol = get_obabel_mol(smi_or_mol)
    #     return mol.NumHvyAtoms()
    # elif mol_method == 'rdkit':
    mol = get_rdkit_mol(smi_or_mol)
    return mol.GetNumHeavyAtoms()


def MSD(x, trg):
    """mean squared deviation"""
    delv = [x[i] - trg[i] for i in range(len(trg))]
    return sum(delv) / len(delv)


def MAD(x, trg):
    """mean absolute deviation"""
    abs_delv = [abs(x[i] - trg[i]) for i in range(len(trg))]
    return sum(abs_delv) / len(abs_delv)


def MAX(x, trg):
    """max error"""
    delv = [x[i] - trg[i] for i in range(len(trg))]
    return max(delv)    


def MIN(x, trg):
    """min error"""
    delv = [x[i] - trg[i] for i in range(len(trg))]
    return min(delv)    


def SD(x):
    return np.array(x).std()


def get_property_fn(props):
    property_fn = {
        "logP" : logP,
        "tPSA" : tPSA,
        "QED"  : QED,
        "SAS"  : SAS,
        "NP"   : NP,
        "MW"   : MW,
        "NATOM": NATOM
    }
    return { p: property_fn[p] for p in props }


def error_fn(x, trg):
    error = {}
    error['MSD'] = MSD(x, trg)
    error['MAD'] = MAD(x, trg)
    error['MIN'] = MIN(x, trg)
    error['MAX'] = MAX(x, trg)
    error['SD'] = SD(x)
    return error