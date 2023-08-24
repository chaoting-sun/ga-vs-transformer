from rdkit import Chem
# import openbabel
# import pybel
from openbabel import openbabel as ob
# from openbabel import openbabel, pybel


# def get_obabel_mol(smi):
#     obConversion = openbabel.OBConversion()
#     obConversion.SetInAndOutFormats("smi", "can")
#     # smiles as an input, canonical smiles as an output
#     mol = openbabel.OBMol()
#     if isinstance(smi, str) and obConversion.ReadString(mol, smi):
#         return mol
#     return None


def get_obabel_mol(smi_or_mol):
    if isinstance(smi_or_mol, str):
        if len(smi_or_mol) == 0:
            return None
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("smi", "can")
        # smiles as an input, canonical smiles as an output
        mol = ob.OBMol()
        if conv.ReadString(mol, smi_or_mol):
            return mol
    return smi_or_mol


def get_rdkit_mol(smi_or_mol):
    """convert smiles to mol. (copied from molgpt)
    """
    if isinstance(smi_or_mol, str):
        if len(smi_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smi_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smi_or_mol


def get_obabel_smi(mol):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats('mol', 'can')
    smiles = conv.WriteString(mol).strip()
    return smiles


def get_rdkit_smi(mol):
    if mol:
        return Chem.MolToSmiles(mol)
    return mol


get_mol = {
    'rdkit': get_rdkit_mol,
    'obabel': get_obabel_mol
}


get_smi = {
    'rdkit': get_rdkit_smi,
    'obabel': get_obabel_smi    
}
