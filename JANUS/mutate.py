import random
import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import dask.dataframe as dd
import selfies
from selfies import encoder, decoder

# Updated SELFIES constraints: 
default_constraints = selfies.get_semantic_constraints()
new_constraints = default_constraints
new_constraints['S'] = 2
new_constraints['P'] = 3
selfies.set_semantic_constraints(new_constraints)  # update constraints


def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


def mutate_sf(sf_chars): 
    '''
    Provided a list of SELFIE characters, this function will return a modified 
    SELFIES. 
    '''
    random_char_idx = random.choice(range(len(sf_chars)))
    choices_ls = [1, 2, 3] # TODO: 1 = mutate; 2 = addition; 3=delete
    mutn_choice = choices_ls[random.choice(range(len(choices_ls)))] # Which mutation to do: 
    alphabet = ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]'] +  ['[C][=C][C][=C][C][=C][Ring1][Branch1_2]']*2 

    # Mutate character: 
    if mutn_choice == 1: 
        random_char = alphabet[random.choice(range(len(alphabet)))]
        change_sf  = sf_chars[0:random_char_idx] + [random_char] + sf_chars[random_char_idx+1: ]
        
    # add character: 
    elif mutn_choice == 2: 
        random_char = alphabet[random.choice(range(len(alphabet)))]
        change_sf  = sf_chars[0:random_char_idx] + [random_char] + sf_chars[random_char_idx: ]
        
    # delete character: 
    elif mutn_choice == 3: 
        if len(sf_chars) != 1: 
            change_sf  = sf_chars[0:random_char_idx] + sf_chars[random_char_idx+1: ]
        else: 
            change_sf = sf_chars
            
    return ''.join(x for x in change_sf)



def get_mutated_smile(smiles, num_random_samples, num_mutations): 
    
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    
    # Obtain randomized orderings of the SMILES: 
    randomized_smile_orderings = []
    for _ in range(num_random_samples): 
        randomized_smile_orderings.append(rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True))
    
    # Convert all the molecules to SELFIES
    selfies_ls = [encoder(x) for x in randomized_smile_orderings]
    selfies_ls_chars = [get_selfie_chars(selfie) for selfie in selfies_ls]
    
    # Obtain the mutated selfies
    mutated_sf    = []
    
    for sf_chars in selfies_ls_chars: 
        
        for i in range(num_mutations): 
            if i == 0:  mutated_sf.append(mutate_sf(sf_chars))
            else:       mutated_sf.append(mutate_sf ( get_selfie_chars(mutated_sf[-1]) ))
            
    mutated_smiles = [decoder(x) for x in mutated_sf]
    mutated_smiles_canon = []
    for item in mutated_smiles: 
        smi_canon = Chem.MolToSmiles(Chem.MolFromSmiles(item, sanitize=True), isomericSmiles=False, canonical=True)
        if len(smi_canon) <= 100: # Size restriction! 
            mutated_smiles_canon.append(smi_canon)
        
    mutated_smiles_canon = list(set(mutated_smiles_canon))
    return mutated_smiles_canon


def calc_r5(smiles):
    
    mut_smi = get_mutated_smile(smiles, num_random_samples=10, num_mutations=100)
    
    return mut_smi


def df_r5(df_in):
    return df_in.SMILES.apply(calc_r5)




def get_mutated_smiles(smiles): 
    
    df = pd.DataFrame(smiles, columns=['SMILES'])
    
    #num_cores = 12 # TODO: maximize this! 
    #ddf = dd.from_pandas(df, npartitions=num_cores)

    df['property'] = [calc_r5(i) for i in df['SMILES']]#ddf.map_partitions(df_r5, meta='float').compute(scheduler='processes')
    map_       = df.set_index('SMILES').to_dict()['property']

    return map_
    

