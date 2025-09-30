########################################################################################################################
# structure.py
# 2024.09.07 -- 2024.10.24 -- 2024.11.08 -- 2024.11.11


########################################################################################################################
# SECTION 1: Import packages, arguments; create output directories

# 1.1 Packages
from ase.io import read
import spglib
import numpy as np
import os
import sys
import math

# 1.2 Material ID (Use the command-line argument if provided; otherwise manually) and CIF file path
material_id_manualy_enter = 'mp-755628'
material_id = sys.argv[1] if len(sys.argv) > 1 else material_id_manualy_enter
cif_file_path = f'input/{material_id}.cif'                   

# 1.3 Create output directories
output_sub1 = f'output/{material_id}/structure'; os.makedirs(output_sub1, exist_ok=True)  

# 1.4 Set the initial magnetic moments automatically
magnetic_elements = {
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ce', 
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 
    'Tm', 'Yb'} # 22 magnetic elements  
initial_magnetic_moments = {       
    'Ti': 0.5,  'V': 2.0,  'Cr': 3.0,  'Mn': 5.0,  'Fe': 2.2, 'Co': 1.7, 'Ni': 0.6,  'Cu': 0.1,  'Zn': 0.1,  'Ce': 2.5,
    'Pr': 3.2, 'Nd': 3.5,  'Pm': 4.0,  'Sm': 0.8,  'Eu': 7.0, 'Gd': 7.0, 'Tb': 9.0,  'Dy': 10.0, 'Ho': 10.0, 'Er': 9.0,
    'Tm': 7.0, 'Yb': 4.0} # A dictionary of values

# 1.5 Write in 
output_file_path = f'{output_sub1}/{material_id}_structure.txt'
with open(output_file_path, 'w') as f:  # write mode (overwrite or create)
    f.write("SECTION 1: Import")
    f.write(f"\nMaterial Name: {material_id}\n")


########################################################################################################################
# SECTION 2: atoms, the Atoms object

# 2.1 Atoms object from reading CIF file
atoms = read(cif_file_path)   

# 2.2 Three important properties for later spglib analysis
lattice                 = atoms.get_cell()                       # three row vectors in cartesian coordinates
positions               = atoms.get_scaled_positions(wrap=True)  # fractional coordinates, wrapped within the unit cell, keeping them between 0 and 1
atomic_numbers          = atoms.get_atomic_numbers()

# 2.3 Other properties
chemical_symbols        = atoms.get_chemical_symbols() 
positions_cartesian     = atoms.get_positions()
cell_lengths_and_angles = atoms.cell.cellpar()
number_of_atoms         = len(atoms) 
cell_chemical_formula   = atoms.get_chemical_formula()  # for structure, alphabetical order  
cell_volume             = atoms.get_volume() 
pbc                     = atoms.get_pbc()
# Note: atoms[i], atoms[i].index, atoms[i].symbol
# Note: atoms[i].tag: tag (integer) for atoms by user, default 0 for all atoms

# 2.4 Write in
with open(output_file_path, 'a') as f:  # append mode 
    f.write("\n\nSECTION 2: Atoms object from reading CIF file")
    f.write(f"\nLattice:\n{lattice}\n")
    f.write(f"\nPositions:\n{positions}\n")
    f.write(f"\nAtomic numbers: {atomic_numbers}\n")
    
    f.write(f"\nChemical symbols: {chemical_symbols}\n")
    f.write(f"\nCartesian positions:\n{positions_cartesian}\n")
    f.write(f"\nCell lengths and angles: {cell_lengths_and_angles}\n")
    f.write(f"\nNumber of atoms: {number_of_atoms}\n")
    f.write(f"\nCell chemical formula: {cell_chemical_formula}\n")
    f.write(f"\nCell volume: {cell_volume}\n")
    f.write(f"\nPeriodic boundary conditions (PBC): {pbc}\n")


########################################################################################################################
# SECTION 3: Symmetry analysis using spglib

# Search symmetry dataset from an input cell with doing standardization. 
# spglib.get_symmetry is for without standardization, see p3 of "find wyckoff positions note".
# Default values: symprec=1e-5, angle_tolerance=-1.0, hall_number=0
cell_spglib = (lattice, positions, atomic_numbers)  # Note: magmoms as the fourth parameter in advanced usage
symmetry_dataset = spglib.get_symmetry_dataset(cell_spglib, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)  

space_group_symbol    = symmetry_dataset.international
space_group_number    = symmetry_dataset.number
point_group           = symmetry_dataset.pointgroup

choice                = symmetry_dataset.choice
transformation_matrix = symmetry_dataset.transformation_matrix  # Transformation matrix from input lattice to standardized lattice
origin_shift          = symmetry_dataset.origin_shift   # Origin shift from standardized to input origin. input_origin=standardized_origin+origin_shift
lattice_std           = symmetry_dataset.std_lattice    # Basis vectors a, b, c of a standardized cell in row vectors. Right-angle coordinates.
positions_std         = symmetry_dataset.std_positions  # Positions of atoms in the standardized cell in fractional coordinates.
atomic_numbers_std    = symmetry_dataset.std_types      # Standardized atomic numbers

rotations             = symmetry_dataset.rotations      # Matrix (rotation) parts of space group operations. Note, not the same order as ITA.
translations          = symmetry_dataset.translations   # Vector (translation) parts of space group operations.
operations            = [(r, t) for r, t in zip(rotations, translations)]  # rotation-translation pair
wyckoff_letters       = symmetry_dataset.wyckoffs       # Wyckoff letters corresponding to the space group type.
site_symmetry_symbols = symmetry_dataset.site_symmetry_symbols # Site symmetry symbols corresponding to the space group type.
equivalent_atoms      = symmetry_dataset.equivalent_atoms # the group identifiers are the indices of the first atom in each symmetry-equivalent group
#crystallographic_orbits = symmetry_dataset.crystallographic_orbits # see "find wyckoff positions note", not trusted

# get crystal system
def get_crystal_system(space_group_number):
    if 1 <= space_group_number <= 2:
        return "Triclinic"
    elif 3 <= space_group_number <= 15:
        return "Monoclinic"
    elif 16 <= space_group_number <= 74:
        return "Orthorhombic"
    elif 75 <= space_group_number <= 142:
        return "Tetragonal"
    elif 143 <= space_group_number <= 167:
        return "Trigonal"
    elif 168 <= space_group_number <= 194:
        return "Hexagonal"
    elif 195 <= space_group_number <= 230:
        return "Cubic"
    else:
        return "Unknown"
crystal_system = get_crystal_system(space_group_number)

with open(output_file_path, 'a') as f:  # append mode
    f.write("\n\n\n\nSECTION 3: Symmetry analysis using spglib")
    
    f.write(f"\nSpace group symbol: {space_group_symbol}\n")
    f.write(f"\nSpace group number: {space_group_number}\n")
    f.write(f"\nPoint group: {point_group}\n")

    f.write(f"\nChoice: {choice}\n")
    f.write(f"\nTransformation matrix:\n{transformation_matrix}\n")
    f.write(f"\nOrigin shift: {origin_shift}\n")
    f.write(f"\nStandardized lattice:\n{lattice_std}\n")
    f.write(f"\nStandardized positions:\n{positions_std}\n")
    f.write(f"\nStandardized atomic numbers: {atomic_numbers_std}\n")
    
    for i, (rotation, translation) in enumerate(operations):
        f.write(f"\nOperation {i + 1}: \nRotation matrix = \n{rotation}, \nTranslation vector = {translation}\n")
    f.write(f"\nWyckoff letters: {wyckoff_letters}\n")
    f.write(f"\nSite symmetry symbols: {site_symmetry_symbols}\n")
    f.write(f"\nEquivalent atoms (group identifiers): {equivalent_atoms}\n")
    f.write(f"\nCrystal system: {crystal_system}")


########################################################################################################################
# SECTION 4: Wyckoff positions

# key: equivalent atom group identifier (first atom index); values: another dictionary with letter and count as two keys
equivalent_atom_groups = {}  
for atom_index, equivalent_atom_id in enumerate(equivalent_atoms):
    wyckoff_letter = wyckoff_letters[atom_index]  
    if equivalent_atom_id not in equivalent_atom_groups:
        equivalent_atom_groups[equivalent_atom_id] = {'letter': wyckoff_letter, 'count': 1}  
    else:
        equivalent_atom_groups[equivalent_atom_id]['count'] += 1

# Group atoms by equivalent_atom_id
atom_groups = {}
# Populate atom_groups dictionary with atoms grouped by equivalent_atom_id
for atom_index, equivalent_atom_id in enumerate(equivalent_atoms):
    if equivalent_atom_id not in atom_groups:
        atom_groups[equivalent_atom_id] = []  # Initialize an empty list for new groups
    symbol         = chemical_symbols[atom_index]
    multiplicity   = equivalent_atom_groups[equivalent_atom_id]['count']
    wyckoff_letter = equivalent_atom_groups[equivalent_atom_id]['letter']
    site_symmetry  = site_symmetry_symbols[atom_index]
    frac_pos       = positions[atom_index]  
    # Append the atom's information to the corresponding group
    atom_info      = f"{symbol}{atom_index}: {multiplicity}{wyckoff_letter}, {site_symmetry}, {frac_pos}"
    atom_groups[equivalent_atom_id].append(atom_info)

with open(output_file_path, 'a') as f:
    f.write("\n\n\n\nSECTION 4: Wyckoff positions")
    f.write("\nEquivalent Atom Groups Dictionary:")
    for key, value in equivalent_atom_groups.items():
        f.write(f"\nEquivalent atom group:  {key} (first atom index), Wyckoff letter: {value['letter']}, Count: {value['count']}\n")
    f.write("\nAtoms grouped by equivalent atom ID:")
    for group_id, atom_info_list in atom_groups.items():  
        f.write(f"\nEquivalent atom group (ID: {group_id}):")
        for atom_info in atom_info_list:  
            f.write(f"\n{atom_info}")


########################################################################################################################
# SECTION 5: Inversion-related pairs and centering-translation-related pairs

# find the first magnetic symbol in chemical_symbols that is also in the magnetic_elements set.
magnetic_symbol = next(i for i in chemical_symbols if i in magnetic_elements)  # note this was above already,
# for understanding of here, I keep it here.
magnetic_atom_indices = [i for i, symbol in enumerate(chemical_symbols) if symbol == magnetic_symbol]  
tolerance = 1e-5

# Find inversion-related pairs
inversion_pairs = []
for i in range(len(magnetic_atom_indices)):
    idx_i = magnetic_atom_indices[i]
    pos_i = positions[idx_i]    

    if space_group_number in [48, 86, 126, 133, 134, 137, 138, 201, 222, 224] and choice == '1':
        inverted_pos_i = np.array([-pos_i[0] + 0.5, -pos_i[1] + 0.5, -pos_i[2] + 0.5]) % 1 

    elif space_group_number in [50, 59, 85, 125, 129, 130] and choice == '1':
        inverted_pos_i = np.array([-pos_i[0] + 0.5, -pos_i[1] + 0.5, -pos_i[2]]) % 1 

    elif space_group_number in [68] and choice == '1':
        inverted_pos_i = np.array([-pos_i[0], -pos_i[1] + 0.5, -pos_i[2] + 0.5]) % 1    

    elif space_group_number in [70, 203, 227] and choice == '1':
        inverted_pos_i = np.array([-pos_i[0] + 0.25, -pos_i[1] + 0.25, -pos_i[2] + 0.25]) % 1    

    elif space_group_number in [88, 141, 142] and choice == '1':
        inverted_pos_i = np.array([-pos_i[0], -pos_i[1] + 0.5, -pos_i[2] + 0.25]) % 1    

    elif space_group_number in [228] and choice == '1':
        inverted_pos_i = np.array([-pos_i[0] + 0.75, -pos_i[1] + 0.75, -pos_i[2] + 0.75]) % 1 

    else:
        inverted_pos_i = (-pos_i) % 1  # Apply inversion operation: (-x, -y, -z) 

    for j in range(i+1, len(magnetic_atom_indices)):  # Start from i+1 to avoid duplicate pairs    
        idx_j = magnetic_atom_indices[j]
        pos_j = positions[idx_j]
        if all(abs(inverted_pos_i - pos_j) < tolerance):
            inversion_pairs.append((idx_i, idx_j))

with open(output_file_path, 'a') as f:  # append mode
    f.write("\n\n\n\nSECTION 5: Inversion-related pairs and centering-translation-related pairs")
    f.write(f"\nPairs of {magnetic_symbol} atoms related by inversion symmetry:")
    if inversion_pairs:
        for pair in inversion_pairs:
            idx_i, idx_j = pair
            f.write(f"\n({magnetic_symbol}{idx_i}, {magnetic_symbol}{idx_j})")
    else:
        f.write(f"\nNo {magnetic_symbol} atoms are related by inversion symmetry.\n")

# Add print statements for terminal output
print("\n\n\n\nSECTION 5: Inversion-related pairs and centering-translation-related pairs")
print(f"Pairs of {magnetic_symbol} atoms related by inversion symmetry:")
if inversion_pairs:
    for pair in inversion_pairs:
        idx_i, idx_j = pair
        print(f"({magnetic_symbol}{idx_i}, {magnetic_symbol}{idx_j})")
else:
    print(f"No {magnetic_symbol} atoms are related by inversion symmetry.\n")

# Determine centering translations based on the first letter of the space group symbol
centering_vectors = []
if space_group_symbol.startswith('I'):    # Body-centered 
    centering_vectors.append(np.array([0.5, 0.5, 0.5]))
elif space_group_symbol.startswith('F'):  # Face-centered 
    centering_vectors.extend([
        np.array([0.5, 0, 0]),
        np.array([0, 0.5, 0]),
        np.array([0, 0, 0.5])])
elif space_group_symbol.startswith('A'):  # A-centered
    centering_vectors.append(np.array([0, 0.5, 0.5]))
elif space_group_symbol.startswith('B'):  # B-centered
    centering_vectors.append(np.array([0.5, 0, 0.5]))
elif space_group_symbol.startswith('C'):  # C-centered
    centering_vectors.append(np.array([0.5, 0.5, 0]))
elif space_group_symbol.startswith('R'):  # Rhombohedral
    centering_vectors.extend([
        np.array([2/3, 1/3, 1/3]),
        np.array([1/3, 2/3, 2/3])])
else:  # Primitive ('P')
    centering_vectors.append(np.array([0, 0, 0]))

# Write centering vectors to the file
with open(output_file_path, 'a') as f:
    f.write(f"\nCentering vectors for space group {space_group_symbol}:\n")
    for idx, vector in enumerate(centering_vectors):
        f.write(f"Centering vector {idx + 1}: {vector}\n")

# Find centering-translation-related pairs
centering_translation_pairs = []
for centering_vector in centering_vectors:
    for i in range(len(magnetic_atom_indices)):
        idx_i = magnetic_atom_indices[i]
        pos_i = positions[idx_i]

        for j in range(i + 1, len(magnetic_atom_indices)):
            idx_j = magnetic_atom_indices[j]
            pos_j = positions[idx_j]
            
            # Check specific centering translation
            translated_pos_i = (pos_i + centering_vector) % 1
            if all(abs(translated_pos_i - pos_j) < tolerance):
                centering_translation_pairs.append((idx_i, idx_j, centering_vector))

with open(output_file_path, 'a') as f:  
    f.write(f"\nPairs of {magnetic_symbol} atoms related by centering translations:")
    if centering_translation_pairs:
        for pair in centering_translation_pairs:
            idx_i, idx_j, vector = pair
            f.write(f"\n({magnetic_symbol}{idx_i}, {magnetic_symbol}{idx_j}) with centering vector: {vector}")
    else:
        f.write(f"\nNo {magnetic_symbol} atoms are related by any centering translations.")

# Add print statements for terminal output
print(f"\nPairs of {magnetic_symbol} atoms related by centering translations:")
if centering_translation_pairs:
    for pair in centering_translation_pairs:
        idx_i, idx_j, vector = pair
        print(f"({magnetic_symbol}{idx_i}, {magnetic_symbol}{idx_j}) with centering vector: {vector}")
else:
    print(f"No {magnetic_symbol} atoms are related by any centering translations.")


########################################################################################################################
# SECTION 6: FM

m = initial_magnetic_moments.get(magnetic_symbol)
magmoms_fm = [0.0] * len(atoms)   
for idx in magnetic_atom_indices:  
    magmoms_fm[idx] = m

with open(output_file_path, 'a') as f:  # append mode
    f.write("\n\n\n\nSECTION 6: FM")
    for idx in magnetic_atom_indices:
        f.write(f"\n{chemical_symbols[idx]}{idx}: {magmoms_fm[idx]}")
        f.write(f"\nInitial magnetic moment m = {m}\n")


########################################################################################################################
# SECTION 7: AFM1 (Kramers degenerate type)

from itertools import product

# 7.1 Generate configurations for inversion-related pairs with opposite moment signs
inversion_configs = []

# Check if there are any inversion-related pairs at the beginning
with open(output_file_path, 'a') as f:  # append mode
    f.write("\n\n\n\nSECTION 7: AFM1 (Kramers degenerate type)")
if not inversion_pairs:
    with open(output_file_path, 'a') as f:  # append mode
        f.write("\nNo inversion-related pairs found\n")
else:
    for signs in product([1, -1], repeat=len(inversion_pairs)):
        magmoms = [0.0] * len(atoms)
        for idx, (idx_i, idx_j) in enumerate(inversion_pairs):
            magmoms[idx_i] = signs[idx] * m
            magmoms[idx_j] = -signs[idx] * m
        
        # Apply "Always Keep the Positive Version" convention
        if magmoms[magnetic_atom_indices[0]] < 0:
            magmoms = list(np.negative(magmoms))  # Flip the signs if the first moment is negative       
        # Check if the configuration or its inverse is already in the list
        if magmoms not in inversion_configs:
            inversion_configs.append(magmoms)

    # Print all distinct configurations for inversion pairs
    with open(output_file_path, 'a') as f:  # append mode
        f.write("\nAll distinct (discarding inverse) configurations for inversion-related pairs with opposite moment signs:")
        for idx, config in enumerate(inversion_configs):
            f.write(f"\nInversion Configuration {idx + 1}: {config}\n")

# 7.2 Generate configurations for centering translation pairs (same sign)
centering_configs = []

# Check if there are any centering translation-related pairs at the beginning
if not centering_translation_pairs:
    with open(output_file_path, 'a') as f:  # append mode
        f.write("\nNo centering-translation-related pairs found\n")
else:
    for signs in product([1, -1], repeat=len(centering_translation_pairs)):
        magmoms = [0.0] * len(atoms)
        for idx, (idx_i, idx_j, _) in enumerate(centering_translation_pairs):
            magmoms[idx_i] = signs[idx] * m
            magmoms[idx_j] = signs[idx] * m

        # Apply "Always Keep the Positive Version" convention
        if magmoms[magnetic_atom_indices[0]] < 0:
            magmoms = list(np.negative(magmoms))  # Flip the signs if the first moment is negative       
        # Check if the configuration or its inverse is already in the list
        if magmoms not in centering_configs:
            centering_configs.append(magmoms)

    # Print all distinct configurations for centering translation pairs
    with open(output_file_path, 'a') as f:  # append mode
        f.write("\nAll distinct (discarding inverse) configurations for centering-translation-related pairs with same sign:")
        for idx, config in enumerate(centering_configs):
            f.write(f"\nCentering Configuration {idx + 1}: {config}\n")

# 7.3 Find common configurations that satisfy both conditions: 
# (1) inversion_configs: opposite signs for inversion-related pairs;
# (2) centering_configs: same sign for centering-translation-related pairs
valid_configs = []
# If there are centering-related pairs, combine inversion and centering configurations
if centering_configs:
    for inv_conf in inversion_configs:
        for cent_conf in centering_configs:
            combined_conf = [0.0] * len(atoms)
            valid = True
            for i in magnetic_atom_indices:
                if inv_conf[i] != 0 and cent_conf[i] != 0:
                    if inv_conf[i] != cent_conf[i]:  # Inconsistent signs, invalid configuration
                        valid = False
                        break
                    combined_conf[i] = inv_conf[i]  # Both inversion and centering match
                elif inv_conf[i] != 0:
                    combined_conf[i] = inv_conf[i]  # Only inversion has non-zero value
                elif cent_conf[i] != 0:
                    combined_conf[i] = cent_conf[i]  # Only centering has non-zero value
            if valid:
                valid_configs.append(combined_conf)  # Store valid configuration
else:
    # Only inversion-related pairs exist, so use inversion configurations directly
    valid_configs = inversion_configs

# Write valid configurations to the file
with open(output_file_path, 'a') as f:
    if valid_configs:
        f.write("\nValid configurations (inversion-related condition or both inversion and centering conditions:\n")
        for idx, config in enumerate(valid_configs):
            f.write(f"Configuration {idx + 1}: {config}\n")
    else:
        f.write("\nNo valid configurations found\n")

# 7.4 Choose a valid AFM1 configuration
# Filter out configurations where the sum of the moments is not zero
filtered_valid_configs = [config for config in valid_configs if sum(config) == 0]
# Write all filtered valid configurations to the file
with open(output_file_path, 'a') as f:
    if filtered_valid_configs:
        f.write("\nValid AFM1 configurations:\n")
        for idx, config in enumerate(filtered_valid_configs):
            f.write(f"Filtered Configuration {idx + 1}: {config}\n")
    else:
        f.write("\nNo valid AFM1 configuration\n")

magmoms_afm1 = filtered_valid_configs[0] if filtered_valid_configs else None

if magmoms_afm1:
    with open(output_file_path, 'a') as f:
        f.write("\nAFM1 configuration:\n")
        for idx in magnetic_atom_indices:
            f.write(f"{chemical_symbols[idx]}{idx}: {magmoms_afm1[idx]}\n")   
else:
    with open(output_file_path, 'a') as f:
        f.write("\nNo valid AFM1 configuration found\n")

# 7.5 Assign remaining configurations as AFM1a, AFM1b, etc.
magmoms_afm1a = filtered_valid_configs[1] if len(filtered_valid_configs) > 1 else None
magmoms_afm1b = filtered_valid_configs[2] if len(filtered_valid_configs) > 2 else None
# Write out the additional configurations if they exist
with open(output_file_path, 'a') as f:
    if magmoms_afm1a:
        f.write("\nConfiguration for AFM1a magnetic moments:\n")
        for idx in magnetic_atom_indices:
            f.write(f"{chemical_symbols[idx]}{idx}: {magmoms_afm1a[idx]}\n")

    if magmoms_afm1b:
        f.write("\nConfiguration for AFM1b magnetic moments:\n")
        for idx in magnetic_atom_indices:
            f.write(f"{chemical_symbols[idx]}{idx}: {magmoms_afm1b[idx]}\n")


########################################################################################################################
# SECTION 8: Generic AFM configurations (AFM2, AFM3, ...)

from itertools import product

# Step 1: Generate all possible configurations for magnetic atoms
all_possible_configs = []

# Iterate over all combinations of -m and +m for each magnetic atom
for signs in product([1, -1], repeat=len(magnetic_atom_indices)):
    magmoms = [0.0] * len(atoms)
    for i, sign in enumerate(signs):
        magmoms[magnetic_atom_indices[i]] = sign * m
    all_possible_configs.append(magmoms)

# Save all possible configurations to a file
with open(output_file_path, 'a') as f:
    f.write("\n\n\n\nSECTION 8: Generic AFM configurations (AFM2, AFM3, ...)\n")
    f.write("\nAll Possible Configurations\n")
    for idx, config in enumerate(all_possible_configs):
        f.write(f"Configuration {idx + 1}: {config}\n")

# Step 2: Discard inverse configurations using "Always Keep the Positive Version"
distinct_configs = []
for config in all_possible_configs:
    # Apply "Always Keep the Positive Version" convention
    if config[magnetic_atom_indices[0]] < 0:
        config = list(np.negative(config))  # Flip the signs if the first moment is negative   
    # Only keep distinct configurations, discarding inverse
    if config not in distinct_configs:
        distinct_configs.append(config)

# Save all distinct configurations to a file
with open(output_file_path, 'a') as f:
    f.write("\nAll Distinct Configurations (discarding inverse)\n")
    for idx, config in enumerate(distinct_configs):
        f.write(f"Configuration {idx + 1}: {config}\n")

# Step 3: Exclude FM and AFM1 after discarding inverse configurations
with open(output_file_path, 'a') as f:
    f.write("\nFM configuration:")
    f.write(f"{magmoms_fm}")
    
    f.write("\nAFM1 configuration:")
    if magmoms_afm1:
        f.write(f"{magmoms_afm1}")
    else:
        f.write("\nNo valid AFM1 configuration found.")

# Filter out FM, AFM1, AFM1a, and AFM1b from distinct_configs
filtered_configs = []
for config in distinct_configs:
    if config != magmoms_fm and config != magmoms_afm1 and config != magmoms_afm1a and config != magmoms_afm1b:
        filtered_configs.append(config)

# Save the filtered configurations (excluding FM and AFM1) to a file
with open(output_file_path, 'a') as f:
    if filtered_configs:
        f.write("\n\nFiltered configurations (excluding FM and AFM1):\n")
        for idx, config in enumerate(filtered_configs):
            f.write(f"Configuration {idx + 1}: {config}\n")
    else:
        f.write("\nNo valid configurations found (after excluding FM and AFM1).\n")

# Step 4: Filter out configurations where the sum of the moments is not zero
filtered_afm_configs = [config for config in filtered_configs if sum(config) == 0]
# Save the AFM configurations to the file
with open(output_file_path, 'a') as f:
    if filtered_afm_configs:
        f.write("\nFiltered configurations for AFM2 and beyond (excluding non-AFM types):\n")
        for idx, config in enumerate(filtered_afm_configs):
            f.write(f"Configuration {idx + 1}: {config}\n")
    else:
        f.write("\nNo valid AFM configurations found\n")

# Step 5: Choose AFM configurations from the valid AFM configurations
chosen_afm_configs = filtered_afm_configs[:2]  # Modify this number to handle more AFMs
afm_names = ['AFM2', 'AFM3']  # Add more names as needed

# Initialize the variables for use later in visualize.py
magmoms_afm2, magmoms_afm3 = None, None

# Store the chosen configurations into their respective variables and write them to the output file
with open(output_file_path, 'a') as f:
    for idx, config in enumerate(chosen_afm_configs):
        # Directly assign the chosen configurations to the correct variables
        if idx == 0:
            magmoms_afm2 = config
        elif idx == 1:
            magmoms_afm3 = config

        # Write the chosen configuration to the output file
        f.write(f"\nChosen configuration for {afm_names[idx]}:\n")
        f.write(f"{config}\n")
    
    # Handle cases where no valid configuration is found for AFM2 or AFM3
    if len(chosen_afm_configs) < 2:
        for idx in range(len(chosen_afm_configs), len(afm_names)):
            f.write(f"\nNo valid configuration found for {afm_names[idx]}.\n")


########################################################################################################################
# SECTION 9: Supercell and Magnetic Configurations 

num_magnetic_atoms_unit_cell = len(magnetic_atom_indices)

if num_magnetic_atoms_unit_cell == 2:
    # Step 1: Create the supercell
    supercell_dimensions = (1, 1, 2)
    supercell = atoms * supercell_dimensions  

    # Step 2: Identify magnetic atom indices in the supercell
    magnetic_atom_indices_supercell = [i for i, symbol in enumerate(supercell.get_chemical_symbols()) if symbol == magnetic_symbol]


    # Step 3: Define FM in the supercell by replicating magmoms_fm from the original unit cell
    magmoms_fm_supercell = magmoms_fm * 2  # Duplicate the FM configuration to fill both halves of the supercell   

    # Step 4: Define AFM1 in the supercell using magmoms_afm1 from the original unit cell
    if magmoms_afm1 is not None:
        magmoms_afm1_supercell = magmoms_afm1 * 2  # Replicate the AFM1 pattern for each half of the supercell
    else:
        print("magmoms_afm1 is not defined.")
        magmoms_afm1_supercell = None  # Assign a default or empty list to avoid errors later

    # Step 5: Generate additional configurations (AFM2 and AFM3) for the supercell
    from itertools import product
    all_possible_configs_supercell = []

    for signs in product([1, -1], repeat=len(magnetic_atom_indices_supercell)):
        magmoms = [0.0] * len(supercell)
        for i, sign in enumerate(signs):
            magmoms[magnetic_atom_indices_supercell[i]] = sign * m
        all_possible_configs_supercell.append(magmoms)

    # Step 6: Exclude configurations that match FM and AFM1 in the supercell to define AFM2 and AFM3
    distinct_configs_supercell = []
    for config in all_possible_configs_supercell:
        if config != magmoms_fm_supercell and config != magmoms_afm1_supercell:
            # Apply "Always Keep the Positive Version" convention
            if config[magnetic_atom_indices_supercell[0]] < 0:
                config = list(np.negative(config))  # Flip the signs if the first moment is negative
            # Only keep distinct configurations, discarding inverse
            if config not in distinct_configs_supercell:
                distinct_configs_supercell.append(config)

    # Step 7: Discard inverse configurations using "Always Keep the Positive Version"
    distinct_configs_supercell = []
    for config in all_possible_configs_supercell:
        # Apply "Always Keep the Positive Version" convention
        if config[magnetic_atom_indices_supercell[0]] < 0:
            config = list(np.negative(config))  # Flip the signs if the first moment is negative
        # Only keep distinct configurations, discarding inverse
        if config not in distinct_configs_supercell:
            distinct_configs_supercell.append(config)

    # Step 8: Filter out configurations that match FM and AFM1 in the supercell
    filtered_configs_supercell = []
    for config in distinct_configs_supercell:
        # Exclude configurations that match FM or AFM1 in the supercell
        if config != magmoms_fm_supercell and config != magmoms_afm1_supercell:
            filtered_configs_supercell.append(config)

    # Step 9: Filter out configurations where the sum of the moments is not zero
    filtered_afm_configs_supercell = [config for config in filtered_configs_supercell if sum(config) == 0]

    # Select the first two valid configurations as AFM2 and AFM3
    magmoms_afm2_supercell = filtered_afm_configs_supercell[0] if len(filtered_afm_configs_supercell) > 0 else None
    magmoms_afm3_supercell = filtered_afm_configs_supercell[1] if len(filtered_afm_configs_supercell) > 1 else None

    # Step 10: Write All Configurations to Output File
    with open(output_file_path, 'a') as f:
        f.write("\n\n\n\nSECTION 9: Supercell and Magnetic Configurations\n")

        # Write Supercell Details
        f.write(f"Supercell created with {supercell_dimensions} expansion\n")
        f.write(f"Number of atoms in supercell: {len(supercell)}\n")
        f.write(f"Magnetic atom indices in supercell: {magnetic_atom_indices_supercell}\n")
        
        # Write FM Configuration in Supercell
        f.write("\nFM Configuration in Supercell:\n")
        for idx in magnetic_atom_indices_supercell:
            f.write(f"{supercell[idx].symbol}{idx}: {magmoms_fm_supercell[idx]}\n")
        
        # Write AFM1 Configuration in Supercell
        f.write("\nAFM1 Configuration in Supercell:\n")
        if magmoms_afm1_supercell:
            for idx in magnetic_atom_indices_supercell:
                f.write(f"{supercell[idx].symbol}{idx}: {magmoms_afm1_supercell[idx]}\n")
        else:
            print("Skipping write: `magmoms_afm1_supercell` is not defined.")

        # Write All Distinct Configurations in Supercell
        f.write("\nAll Distinct Configurations in Supercell (after discarding inverse):\n")
        for idx, config in enumerate(distinct_configs_supercell):
            f.write(f"Configuration {idx + 1}: {config}\n")

        # Write Filtered Configurations (excluding FM and AFM1)
        f.write("\nFiltered Configurations in Supercell (excluding FM and AFM1):\n")
        for idx, config in enumerate(filtered_configs_supercell):
            f.write(f"Configuration {idx + 1}: {config}\n")

        # Write Configurations with Zero Net Moment
        f.write("\nFiltered Configurations with Zero Net Moment (AFM Condition):\n")
        for idx, config in enumerate(filtered_afm_configs_supercell):
            f.write(f"Configuration {idx + 1}: {config}\n")

        # Write AFM2 and AFM3 Configurations if available
        if magmoms_afm2_supercell:
            f.write("\nChosen Configuration for AFM2 in Supercell:\n")
            for idx in magnetic_atom_indices_supercell:
                f.write(f"{supercell[idx].symbol}{idx}: {magmoms_afm2_supercell[idx]}\n")
        else:
            f.write("\nNo valid configuration found for AFM2 in Supercell.\n")
        
        if magmoms_afm3_supercell:
            f.write("\nChosen Configuration for AFM3 in Supercell:\n")
            for idx in magnetic_atom_indices_supercell:
                f.write(f"{supercell[idx].symbol}{idx}: {magmoms_afm3_supercell[idx]}\n")
        else:
            f.write("\nNo valid configuration found for AFM3 in Supercell.\n")


########################################################################################################################
# SECTION

print("\nspace group number:", space_group_number, "\n")
print("choice:", choice, "\n")

magmoms_afm1_mag = [moment for moment in magmoms_afm1 if moment != 0] if magmoms_afm1 is not None else None
magmoms_afm1a_mag = [moment for moment in magmoms_afm1a if moment != 0] if magmoms_afm1a is not None else None
magmoms_afm1b_mag = [moment for moment in magmoms_afm1b if moment != 0] if magmoms_afm1b is not None else None
magmoms_afm2_mag = [moment for moment in magmoms_afm2 if moment != 0] if magmoms_afm2 is not None else None
magmoms_afm3_mag = [moment for moment in magmoms_afm3 if moment != 0] if magmoms_afm3 is not None else None

print(f"Material: {material_id},   AFM1: {magmoms_afm1_mag}")
print(f"Material: {material_id},  AFM1a: {magmoms_afm1a_mag}")
print(f"Material: {material_id},  AFM1b: {magmoms_afm1b_mag}")
print(f"Material: {material_id},   AFM2: {magmoms_afm2_mag}")
print(f"Material: {material_id},   AFM3: {magmoms_afm3_mag}")

print("\nmagnetic element:", magnetic_symbol)

# Print the Wyckoff information
print("\nAtoms grouped by equivalent atom ID:")
for group_id, atom_info_list in atom_groups.items():  
    for atom_info in atom_info_list:  
        print(f"{atom_info}")

if num_magnetic_atoms_unit_cell == 2:
    print("\nThis is N2 material!\n")
    magmoms_afm2_supercell_mag = [moment for moment in magmoms_afm2_supercell if moment != 0] if magmoms_afm2_supercell is not None else None
    magmoms_afm3_supercell_mag = [moment for moment in magmoms_afm3_supercell if moment != 0] if magmoms_afm3_supercell is not None else None
    print(f"Material: {material_id},   AFM2: {magmoms_afm2_supercell_mag}")
    print(f"Material: {material_id},   AFM3: {magmoms_afm3_supercell_mag}")

    # Print AFM2 and AFM3 Configurations in Supercell
    print("\nChosen Configuration for AFM2 in Supercell:")
    for idx in magnetic_atom_indices_supercell:
        print(f"{supercell[idx].symbol}{idx}: {magmoms_afm2_supercell[idx]}")

    print("\nChosen Configuration for AFM3 in Supercell:")
    for idx in magnetic_atom_indices_supercell:
        print(f"{supercell[idx].symbol}{idx}: {magmoms_afm3_supercell[idx]}")


# ########################################################################################################################
# # SECTION 10: Magnetic Point Group for AFM1 Configuration

# # Input angles in degrees for magnetic moment orientation
# theta_x = 90  # Angle between magnetic moment and x-axis
# theta_y = 90  # Angle between magnetic moment and y-axis
# theta_z = 0 # Angle between magnetic moment and z-axis

# # Calculate direction cosines
# cos_theta_x = math.cos(math.radians(theta_x))
# cos_theta_y = math.cos(math.radians(theta_y))
# cos_theta_z = math.cos(math.radians(theta_z))

# # Convert scalar moments to vectors along the specified direction
# magmoms_afm1_vectors = [[moment * cos_theta_x, moment * cos_theta_y, moment * cos_theta_z] for moment in magmoms_afm1]

# print(f"\nmagmoms_afm1: {magmoms_afm1}")
# print(f"magmoms_afm1_vectors: {magmoms_afm1_vectors}")

# cell = (lattice, positions, atomic_numbers, magmoms_afm1_vectors)   # Add magmoms

# dataset = spglib.get_magnetic_symmetry_dataset(cell, is_axial=True, symprec=1e-5, angle_tolerance=-1.0, mag_symprec=-1.0)

# # Extract magnetic space group information
# uni_number     = dataset.uni_number  # int The serial number from 1 to 1651 of UNI or BNS symbols
# msg_type       = dataset.msg_type    # int from 1 to 4
# hall_number    = dataset.hall_number # int from 1 to 530, For type-I, II, III, hall number of FSG; for type-IV, that of XSG
# tensor_rank    = dataset.tensor_rank # 0 for collinear spins, 1 for non-collinear spins   # Wu: misleading
# magnetic_spacegroup_type = spglib.get_magnetic_spacegroup_type(uni_number) # Additional extraction for more information

# # Extract magnetic symmetry operations
# n_operations   = dataset.n_operations   # Number of magnetic symmetry operations.
# rotations      = dataset.rotations      # Rotation (matrix) parts of symmetry operations
# translations   = dataset.translations   # Translation (vector) parts of symmetry operations
# time_reversals = dataset.time_reversals # Time reversal part of magnetic symmetry operations. # 1 indicates time reversal operation, and 0 indicates an ordinary operation.
# # Combine rotations and time-reversals to infer the point group
# unique_rotations = set()
# for rotation, time_reversal in zip(rotations, time_reversals):
#     unique_rotations.add((tuple(map(tuple, rotation)), time_reversal))

# # Extract generators of the magnetic point group (仅供参考，不准)
# generators = []
# generated_operations = set()

# for rotation, time_reversal in unique_rotations:
#     if (rotation, time_reversal) not in generated_operations:
#         generators.append((rotation, time_reversal))
#         for gen_rotation, gen_time_reversal in generators:
#             for op_rotation, op_time_reversal in unique_rotations:
#                 combined_rotation = tuple(map(tuple, np.dot(rotation, gen_rotation)))
#                 combined_time_reversal = time_reversal ^ gen_time_reversal
#                 generated_operations.add((combined_rotation, combined_time_reversal))

# # Symmetrically equivalent atoms
# n_atoms                   = dataset.n_atoms
# equivalent_atoms_magnetic = symmetry_dataset.equivalent_atoms

# # Output the magnetic space group and point group information
# print("\nMagnetic Space Group Information:")
# print(f"UNI Number: {uni_number}")
# print(f"msg_type: {msg_type}")
# print(f"Hall Number: {hall_number}")
# print(f"tensor_rank: {tensor_rank}")
# print(f"Magnetic Space Group Type: {magnetic_spacegroup_type}")

# print(f"\nn_operations: {n_operations}")
# print("\nMagnetic Point Group Symmetry Operations:")
# for rotation, time_reversal in unique_rotations:
#     print(f"Rotation Matrix: {rotation}, Time Reversal: {'Yes' if time_reversal else 'No'}")

# print(f"\nGenerators of Magnetic Point Group:")
# for rotation, time_reversal in generators:
#     print(f"Rotation Matrix: {rotation}, Time Reversal: {'Yes' if time_reversal else 'No'}")

# print(f"\nn_atoms: {n_atoms}")
# print(f"equivalent_atoms_magnetic: {equivalent_atoms_magnetic}")


# ########################################################################################################################
# print('\nStructure Saved!\n')
