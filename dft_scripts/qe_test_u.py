# qe_test_u.py
# 2024.10.13 
# 2024.11.28
# 2025.08.05 add DFT+U
# 2025.09.06
# Test ecut(ecutwfc), rf(ecutrho_factor), nk, degauss, conv_thr, special (single point)


########################################################################################################################
# SECTION 1: Import packages, arguments; create sub-output directories

# 1.1 Packages
from structure import * 
from ase.calculators.espresso import Espresso, EspressoProfile
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import sys
import time
import shutil
import json
from mp_api.client import MPRester
import math

from ase.io import write
import subprocess

from ase.io.espresso import read_espresso_out

# 1.2 Extract variables from environment variable
root_dir         = os.getenv("ROOT_DIR")
structure_choice = os.getenv("STRUCTURE_CHOICE")

# 1.3 Extract arguments from command-line arguments 
script_base_name = '_'.join(sys.argv[0].replace('.py', '').split('_')[:2])
material_id      = sys.argv[1]
magnetic_structure = sys.argv[2] 
test_parameter     = sys.argv[3]    

# 1.4 Extract arguments from SLURM
partition     = os.environ["SLURM_JOB_PARTITION"]
nodelist      = os.environ["SLURM_NODELIST"]
ntasks        = os.environ["SLURM_NTASKS"]  
mem_per_cpu   = os.environ["SLURM_MEM_PER_CPU"]
cpus_per_task = os.environ["SLURM_CPUS_PER_TASK"]
slurm_job_id  = os.environ["SLURM_JOB_ID"]  

# 1.5 Create output sub-directories
output_sub1 = f'output/{material_id}';                os.makedirs(output_sub1, exist_ok=True) 
output_sub2 = f'{output_sub1}/{magnetic_structure}';    os.makedirs(output_sub2, exist_ok=True) 
output_sub3 = f'{output_sub2}/{script_base_name}';      os.makedirs(output_sub3, exist_ok=True) 
output_sub4 = f'{output_sub3}/{test_parameter}';        os.makedirs(output_sub4, exist_ok=True) 
output_sub5 = f'{output_sub4}/job{slurm_job_id}';    os.makedirs(output_sub5, exist_ok=True)

# 1.6 Copy scripts to the Slurm job directory for record-keeping
shutil.copy(sys.argv[0],        output_sub5)
shutil.copy("job.sbatch",       output_sub5)

# 1.7 Query the band gap from the Materials Project
api_key = " "  # My API key for Materials Project
with MPRester(api_key) as mpr:
    material_data = mpr.materials.summary.search(material_ids=[material_id], fields=["band_gap"])
    band_gap = material_data[0].band_gap
print(f"Band gap for {material_id}: {band_gap} eV")

# 1.8 Choose atoms or supercell based on STRUCTURE_CHOICE
if structure_choice == "supercell":
    structure = supercell
    print("Using supercell for calculations.")
else:
    structure = atoms
    print("Using atoms (unit cell) for calculations.")

# 1.9 Determine indices of magnetic atoms in the selected structure
magnetic_atom_indices = [i for i, symbol in enumerate(chemical_symbols) if symbol == magnetic_symbol]


########################################################################################################################
# SECTION 2: The ASE atoms object

# 2.1 Set magnetic moments to atoms 
magmoms_dict = {
    "FM":   magmoms_fm,
    "AFM1": magmoms_afm1, 
    "AFM1a": magmoms_afm1a, 
    "AFM1b": magmoms_afm1b,
    "AFM2": magmoms_afm2_supercell if num_magnetic_atoms_unit_cell == 2 else magmoms_afm2,
    "AFM3": magmoms_afm3_supercell if num_magnetic_atoms_unit_cell == 2 else magmoms_afm3,
}
magmoms = magmoms_dict.get(magnetic_structure)

# 2.2 Check if the magnetic moments are None (avoid continuing calculation using default NM when none is met)
if magmoms is None:
    # Raise an error with a meaningful message if the magnetic configuration is not found
    raise ValueError(
        f"Error: The magnetic configuration '{magnetic_structure}' is not available for the material '{material_id}'."
        " Please verify that the configuration exists in 'structure.py'."
    )

# 2.3 Assign the magnetic moments to the atoms
structure.set_initial_magnetic_moments(magmoms)
print(f"Magnetic structure {magnetic_structure} applied successfully.")


########################################################################################################################
# SECTION 3: Test

# 3.1 Profile and PP

# Exchange-correlation functional types:
# (1) LDA: pz, vwn
# (2) GGA: pbe, pbesol, blyp, pw91 (pbe is commonly used for magnetic materials )

# Pseudopotential types: 
# (1) NC: mt, bhs, vbc, rrkj   # accurate
# (2) US: van, rrkjus          # fast (accuracu loss)
# (3) PAW: kjpaw, bpaw         # accurate and fast

# No matter what, SSSP gives a full PBE library with all types of PPs, it is a mixed library, but should be optimized.   

# profile = EspressoProfile(    
#     command    = f'mpirun -np {ntasks} {root_dir}/conda/envs/dft1-env/bin/pw.x',
#     pseudo_dir = f'{root_dir}/pp_qe/SSSP_1.3.0_PBE_efficiency')   # switch to _precision if needed

# Load the JSON file containing the metadata
with open(f'{root_dir}/pp_qe/SSSP_1.3.0_PBE_efficiency.json', 'r') as f:  # switch to _precision if needed
    pp_metadata = json.load(f)

# Specify your target elements
elements = list(set(chemical_symbols))  # # Converts to set (removes duplicates) and back to list

# Extract pseudopotentials from the JSON and define the dictionary
pseudopotentials = {elem: pp_metadata[elem]['filename'] for elem in elements if elem in pp_metadata}

# Check the status of each element
for elem in elements:
    if elem in pp_metadata:
        filename = pp_metadata[elem]['filename']  # Get the filename from JSON
        print(f"\n{elem} found in JSON with file {filename}\n")
    else:
        print(f"WARNING: {elem} not found in JSON.")

# 3.2 Parameters  # INPUT
ecut     = int(os.getenv("ECUT"))   # The value retrieved via os.getenv() will be a string
rf       = int(os.getenv("RF"))     # 4 for NC; 8-12 for US; testing fot PAW (can be used at 4)
nk       = int(os.getenv("NK")) 
degauss  = float(os.getenv("DEGAUSS"))  
conv_thr = float(os.getenv("CONV_THR")) 
special  = "special"

parameters_lists = {
    'ecut'    : [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
    'rf'      : [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'nk'      : [10, 11, 12, 13, 14, 15],
    'degauss' : [0.01, 0.009, 0.008],
    'conv_thr': [1e-6, 1e-7, 1e-8],
    'special' : ["special"]
}
values_to_test = parameters_lists[test_parameter]

# 3.3 Run test
time_start = time.time()

energy_list  = []
moments_list = []

for value in values_to_test:
    print(f"\nTesting {test_parameter} with value: {value}\n")

    # Update the value for the parameter being tested 
    if test_parameter   == 'ecut':
        ecut = value
    elif test_parameter == 'rf':
        rf = value
    elif test_parameter == 'nk':
        nk = value
    elif test_parameter == 'degauss':
        degauss = value
    elif test_parameter == 'conv_thr':
        conv_thr = value
    elif test_parameter == 'special':
        special = value    

    # Define the input data for the QE calculation

    # Set occupations based on band gap

    use_smearing_for_insulator = True  # INPUT
   
    if band_gap == 0:
        system = {
            'ecutwfc': ecut,
            'ecutrho': rf * ecut,
            'nspin': 2,
            'occupations': 'smearing',
            'smearing': 'mv',   # also called 'cold'
            'degauss': degauss,
            'nosym': os.getenv("NOSYM")=="1",
            'noinv': os.getenv("NOINV")=="1"
        }

    elif use_smearing_for_insulator:
        system = {
            'ecutwfc': ecut,
            'ecutrho': rf * ecut,
            'nspin': 2,
            'occupations': 'smearing',
            'smearing': 'gauss',   # better for insulator
            'degauss': float(os.getenv("DEGAUSS_INSULATOR")) ,    # Very small smearing
            'nosym': os.getenv("NOSYM")=="1",
            'noinv': os.getenv("NOINV")=="1"
        }

    elif band_gap > 0:  
        number_of_magnetic_atoms = len(magnetic_atom_indices)
        if magnetic_structure.startswith('AFM'):
            tot_magnetization = 0
        elif magnetic_structure == 'FM':
            tot_magnetization = int(math.ceil(m * number_of_magnetic_atoms))  # if error, plus one
        else:
            raise ValueError(f"Unknown magnetic configuration: {magnetic_structure}")
    
        system = {
            'ecutwfc': ecut,
            'ecutrho': rf * ecut,
            'nspin': 2,
            'occupations': 'fixed',
            'tot_magnetization': tot_magnetization,
            'nosym': os.getenv("NOSYM")=="1",
            'noinv': os.getenv("NOINV")=="1"
        }    

    else:
        raise ValueError(f"Invalid band gap value for material: {material_id}")

    input_data_scf = {    
        'control': {
            'calculation': 'scf',
            'pseudo_dir': f'{root_dir}/pp_qe/SSSP_1.3.0_PBE_efficiency',
            'prefix': 'pwscf',  # Default:  'pwscf',
            'max_seconds' : int(os.getenv("MAX_SECONDS")) 
        }, 
        'system': system,
        'electrons': {
            'conv_thr': conv_thr,
            'mixing_mode': os.getenv("MIXING_MODE"),  # Default: 'plain' (Broyden mixing)
            'mixing_ndim': int(os.getenv("MIXING_NDIM")) ,    # default is 8
            'mixing_beta': float(os.getenv("MIXING_BETA")),    # change to 0.3 even 0.1 as needed
            'diagonalization': os.getenv("DIAGONALIZATION"),
            'diago_david_ndim': int(os.getenv("DIAGO_DAVID_NDIM")),
            'electron_maxstep': int(os.getenv("STEP"))  # Set maximum SCF iterations
        }
    }

    # Set up Espresso calculator

    # 2025.08.03 DFT+U calculation
    # Define Hubbard_U values for magnetic elements
    # in the order of Element table
    U_dict = {
        'Cr': 4.0,
        'Mn': 4.0, 
        'Fe': 5.0, 
        'Co': 3.0,
        'Ni': 6.6,    
        'Nd': 7.0, 
        'Pm': 7.0,
        'Sm': 7.0, 
        'Gd': 7.0, 
        'Ho': 7.0, 
        'Er': 7.0,
    }

    # 3d transition elements and 4f rear earth elements
    orbital_dict = {
        'Cr': '3d',
        'Mn': '3d',
        'Fe': '3d',
        'Co': '3d',
        'Ni': '3d',
        'Nd': '4f',
        'Pm': '4f',
        'Sm': '4f',
        'Gd': '4f',
        'Ho': '4f',
        'Er': '4f',
    }
    

    # calc = Espresso(
    #     profile          = profile,
    #     pseudopotentials = pseudopotentials,
    #     input_data       = input_data_scf,
    #     kpts             = (nk, nk, nk),
    #     koffset          = (1, 1, 1),
    #     directory        = output_sub5,    #INPUT
    # )


    
    #  The lines below are commented
    # structure.calc = calc
    # # Perform the calculation and store results
    # energy  = structure.get_potential_energy() / len(structure)  # Energy per atom, in eV/atom
    # moments = structure.get_magnetic_moments()
    # energy_list.append(energy)
    # moments_list.append(moments)



    # The lines below are new
    # === Use write() to create QE input file with DFT+U (HUBBARD) ===
    input_file_path = os.path.join(output_sub5, 'espresso.pwi')
    additional_cards = [
        "HUBBARD ortho-atomic",
        f"U {magnetic_symbol}-{orbital_dict[magnetic_symbol]} {U_dict[magnetic_symbol]}"
    ]
    write(
        filename=input_file_path,
        images=structure,
        format='espresso-in',
        input_data=input_data_scf,
        pseudopotentials=pseudopotentials,
        kpts=(nk, nk, nk),
        koffset=(1, 1, 1),
        additional_cards=additional_cards
    )

    # === Run QE manually using subprocess ===
    qe_command = f"mpirun -np {ntasks} {root_dir}/conda/envs/dft1-env/bin/pw.x < espresso.pwi > espresso.pwo"
    subprocess.run(qe_command, shell=True, cwd=output_sub5)

    pwo_file = os.path.join(output_sub5, 'espresso.pwo')

    # === Extract total energy in eV ===
    def extract_total_energy_ev(pwo_path):
        """Extract final total energy from QE .pwo file, return in eV."""
        RY_TO_EV = 13.60569312299
        with open(pwo_path, 'r') as f:
            for line in f:
                if line.strip().startswith('!'):
                    energy_ry = float(line.split('=')[1].split()[0])
                    return energy_ry * RY_TO_EV
        raise ValueError("No total energy line found in file.")

    energy = extract_total_energy_ev(pwo_file) / len(structure)        # unit: eV/atom

    def extract_magnetic_moments(pwo_path):
        """Extract atomic magnetic moments from QE .pwo file."""
        moments = []
        record = False
        with open(pwo_path, 'r') as f:
            for line in f:
                if 'Magnetic moment per site' in line:
                    record = True
                    continue
                if record:
                    if not line.strip():  # empty line ends section
                        break
                    if 'atom' in line and 'magn=' in line:
                        tokens = line.strip().split()
                        for i, token in enumerate(tokens):
                            if token.startswith('magn='):
                                moments.append(float(tokens[i + 1]))
                                break
        if not moments:
            raise ValueError("No magnetic moments found in QE output.")
        return moments

    moments = extract_magnetic_moments(pwo_file)

    energy_list.append(energy)
    moments_list.append(moments)



    print(f"Test parameter: {test_parameter}, Value: {value}, Energy per atom: {energy} eV/atom")

    # Define the output file path in output_sub1 with a clear suffix
    base_name = f"{material_id}_{magnetic_structure}_{script_base_name}_{test_parameter}_job{slurm_job_id}"
    log_file_path = f"{output_sub1}/{base_name}_log_results.txt"

    # Write the test result to the log file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Test parameter: {test_parameter}, Value: {value}, Energy per atom: {energy} eV/atom\n")

time_end = time.time()
hours, remainder = divmod(time_end - time_start, 3600)
minutes, seconds = divmod(remainder, 60)
#print(f"Total Calculation Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")




########################################################################################################################
# SECTION 4: Save Results and Plot, analyze convergence

# 4.1 Save results to a text file

convergence_test_criterion = 1  # INPUT  # meV/atom

def write_results(f):
    # 1st block: ID, time, PP
    f.write(f"{'SLURM Job ID:':<25} {slurm_job_id}\n")
    f.write(f"{'Start Time:':<25} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_start))}\n")
    f.write(f"{'Calc. Time:':<25} {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
    f.write("Pseudopotentials:\n")
    for element, pseudo in pseudopotentials.items():
        f.write(f"  {element:<5} {pseudo}\n")
    
    # 2nd block: general parameters
    f.write(f"\n{'-' * 144}\n")   # sum of width bolow, be careful that space between {} can add to length
    f.write(
        f"{'Mag.':<8}{'m':<6}"
        f"{'mixing_mode':<13}{'mixing_beta':<13}{'occupations':<13}{'smearing':<10}"
        f"{'partition':<11}{'nodelist':<25}{'ntasks':<8}{'mem-per-cpu':<13}{'cpus-per-task':<15}\n"
    )
    f.write(
        f"{magnetic_structure:<8}{m:<6}"
        f"{input_data_scf['electrons'].get('mixing_mode', 'N/A'):<13}"
        f"{input_data_scf['electrons'].get('mixing_beta', 'N/A'):<13}"
        f"{input_data_scf['system'].get('occupations', 'N/A'):<13}"
        f"{input_data_scf['system'].get('smearing', 'N/A'):<10}"
        f"{partition:<11}{nodelist:<25}{ntasks:<8}{mem_per_cpu:<13}{cpus_per_task:<15}\n"
    )

    # 3rd block: Five key to-test parameters
    f.write(f"\n{'-' * 144}\n")
    f.write(
        f"{'ecut':<6}{'rf':<4}{'nk':<4}{'degauss':<9}{'conv_thr':<10}"
        f"{'Energy (eV/atom)':<25}{'Energy Diff (meV/atom)':<25}{'Converged?':<12}\n"
    )
    first_converged = None      # Initialize for convergence tracking
    for i, (value, energy, moments) in enumerate(zip(values_to_test, energy_list, moments_list)):
        # Dynamically update the tested parameter in each row
        row_ecut = value if test_parameter   == 'ecut' else ecut
        row_rf   = value if test_parameter   == 'rf'   else rf
        row_nk   = value if test_parameter   == 'nk'   else nk
        row_degauss = (value if test_parameter == 'degauss' 
            else input_data_scf['system'].get('degauss', 'N/A'))
        row_conv_thr = value if test_parameter == 'conv_thr' else conv_thr

        # Write results in 5 key parameters or special case
        if i == 0 or test_parameter == 'special':
            f.write(
                f"{row_ecut:<6}{row_rf:<4}{row_nk:<4}{row_degauss:<9}{row_conv_thr:<10.1e}"
                f"{energy:<25}{'-':<25}{'-':<12}\n"
            )
        else:
            diff = (energy_list[i] - energy_list[i - 1]) * 1000   # transform to meV/atom
            converged = "(c)" if abs(diff) <= convergence_test_criterion else ""
            f.write(
                f"{row_ecut:<6}{row_rf:<4}{row_nk:<4}{row_degauss:<9}{row_conv_thr:<10.1e}"
                f"{energy:<25}{diff:<25}{converged:<12}\n"
            )
            if converged and first_converged is None:
                first_converged = value
    # Log the first converged value, if found
    if first_converged:
        f.write(
            f"\nFirst converged {test_parameter} value: {first_converged} "
            f"under convergence_test_criterion {convergence_test_criterion} meV/atom\n"
        )
    
    # 4th block: Write magnetic moments separately in a new block
    f.write(f"\n{'-' * 144}\n")
    f.write("Magnetic Moments for Each Test Value:\n")
    # Loop through each test value and write moments
    for i, moments in enumerate(moments_list):
        f.write(f"Test Value {values_to_test[i]}:\n")
        # Split moments into chunks of 10 values per line
        chunks = [moments[j:j + 10] for j in range(0, len(moments), 10)]       
        # Write each chunk on a separate line
        for chunk in chunks:
            moments_str = ', '.join([f"{m}" for m in chunk])  # Format each moment with 6 decimals
            f.write(f"  {moments_str}\n\n")

base_name = f"{material_id}_{magnetic_structure}_{script_base_name}_{test_parameter}_job{slurm_job_id}"
with open(f"{output_sub5}/{base_name}.txt", "a") as f1, open(f"{output_sub1}/{base_name}.txt", "a") as f2:
    write_results(f1)  # Write to last job-folder sub5
    write_results(f2)  # Write to  sub1 folder

# 4.2 Plot the results only if the test is not "special"
if test_parameter != 'special':
    plt.plot(values_to_test, energy_list, marker='o')
    plt.xlabel(f'{test_parameter}')
    plt.ylabel('Energy (eV/atom)')
    plt.title(f'{test_parameter} Convergence Test: {magnetic_structure}')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.tight_layout()
    plt.savefig(f"{output_sub5}/{base_name}.png")  # Job-specific folder
    plt.savefig(f"{output_sub1}/{base_name}.png")  # Shared folder
    plt.close()
