########################################################################################################################
# visualize.py
# 2024.09.08

########################################################################################################################
# SECTION 1: Import and input

# 1.1 Packages
from structure import *
import pyvista as pv
import numpy as np

# 1.2 User choice for visualizing either the unit cell (atoms) or the supercell
structure_choice = input("Enter 'atoms' or 'supercell': ").strip().lower()

if structure_choice == 'atoms':
    structure = atoms
elif structure_choice == 'supercell':
    structure = supercell
else:
    raise ValueError("Invalid choice. Please enter 'atoms' or 'supercell'.")

# 1.3 Get Cartesian positions, atomic numbers, and other necessary properties from the selected structure
positions_cartesian = structure.get_positions()
atomic_numbers      = structure.get_atomic_numbers()
chemical_symbols    = structure.get_chemical_symbols()
lattice             = structure.get_cell()
positions           = structure.get_scaled_positions(wrap=True)  # fractional coordinates
# Determine indices of magnetic atoms in the selected structure
magnetic_atom_indices = [i for i, symbol in enumerate(chemical_symbols) if symbol == magnetic_symbol]


# 1.4 Create output sub-directories
output_sub1 = f'output/{material_id}/structure';   os.makedirs(output_sub1, exist_ok=True) 


# Define available magnetic structures and their corresponding names
magmoms_dict = {
    "FM":   magmoms_fm,
    "AFM1": magmoms_afm1, "AFM1a": magmoms_afm1a, "AFM1b": magmoms_afm1b,
    #"AFM2": magmoms_afm2_supercell if num_magnetic_atoms_unit_cell == 2 else magmoms_afm2,
    "AFM2": magmoms_afm2, # 2025.04.11 only for altermagnet 
    "AFM3": magmoms_afm3_supercell if num_magnetic_atoms_unit_cell == 2 else magmoms_afm3,
}

while True:
    magnetic_structure = input(
    "\nEnter FM, AFM1, AFM1a, AFM1b, AFM2, AFM3: ").strip()

    # Fetch the corresponding magmoms based on the chosen structure
    magmoms = magmoms_dict.get(magnetic_structure)

    # Check if the chosen structure is valid, else prompt again
    if not magmoms:
        print(f"No magnetic configuration found for {magnetic_structure}, try again.")
        continue

    # Set the magnetic moments to atoms
    structure.set_initial_magnetic_moments(magmoms)
    print(f"Magnetic structure {magnetic_structure} applied successfully.")


    ####################################################################################################################
    # SECTION: Add lattice

    plotter = pv.Plotter()  # Initialize the plotter

    # Define the vertices of the unit cell based on lattice_std vectors
    vertices = np.array([
        [0, 0, 0],
        lattice[0],
        lattice[1],
        lattice[2],
        lattice[0] + lattice[1],
        lattice[0] + lattice[2],
        lattice[1] + lattice[2],
        lattice[0] + lattice[1] + lattice[2]])

    # Define the edges of the unit cell using pairs of vertices
    edges = [
        [vertices[0], vertices[1]],
        [vertices[0], vertices[2]],
        [vertices[0], vertices[3]],
        [vertices[1], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[4]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[5]],
        [vertices[3], vertices[6]],
        [vertices[4], vertices[7]],
        [vertices[5], vertices[7]],
        [vertices[6], vertices[7]]]

    # Add the unit cell edges to the plotter
    for edge in edges:
        plotter.add_mesh(pv.Line(edge[0], edge[1]), color='black', line_width=1)

    ####################################################################################################################
    # SECTION: Add atoms
    # Jmol Color Map (hex code) for All 118 Common Elements
    # manually set light gray(#D3D3D3) for atoms that are too close
    color_map = {
        'H':  '#F5F5F5',      # 1   - Hydrogen     - Very Light Gray (white in Jmol)
        'He': '#D9FFFF',      # 2   - Helium       - Cyan
        'Li': '#CC80FF',      # 3   - Lithium      - Purple
        'Be': '#C2FF00',      # 4   - Beryllium    - Dark Green
        'B':  '#FFB5B5',      # 5   - Boron        - Salmon
        'C':  '#909090',      # 6   - Carbon       - Gray
        'N':  '#3050F8',      # 7   - Nitrogen     - Blue
        'O':  '#FF0D0D',      # 8   - Oxygen       - Red
        'F':  '#90E050',      # 9   - Fluorine     - Green
        'Ne': '#B3E3F5',      # 10  - Neon         - Light Blue
        'Na': '#AB5CF2',      # 11  - Sodium       - Purple
        'Mg': '#8AFF00',      # 12  - Magnesium    - Dark Green
        'Al': '#BFA6A6',      # 13  - Aluminum     - Gray
        'Si': '#F0C8A0',      # 14  - Silicon      - Orange
        'P':  '#FF8000',      # 15  - Phosphorus   - Orange
        'S':  '#FFFF30',      # 16  - Sulfur       - Yellow
        'Cl': '#1FF01F',      # 17  - Chlorine     - Green
        'Ar': '#80D1E3',      # 18  - Argon        - Light Blue
        'K':  '#8F40D4',      # 19  - Potassium    - Purple
        'Ca': '#3DFF00',      # 20  - Calcium      - Light Green
        'Sc': '#E6E6E6',      # 21  - Scandium     - Light Gray
        'Ti': '#BFC2C7',      # 22  - Titanium     - Light Gray
        'V':  '#A6A6AB',      # 23  - Vanadium     - Light Gray
        'Cr': '#8A99C7',      # 24  - Chromium     - Light Blue
        'Mn': '#9C7AC7',      # 25  - Manganese    - Purple
        'Fe': '#E06633',      # 26  - Iron         - Dark Orange
        'Co': '#F090A0',      # 27  - Cobalt       - Pink
        'Ni': '#50D050',      # 28  - Nickel       - Green
        'Cu': '#C88033',      # 29  - Copper       - Brown
        'Zn': '#7D80B0',      # 30  - Zinc         - Purple
        'Ga': '#C28F8F',      # 31  - Gallium      - Light Pink
        'Ge': '#50D050',      # 32  - Germanium    - Green (was Light Blue-Gray: #668F8F)
        'As': '#90E050',      # 33  - Arsenic      - Green (was Purple: #BD80E3)
        'Se': '#FFA100',      # 34  - Selenium     - Orange
        'Br': '#A62929',      # 35  - Bromine      - Dark Red
        'Kr': '#5CB8D1',      # 36  - Krypton      - Light Blue
        'Rb': '#702EB0',      # 37  - Rubidium     - Purple
        'Sr': '#3050F8',      # 38  - Strontium    - blue was Green #00FF00
        'Y':  '#94FFFF',      # 39  - Yttrium      - Light Blue
        'Zr': '#94E0E0',      # 40  - Zirconium    - Light Blue
        'Nb': '#73C2C9',      # 41  - Niobium      - Light Blue
        'Mo': '#54B5B5',      # 42  - Molybdenum   - Light Blue
        'Tc': '#3B9E9E',      # 43  - Technetium   - Light Blue
        'Ru': '#248F8F',      # 44  - Ruthenium    - Dark Cyan
        'Rh': '#0A7D8C',      # 45  - Rhodium      - Dark Cyan
        'Pd': '#006985',      # 46  - Palladium    - Dark Cyan
        'Ag': '#C0C0C0',      # 47  - Silver       - Silver
        'Cd': '#FFD98F',      # 48  - Cadmium      - Light Yellow
        'In': '#A67573',      # 49  - Indium       - Light Brown
        'Sn': '#668080',      # 50  - Tin          - Gray
        'Sb': '#9E63B5',      # 51  - Antimony     - Purple
        'Te': '#D47A00',      # 52  - Tellurium    - Orange
        'I':  '#940094',      # 53  - Iodine       - Purple
        'Xe': '#429EB0',      # 54  - Xenon        - Cyan
        'Cs': '#57178F',      # 55  - Cesium       - Purple
        'Ba': '#D3D3D3',      # 56  - Barium       - was #00C900 Green
        'La': '#70D4FF',      # 57  - Lanthanum    - Light Blue
        'Ce': '#9C7AC7',      # 58  - Cerium       previously #FFFFC7 - Light Yellow
        'Pr': '#D9FFC7',      # 59  - Praseodymium - Light Green
        'Nd': '#C7FFC7',      # 60  - Neodymium    - Light Green
        'Pm': '#A3FFC7',      # 61  - Promethium   - Light Green
        'Sm': '#8FFFC7',      # 62  - Samarium     - Light Green
        'Eu': '#61FFC7',      # 63  - Europium     - Light Green
        'Gd': '#45FFC7',      # 64  - Gadolinium   - Light Green
        'Tb': '#30FFC7',      # 65  - Terbium      - Light Green
        'Dy': '#1FFFC7',      # 66  - Dysprosium   - Light Green
        'Ho': '#00FF9C',      # 67  - Holmium      - Light Green
        'Er': '#575961',      # 68  - Erbium       - Gray
        'Tm': '#00D452',      # 69  - Thulium      - Dark Green
        'Yb': '#00BF38',      # 70  - Ytterbium    - Dark Green
        'Lu': '#00AB24',      # 71  - Lutetium     - Dark Green
        'Hf': '#4DC2FF',      # 72  - Hafnium      - Light Blue
        'Ta': '#4DA6FF',      # 73  - Tantalum     - Light Blue
        'W':  '#2194D6',      # 74  - Tungsten     - Blue
        'Re': '#267DAB',      # 75  - Rhenium      - Blue
        'Os': '#266696',      # 76  - Osmium       - Dark Blue
        'Ir': '#175487',      # 77  - Iridium      - Dark Blue
        'Pt': '#D0D0E0',      # 78  - Platinum     - Light Gray
        'Au': '#FFD123',      # 79  - Gold         - Gold
        'Hg': '#B8B8D0',      # 80  - Mercury      - Light Purple
        'Tl': '#A6544D',      # 81  - Thallium     - Light Brown
        'Pb': '#575961',      # 82  - Lead         - Gray
        'Bi': '#9E4FB5',      # 83  - Bismuth      - Purple
        'Po': '#AB5C00',      # 84  - Polonium     - Orange
        'At': '#754F45',      # 85  - Astatine     - Brown
        'Rn': '#428296',      # 86  - Radon        - Cyan
        'Fr': '#420066',      # 87  - Francium     - Purple
        'Ra': '#007D00',      # 88  - Radium       - Green
        'Ac': '#70ABFA',      # 89  - Actinium     - Light Blue
        'Th': '#00BAFF',      # 90  - Thorium      - Light Blue
        'Pa': '#00A1FF',      # 91  - Protactinium - Light Blue
        'U':  '#008FFF',      # 92  - Uranium      - Light Blue
        'Np': '#0080FF',      # 93  - Neptunium    - Light Blue
        'Pu': '#006BFF',      # 94  - Plutonium    - Blue
        'Am': '#545CF2',      # 95  - Americium    - Blue
        'Cm': '#785CE3',      # 96  - Curium       - Purple
        'Bk': '#8A4FE3',      # 97  - Berkelium    - Purple
        'Cf': '#A136D4',      # 98  - Californium  - Purple
        'Es': '#B31FD4',      # 99  - Einsteinium  - Purple
        'Fm': '#B31FBA',      # 100 - Fermium      - Purple
        'Md': '#B30DA6',      # 101 - Mendelevium  - Purple
        'No': '#BD0D87',      # 102 - Nobelium     - Purple
        'Lr': '#C70066',      # 103 - Lawrencium   - Red
        'Rf': '#CC0059',      # 104 - Rutherfordium- Red
        'Db': '#D1004F',      # 105 - Dubnium      - Red
        'Sg': '#D90045',      # 106 - Seaborgium   - Red
        'Bh': '#E00038',      # 107 - Bohrium      - Red
        'Hs': '#E6002E',      # 108 - Hassium      - Red
        'Mt': '#EB0026',      # 109 - Meitnerium   - Red
        'Ds': '#EC0020',      # 110 - Darmstadtium - Red
        'Rg': '#ED0020',      # 111 - Roentgenium  - Red
        'Cn': '#EE0020',      # 112 - Copernicium  - Red
        'Nh': '#F00020',      # 113 - Nihonium     - Red
        'Fl': '#F1001C',      # 114 - Flerovium    - Red
        'Mc': '#F2001A',      # 115 - Moscovium    - Red
        'Lv': '#F30018',      # 116 - Livermorium  - Red
        'Ts': '#F40016',      # 117 - Tennessine   - Red
        'Og': '#F50014'       # 118 - Oganesson    - Red
    }

    # To keep track of which elements are added to the legend
    added_elements = set()

    for idx, (pos, atomic_number) in enumerate(zip(positions_cartesian, atomic_numbers)):
        symbol = chemical_symbols[idx]  # Get the symbol directly from chemical_symbols list
        sphere = pv.Sphere(radius=0.4, center=pos)
        color = color_map[symbol]

        # Add mesh with label only if the element hasn't been added before
        if symbol not in added_elements:
            plotter.add_mesh(sphere, color=color, label=symbol)  # Add label for legend
            added_elements.add(symbol)
        else:
            plotter.add_mesh(sphere, color=color)

        # # Wu 显示数字编号在原子上
        # # Attach index directly on the surface of the sphere
        # if idx in magnetic_atom_indices:
        #     plotter.add_point_labels(
        #         points=[pos],  # Position on the surface of the sphere
        #         labels=[str(idx)],  # The index to be displayed
        #         font_size=30,  # Adjust font size as needed
        #         text_color='white',  # Change to 'black' or other color if needed for better visibility
        #         point_size=0,  # Size of the label points, adjust if necessary
        #         render_points_as_spheres=False,  # Render the labels as text
        #         always_visible=True  # Ensure the labels are always visible
        #     )


    ####################################################################################################################
    # SECTION: Add custom axes lines
    a_axis = lattice[0]  # Vector for a-axis
    b_axis = lattice[1]  # Vector for b-axis
    c_axis = lattice[2]  # Vector for c-axis
    plotter.add_mesh(pv.Line([0, 0, 0], a_axis), color='red', line_width=6)
    plotter.add_mesh(pv.Line([0, 0, 0], b_axis), color='green', line_width=6)
    plotter.add_mesh(pv.Line([0, 0, 0], c_axis), color='blue', line_width=6)


    ####################################################################################################################
    # SECTION: Add pyvista axes 
    plotter.add_axes(interactive=0, xlabel='a', ylabel='b', zlabel='c', line_width=4, viewport=(0, 0, 0.3, 0.3))
    plotter.show_axes()

    plotter.background_color = 'white'
    
    # Wu: legend 
    plotter.add_legend(face='circle', size=(0.15, 0.15), loc='lower right', font_family='arial')


    ####################################################################################################################
    # SECTION: Clicking atoms to display information
    # Function to find the nearest atom to the clicked position
    def find_nearest_atom(clicked_pos, atom_positions):
        distances = np.linalg.norm(atom_positions - clicked_pos, axis=1)
        nearest_atom_index = np.argmin(distances)
        return nearest_atom_index

    # Callback function to handle clicks and show information about the nearest atom
    def my_callback(pos):
        A = np.array(pos)
        nearest_atom_index  = find_nearest_atom(A, positions_cartesian)
        fractional_coords   = positions[nearest_atom_index]
        symbol              = structure[atomic_numbers.tolist().index(atomic_numbers[nearest_atom_index])].symbol # structure
        
        # Check if Wyckoff information is available for this atom (Wyckoff only applies for original unit cell)
        if nearest_atom_index < len(wyckoff_letters):
            wyckoff_letter      = wyckoff_letters[nearest_atom_index]
            multiplicity        = equivalent_atom_groups[equivalent_atoms[nearest_atom_index]]['count']
            site_symmetry       = site_symmetry_symbols[nearest_atom_index]  # Add this line to get site symmetry
        else:
            wyckoff_letter, multiplicity, site_symmetry = None, None, None

        # Display the chemical symbol, atom index, fractional coordinates, Wyckoff position, and site symmetry  
        if wyckoff_letter is not None:
            plotter.add_text(
                f'{symbol}{nearest_atom_index}  {multiplicity}{wyckoff_letter}  {site_symmetry}' 
                f'  {fractional_coords}', 
                font_size=16,
                position='upper_edge',
                name='atom_info'
                )
        else:
            plotter.add_text(
            f'{symbol}{nearest_atom_index}  {fractional_coords}', 
            font_size=16,
            position='upper_edge',
            name='atom_info'
        )

        enlarged_sphere = pv.Sphere(radius=0.6, center=positions_cartesian[nearest_atom_index])
        plotter.add_mesh(enlarged_sphere, color=color_map[symbol], name='highlighted_atom')
        plotter.update()

    # Function to clear picking and reset sphere sizes
    def clear_picking():
        plotter.remove_actor('atom_info')
        plotter.remove_actor('highlighted_atom')
        plotter.update()

    # Enable click position tracking and add key event for clearing
    plotter.track_click_position(callback=my_callback)
    plotter.add_key_event("c", clear_picking)  # keyboard c


    ####################################################################################################################
    # SECTION: Add magnetic arrows
    # View the magnetic moment vectors
    # Define the scale for magnetic moment arrows
    arrow_length = 2  # Adjust this length to control the arrow size

    # Add magnetic moment arrows for magnetic atoms
    for idx in magnetic_atom_indices:
        pos = positions_cartesian[idx]  # Get the Cartesian position of the atom
        moment = np.array([0, 0, magmoms[idx]])  # moment vector, assuming along z-axis
        #moment = np.array([-magmoms[idx], 0, 0] )  # moment vector, try along -x-axis

        # Calculate the start position for the arrow such that the arrow is centered on the atom
        start_pos = pos - 0.5 * arrow_length * moment / np.linalg.norm(moment)

        # Create the arrow
        arrow = pv.Arrow(start=start_pos, direction=moment, scale=arrow_length)
        plotter.add_mesh(arrow, color='red')  # Change color as needed


    # ####################################################################################################################
    # SECTION: Add title
    # Define your structure type
    # The space group information is for the original unit cell, not for supercell
    structure_type = magnetic_structure  # Use the structure type the user selected
    space_group_info = f"{crystal_system}  {space_group_number} ({space_group_symbol}) " 

    # Add the title (material name and structure type) at the bottom
    title_text = f"{material_id}    {space_group_info}    {structure_type}"
    plotter.add_text(title_text, position='lower_edge', font_size=16, color='black')


    ####################################################################################################################
    # SECTION: Save and show

    # Save the current view (screenshot)
    fig_file_path1 = f'{output_sub1}/{material_id}_{magnetic_structure}.png' 
    # PNG provides lossless compression, better for scientific data than jpg
    def save_plotter_view1():
        plotter.screenshot(fig_file_path1, window_size=(6000, 4000), # (6000, 4000) # (3840, 2160)   
                           scale=1.98, # Scale factor for even higher resolution
                           transparent_background=False,  # Avoids artifacts from transparency to get high quality
                           )     
        
        print(f"Plot saved as '{fig_file_path1}'")
    # Add a keyboard shortcut to save the plot when 's' is pressed
    plotter.add_key_event("S", save_plotter_view1)


    # Save the current view (save_graphic)
    fig_file_path2 = f'{output_sub1}/{material_id}_{magnetic_structure}.eps'
    def save_plotter_view2():
        plotter.save_graphic(fig_file_path2)  # svg, eps, ps, pdf, tex
        print(f"Plot saved as '{fig_file_path2}'")
    # Add a keyboard shortcut to save the plot when 's' is pressed
    plotter.add_key_event("E", save_plotter_view2)


    gltf_file_path1 = f'{output_sub1}/{material_id}_{magnetic_structure}.gltf'  
    def save_plotter_view_gltf():
        # Save the 3D view as a .glb file
        plotter.export_gltf(gltf_file_path1)
        print(f"3D model saved as '{gltf_file_path1}'")
    # Add a keyboard shortcut to save the .glb file when 'G' is pressed
    plotter.add_key_event("G", save_plotter_view_gltf)


    # Show the plot interactively
    plotter.show()