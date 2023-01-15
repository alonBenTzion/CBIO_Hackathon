from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import os
import sys


def read_fasta(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    sequence = ''
    for line in lines:
        if not line.startswith('>'):
            sequence += line.strip()
    return sequence


def read_pdb_file(input_file_name, output_file_name) -> None:
    """
    reads a pdb file and parsed it to dssp format. Writes (or creates) a file where the first line is tha AA sequence,
    and the second line is the secondary structures for each residue. Structures are clustered to three groups:
    H (helix) : {'G','H',I'}, S (strand): {'E','B'}, L (loop) : {'S,'T','-'}
    (see https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html) for more details.
    :param input_file_name: path to pdb file
    :param output_file_name: file path to write the output.
    """
    structure_dic = {"G": "H", "H": "H", "I": "H", "P": "H", "E": "S", "B": "S", "S": "L", "T": "L", "-": "L", "C": "L"}
    seq, sec_structures = "", ""
    p = PDBParser()
    structure = p.get_structure("1MOT", input_file_name)
    model = structure[0]
    dssp = DSSP(model, input_file_name, dssp='dssp')
    for i, residue in enumerate(dssp.keys()):
        key = dssp.keys()[i]
        amino_acid, struc = dssp[key][1], structure_dic[dssp[key][2]]
        seq += amino_acid
        sec_structures += struc
    with open(output_file_name, "w") as f:
        f.write(seq + "\n")
        f.write(sec_structures)


def read_pdb_dir(input_dir_name, output_dir_name) -> None:
    """
    Parses all pdb files in a given directory. for each pdb file, a file with the same name and extension ".txt"
    is created in the given output directory. Each created file contains two lines: the first line is the
    amino acids sequence. The second line is the secondary structures (H, S, or L) for each residue (see read_pdb_file)
    for explanation about the groups of secondary structures.
    :param input_dir_name: path to directory from which to read the pdb files.
    :param output_dir_name: path to directory to write the output files.
    """
    for file_name in os.listdir(input_dir_name):
        if file_name.endswith('.pdb'):
            input_file_path = input_dir_name + "/" + file_name
            output_file_path = output_dir_name + "/" + file_name.split(".")[0] + ".txt"
            read_pdb_file(input_file_path, output_file_path)
    print("Files created successfully")


if __name__ == '__main__':
    args = sys.argv
    input_dir, output_dir = args[1], args[2]
    read_pdb_dir(input_dir, output_dir)
