import os
import time
import pandas as pd

csv_path = 'test_data.csv'
fasta_dir = 'data/fasta'

pdb_csv = pd.read_csv(csv_path)
row_count = pdb_csv.shape[0]

for i in range(row_count):
    d = pdb_csv.iloc[i]['pdb_id']
    print(d, time.strftime("%Y-%M-%D %H:%M:%S", time.localtime(time.time())))

    protein_name = d

    fasta_path = os.path.join(fasta_dir, protein_name+'.fasta')
    sequence = pdb_csv.iloc[i]['sequence']

    with open(fasta_path, 'w') as file:
        first_line = '>' + protein_name
        file.write(first_line)
        file.write('\n')
        file.write(sequence)
