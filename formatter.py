import pickle

encoding = {
'A':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'R':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'N':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'D':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'C':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'Q':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'E':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
'G':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
'H':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
'I':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
'L':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
'K':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
'M':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
'F':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
'P':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
'S':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
'W':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
'Y':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
'V':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

label_encoding = {'H':1, 'G':1, 'I':1, 'B':0, 'E':0, 'T':0, 'S':0, ' ':0}


dataset = {}
window_size = 10

##### open a file with one pdbcode for line and 
##### scrolls code by code
id_list = open('dssp_list','r')
for code in id_list:
    code = code.strip()

##### this part opens a specific dssp file and save in a list
##### amino acid and secondary structure code, cutting out the dssp header
    dssp = []
    sequence = 'n'
    dssp_file = open('DSSP/'+code+'.dssp','r')
    for line in dssp_file:
        if line.startswith('  #  RESIDUE'): sequence = 'y'
        if sequence == 'n': continue
        dssp.append([line[13], line[16]])
    dssp_file.close()

##### scroll the list of residues saved from dssp in the list
    for pos in range(len(dssp)):

##### check each possible window of the specified length and
##### skips the ones with missing or modified residues
        broken_window = 'n'
        for n in range(-window_size, window_size+1):
            if pos+n < 0 or pos+n >= len(dssp): continue
            if dssp[pos+n][0] not in encoding: broken_window = 'y'
        if broken_window == 'y': continue

##### encode the non skipped windows and store them in dataset
        window = []
        example = []
        for n in range(-window_size, window_size+1):
            if pos+n < 0 or pos+n >= len(dssp): window.append([0]*len(encoding['A']))
            else: window.append(encoding[dssp[pos+n][0]])

        example = [window[:], label_encoding[dssp[pos][1]]]
        dataset[code] = dataset.get(code, [])
        dataset[code] = dataset[code] + [example[:]]
id_list.close()

##### save the dataset dictionary in pickle format
with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)
