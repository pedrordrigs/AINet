import random

# initialize a 8 genes long cell by randomness

def create_cell(cell_len, gene_library):
    cell_string = ''
    # gene_library = [['7 1 8 ', '1 6 9 ', '9 1 7 ', '3 4 5 '],['8 2 5 ', '1 9 7 ', '2 4 3 ', '1 6 2 '],['1 8 2 ', '6 1 9 ', '7 1 9 ', '6 1 2 ']]
    cell = []
    for i in range(cell_len):
        lib =  random.randint(0, len(gene_library)-1)
        gene = random.randint(0, len(gene_library[0])-1)
        cell.append(gene_library[lib][gene])
    for i in range(len(cell)):
        cell_string = cell_string + str(cell[i])
    return cell_string

def affinity(cell1, cell2):
    affinity = 0
    for i in range(len(cell1)):
        if cell1[i] != cell2[i]:
            affinity += 1
    return affinity

def fix(cell):
    cell = cell.replace(" ", "")
    print(cell)

cell = create_cell(3)
print(cell)
fix(cell)