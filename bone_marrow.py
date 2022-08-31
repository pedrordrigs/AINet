import random

# initialize a 8 genes long cell by randomness

def create_cell():
    gene_library = ['1A','1B','1C','1D','2A','2B','2C','2D']
    cell = []
    for i in range(8):
        cell.append(gene_library[random.randint(0, 7)])
    print(cell)
    return cell

def affinity(cell1, cell2):
    affinity = 0
    for i in range(len(cell1)):
        if cell1[i] != cell2[i]:
            affinity += 1
    return affinity

