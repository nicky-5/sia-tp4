import itertools
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_letter_plot(letter, ax, cmap='Blues'):
    p = sns.heatmap(letter, ax=ax, annot=False, char=False,cmap=cmap, square=True,linewidth=2,linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p

def print_letters_line(letters,cmap='Blues',cmaps=[]):
    flg,ax = plt.subplots(1,len(letters))
    fig.set_dpi(360)
    if not cmaps:
        cmaps = [cmap]*len(letters)
    if len(cmaps) != len(letters):
        raise Exception('cmap list should be the same length as letters')
    for i,subplot in enumerate(ax):
        create_letter_plot(letters[i].reshape(5,5),ax=subplot,cmap=cmaps[i])
    plt.show()

with open("../datasets/letters.txt") as fp:
    letters = {}
    current = np.ones((5,5))*-1
    idx = 0
    for line in fp:
        if line[0] == '-':
            letters[string.ascii_uppercase[len(letters)]] = current
            current = np.ones((5,5)) * -1
            idx = 0
        else:
            for i,c in enumerate(line.strip('\n')):
                current[idx][i] = 1 if c == '*' else -1
            idx += 1

n = 6
letters_list = list(letters.values())

letters_list += [np.ones((5,5))*-1]*(n-len(letters_list)%n)

for letter_group in [letters_list[i * n:(i + 1)* n] for i in range(len(letters_list) // n)]:
    print_letters_line(letter_group)

# best 4 group combo

flat_letters = {
    k: m.flatten() for k,m in letters.items()
}

all_groups = itertools.combinations(flat_letters.keys(), r = 4)

avg_dot_product = []
max_dot_product = []

for g in all_groups:
    gropu = np.array([ v for k,v in flat_letters.items() if k in g])
    orto_matrix = group.dot(group.T)
    np.fill_diagonal(orto_matrix,0)
    print(f'{g}\n{orto_matrix}\n-------------------')
    row,_ = orto_matrix.shap
    avg_dot_product.append((np.abs(orto_matrix).sum()/(orto_matrix.size-row), g))
    max_v = np.abs(orto_matrix).max()
    max_dot_product.append(((max_v,np.count_nonzero(np.abs(orto_matrix) == max_v) / 2), g))

df = pd.Dataframe(sorted(avg_dot_product), columns = ["|<,>| medio", "grupo"])
df.head(15).style.format({'|<,>| medio': "{:.2f}"}).hide(axis='index')

df.tail(5).style.format({'|<,>| medio': "{:.2f}"}).hide(axis='index')

#minimum max dot product

df2 = pd.Dataframe(sorted(max_dot_product), columns=["|<,>| max","grupo"])
df2.head(15).style.format({'|<,>| max': lambda x: 'max : {:,.0f} | count: {:,.0f}'.format(*x)}).hide(axis='index')

df3 = df2.merge(df)
df3 = df3[['|<,>| max', '|<,>| medio','grupo']]
df3.head(15).style.format({'|<,> max': lambda x: 'max: {:,.0f} | count: {:,.0f}'.format(*x), '|<,>| medio': "{:,2f}"}).hide(axis='index')

#TESTING PATTERNS

patterns = ['A', 'J', 'L','X']
flat_letters= {
    k: m.flatten() for k,m in letters.items() if k in patterns
}

flat_letters_arr = np.array(list(flat_letters.values()))

for k,v in flat_letters.items():
    for k2,v2 in flat_letters.items():
        if k != k2 and k < k2:
            print(f"<{k},{k2}>: {v,dot(v2)}")

altv = [letters[k] for k in patterns]
print_letters_line(altv,'Greens')