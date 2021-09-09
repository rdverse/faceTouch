import re
import os
import itertools
'''
This script renames the files name to their respective activity number

Requisities:
       activity number should be present ahead of the string 
'''


def get_digit_name(name):
    name = re.findall('\d+', name)
    name = name[0] + '.csv'
    return(name)
    


def rename(filePATHs , renamePATHS):
    for file, rename in zip(filePATHS, renamePATHS):
        os.rename(file, rename)


dataPATH = '../data'
personPATHS = [dataPATH + '/' + path for path in  os.listdir(dataPATH)]
filePATHS = list()
changePATHS = list()

for path in personPATHS:
    files = os.listdir(path)
    filePATHS.append([path + '/' + file for file in os.listdir(path)])
    changePATHS.append([path + '/' + get_digit_name(file) for file in os.listdir(path)])
    
filePATHS = list(itertools.chain(*filePATHS))
changePATHS = list(itertools.chain(*changePATHS))

rename(filePATHS, changePATHS)
