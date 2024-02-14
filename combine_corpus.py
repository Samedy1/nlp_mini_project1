'''
run this file to combine each individual corpus into corpus.txt
'''

import os

files = os.listdir('corpus')

files.remove('corpus.txt')

with open(f'corpus/corpus.txt', 'w') as outfile:
    outfile.write('')

for file in files:
    with open(f'corpus/{file}', 'r') as infile:
        with open(f'corpus/corpus.txt', 'a') as outfile:
            outfile.write(infile.read())