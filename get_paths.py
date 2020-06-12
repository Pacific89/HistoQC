import os
import argparse
from pathlib import Path
import csv



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--folder', help="folder to look for files", default=os.getcwd(), type=str)

    args = parser.parse_args()
    folder = str(args.folder)
    file_list = []

    for r,d,f in os.walk(folder):
        for item in f:
            if ".mrxs" in item:
                file_list.append(os.path.join(r,item))

    
    print("Files found: ",len(file_list))
    print(file_list)

    with open('file_list.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_ALL)
        for p in file_list:
            out_file.write(p)
            out_file.write("\n")