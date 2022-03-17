#!/usr/bin/env python3
import argparse

def formula2cnf(args):
    with open('GFG.txt','r') as f:
        str = f.read()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file.")
    parser.add_argument("output", nargs='?', type=str, help="Output file.") 
    args = parser.parse_args([] if "__file__" not in globals() else None)

    formula2cnf(args)