
"""
 Yusuf Alptug Aydin
 260201065

-------------------------------------------------------------------------------------------------
 Names of files indicate question letters:

 A.py file is for 2-layer NN, 300 HU , mean square error
 B.py file is for 2-layer NN, 300 + 100 HU , mean square error
 C.py file is for 3-layer NN, 500 + 300 HU , softmax, cross entropy, weight decay

 dlc_practical_prologue_edited.py file is 3 line different version of the provided in the course.

 -------------------------------------------------------------------------------------------------
"""

import A
import B
import C
import argparse

parser = argparse.ArgumentParser(description='DLC prologue file for practical sessions.')
parser.add_argument('-a','--arch',required=True)
args = parser.parse_args()

if(args.arch == "1"):
    A.run()

if(args.arch == "2"):
    B.run()

if(args.arch == "3"):
    C.run()