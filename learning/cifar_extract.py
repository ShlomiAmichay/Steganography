import shutil
import os
import sys

dir = "./data/cifar100/train/"

with open('output.txt','w') as fout:
   for root, subFolders, files in os.walk(dir):
      for file in files:
         if '.png' in file:
            p = root +"/"+ file
            d = "./train1/" +file
            shutil.move(p,d)
