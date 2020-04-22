import os
import sys
import shutil
from distutils import dir_util

# defining a python class to redirect script's output to both logfile and stdout
class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log.txt", "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def existOldData(path=None):
    if not isinstance(path,str):
        raise Exception("Wrong path data type!")
    return os.path.exists(path)

def lsgrep(path, wild_regex, with_path=True):
    base = path+'/' if with_path else ''
    ret = [base+f for f in os.listdir(path) if any(map(lambda wild: wild in f, wild_regex))]
    return ret

def mkdir_p(path):
    for p in path:
        os.makedirs(p,mode=0o755,exist_ok=True)

def cp(path1,path2):
    for path in path1:
        shutil.copy2(path,path2)

def cpr(path1,path2):
    for path in path1:
        dir_util.copy_tree(path, path2)

def rm(path):
    for p in path:
        os.remove(p)

def rmr(path):
    for p in path:
        shutil.rmtree(p,ignore_errors=True)

def mv(path1,path2):
    for path in path1:
        shutil.move(path,path2)

def cd(path):
    os.chdir(path)

def pwd():
    return os.getcwd()

def echo(path,text):
    with open(path,'w') as f:
        print(text,file=f)

def sexec(cmd):
    return os.system(cmd)

def cat(path):
    with open(path,'r') as f:
        return f.read()
