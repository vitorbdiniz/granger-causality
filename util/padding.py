import pandas as pd
import os

#Verbose configurations

def verbose(msg, level, verbose=0):
    '''
        Printa mensagens na tela para um nível de verbose
        level in {1,2,3,4,5}
        v in {0,1,2,3,4,5}
    '''
    if msg != '' and level <= verbose:
        if msg[0] == '-' and msg[-1] == '-':
            msg = betwen_lines(msg)
        elif msg == "line" or msg == "l":
            msg = get_line()
        print(msg)
    
    return

def get_line():
    return "-"*91

def betwen_lines(msg):
    s = 91 - len(msg)
    line = str("-" * int(s/2))
    return line + msg + line

#Persist configurations

def persist(data, path, to_persist=True, _verbose=0, verbose_level=1, verbose_str=""):
    if to_persist:
        verbose(verbose_str, level=verbose_level, verbose=_verbose)
        
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data.to_csv(path)
        else:
            write_file(data, path)
            
        verbose("-- OK.", level=verbose_level, verbose=_verbose)



def persist_collection(collection, path, extension=".csv", to_persist=True, _verbose=0, verbose_level=0, verbose_str=""):
    if to_persist:
        make_dir(path)
        verbose(verbose_str, level=verbose_level, verbose=_verbose)
        for c in collection:
            c_path = path + str(c) + extension
            persist(collection[c], c_path, to_persist=to_persist)
        verbose("-- OK.", level=verbose_level, verbose=_verbose)


def write_file(data, path):
    f = open(path, "w")
    f.write(data) 
    f.close()


def list_to_string(seq, sep=','):
    return sep.join(str(i) for i in seq)

def make_dir(path='./'):
    if not os.path.isdir(path):
        dir_list = [d for d in path.split('/') if len(d) > 0]
        for i in range(1, len(dir_list)+1):
            directory = list_to_string(dir_list[0:i], sep='/')
            if not os.path.isdir(directory):
                os.mkdir(directory)