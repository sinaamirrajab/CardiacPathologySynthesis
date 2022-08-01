"""
To plot the loss curves.
Example:
    python3 plot_loss.py  --log_file checkpoints/test-cityscape/loss_log.txt  --out_dir checkpoints/test-cityscape/ --nrows 2
"""
import matplotlib.pyplot as plt
import argparse
import os
import math

class Plot_loss():
    def __init__(self, opt):
        self.opt = opt
        plt.switch_backend('agg')
        parser = argparse.ArgumentParser()
        self.opt = parser.parse_args()

        assert os.path.exists(self.opt.log_file),'the given log file is not valid'
    
    def plot_loss(self):
        self.out_dir = './curves'
        

        with open(self.log_file) as f:
            lines = f.readlines()
            sl = [line.split() for line in lines]
            
        # del(sl[0]) # remove the first line
    
        dict={}
        columns = int(len(sl[-1]))
        for rows in range(len(sl)):
            if not '=============' in sl[rows][0]:
                for i in range(0,columns,2):
                    key=sl[rows][i]
                    newstr = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in sl[rows][i+1])
                    val = [float(i) for i in newstr.split()][0]
                    if key not in dict.keys():
                        dict[key]= [val]
                    else:
                        dict[key].append(val)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        ## plot all the curves in one figure
        # rows = self.nrows
        rows = 2
        num_cur =  len(dict.keys())-3
        col =math.ceil(num_cur/ rows)

        fig, axs = plt.subplots(rows, col,figsize=(15, 8))
        axs = axs.ravel()

        for i,key in enumerate(dict.keys()):
            if i>2:
                axs[i-3].plot(dict[key])
                axs[i-3].set_title(key)

        # name = os.path.basename(args.log_file).split('.')[0]
        name = self.log_file.split('/')[2]


        path = os.path.join(self.out_dir, name +'.png')
        plt.savefig(path)
        # print('saved in:', path)
        plt.close(fig)