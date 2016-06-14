from NNet import *
import numpy as np
import matplotlib.pyplot as plt

class Anduril():

    def __init__(self):
        self.Net = NNet()
        return 
    
    def init(self,sconfig,classreg = 0, numcores = 1, gradd = 0, costfunc = 0):
        if (type(sconfig) == str) and (type(classreg) == int) and (type(numcores) == int) and (type(gradd) == int) and (type(costfunc) == int):
            self.Net.init(sconfig,classreg,numcores,gradd,costfunc)
        else:
            print "Invalid input, network not initailized"
            return

    def func_arch(self,flayer):
        if type(flayer) != str:
            print "flayer must be a string"
            return
        else:
            self.Net.func_arch(flayer)
        return

    def load(self,filename, mode = 0, sep1 = ",", sep2 = " "):
        if (type(filename) == str) and (type(mode) == int) and (type(sep1) == str) and (type(sep2) == str):
            self.Net.load(filename,mode,sep1,sep2)
            return
        else:
            print "Invalid input, files not loaded!"
            return

    def test_file(self,filename, netname = " ", sep1 = ",", sep2 = " "):
        if (type(filename) == str) and (type(netname) == str) and (type(sep1) == str) and (type(sep2) == str):
            self.Net.test_file(filename, netname, sep1, sep2)
            return
        else:
            print "Invalid input type!"
            return

    def error_stats(self, num_bins):
        if type(num_bins) != int:
            print "Please enter an integer for the number of bins"
            return
        self.Net.error_stats()
        f_train = open(".train","r")
        f_test = open(".test", "r")
        train_vals = []
        test_vals = []
        for line in f_train:
            train_vals.append(float(line))
        for line in f_test:
            test_vals.append(float(line))
        mean_train = np.mean(train_vals)
        std_train = np.std(train_vals)
        mean_test = np.mean(test_vals)
        std_test = np.std(test_vals)
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.hist(train_vals, bins=num_bins)
        ax1.set_title("Histogram on training error (Mean: " + str(mean_train) + "," + "Standard Deviation: " + str(std_train) + ")")
        ax1.set_xlabel("Error")
        ax1.set_ylabel("Number of points")
        ax2.hist(test_vals, bins=num_bins)
        ax2.set_title("Histogram on test error (Mean: " +str(mean_test) + "," + "Standard Deviation: " + str(std_test) + ")")
        ax2.set_xlabel("Error")
        ax2.set_ylabel("Number of points")
        plt.show()
        return

    def train_net(self,epoch,lrate, mode = 1, verbose = 0, logfile = " "):
        if (type(epoch) == int) and (type(lrate) == float) and (type(mode) == int) and (type(verbose) == int) and (type(logfile) == str):
            if (mode < 0) or (mode > 1):
                print "The mode must be either 0 or 1"
                return
            if (verbose < 0) or (verbose > 1):
                print "Verbose must either be 0 or 1"
                return
            self.Net.train_net(epoch,lrate,mode,verbose,logfile)
            return
        else:
            print "Invalid input, network not trained!"
            return

    def train_rprop(self,epoch, mode = 1, verbose = 0, logfile = " ", tmax = 1.0):
        if  (type(epoch) == int) and (type(mode) == int) and (type(verbose) == int) and (type(logfile) == str) and (type(tmax) == float):
            if (mode < 0) or (mode > 1):
                print "The mode must be either 0 or 1"
                return
            if (verbose < 0) or (verbose > 1):
                print "Verbose must either be 0 or 1"
                return
            self.Net.train_rprop(epoch,mode,verbose,logfile,tmax)
            return
        else:
            print "Invalid input, network not trained!"
            return

    def d_trainrprop(self,mode = 1, verbose = 0, logfile = " ", tmax = 1.0):
        if (type(mode) == int) and (type(verbose) == int) and (type(logfile) == str) and (type(tmax) == float):
            if (mode < 0) or (mode > 1):
                print "The mode must be either 0 or 1"
                return
            if (verbose < 0) or (verbose > 1):
                print "Verbose must either be 0 or 1"
                return
            self.Net.d_trainrprop(mode,verbose,logfile,tmax)
            return
        else:
            print "Invalid input, network not trained!"
            return

    def test_net(self,verbose = 0):
        if (type(verbose) == int):
            if (verbose != 0) or (verbose != 1):
                print "Verbose must either be 0 or 1"
                return
            self.Net.test_net(verbose)
            return
        else:
            print "Invalid input!"
            return

    def savenet(self,netname):
        if (type(netname) == str):
            self.Net.savenet(netname)
            return
        else:
            print "Invalid input, netname must be a string!"
            return

    def loadnet(self,netname):
        if (type(netname) == str):
            self.Net.loadnet(netname)
            return
        else:
            print "Invalid input, netname must be a string!"
            return

    def snets(self):
        self.Net.snets()
        return

    
             
