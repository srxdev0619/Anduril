from NNet import *

class Anduril():

    def __init__(self):
        self.Net = NNet()
        return 
    
    def init(self,sconfig,classreg = 0, numcores = 1, gradd = 0, costfunc = 0, epoch = 1):
        self.Net.init(sconfig,classreg,numcores,gradd,costfunc,epoch)

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
            return;

    def test_file(self,filename, verbose = 0, netname = " ", sep1 = ",", sep2 = " "):
        if (type(filename) == str) and (type(verbose) == int) and (type(netname) == str) and (type(sep1) == str) and (type(sep2) == str):
            self.Net.test_file(filename, verbose, netname, sep1, sep2)
            return
        else:
            print "Invalid input type!"
            return

    def train_net(self,lrate, mode = 0, verbose = 0):
        if (type(lrate) == float) and (type(mode) == int) and (type(verbose) == int):
            self.Net.train_net(lrate,mode,verbose)
            return
        else:
            print "Invalid input, network not trained!"
            return

    def train_rprop(self,mode = 0, verbose = 0, tmax = 15.0):
        if (type(mode) == int) and (type(verbose) == int) and (type(tmax) == float):
            self.Net.train_rprop(mode,verbose,tmax)
            return
        else:
            print "Invalid input, network not trained!"
            return

    def d_trainrprop(self,mode = 0, verbose = 0, tmax = 15.0):
        if (type(mode) == int) and (type(verbose) == int) and (type(tmax) == float):
            self.Net.d_trainrprop(mode,verbose,tmax)
            return
        else:
            print "Invalid input, network not trained!"
            return

    def test_net(self,verbose = 0):
        if (type(verbose) == int):
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

    
    def ls_init(self,nconfig,classreg = 0,gradd = 0, costfunc = 0, epoch = 1):
        if (type(nconfig) == str) and (type(classreg) == int) and (type(gradd) == int) and (type(costfunc) == int) and (type(epoch) == int):
            self.Net.ls_init(nconfig,classreg,gradd,costfunc,epoch)
            return
        else:
            print "Invalid input!"
            return
    

    def ls_load(self,outputfiles, Qmatrix = " ", input_file = " ", sep1 = ","):
        if (type(outputfiles) == str) and (type(Qmatrix) == str) and (type(input_file) == str) and (type(sep1) == str):
            self.Net.ls_load(outputfiles,Qmatrix,input_file,sep1)
            return
        else:
            print "Invalid input!"
            return

    def l_load(self,Qmatrix = " ", input_file = " ", sep1 = ","):
        if (type(Qmatrix) == str) and (type(input_file) == str) and (type(sep1) == str):
            self.Net.l_load(Qmatrix,input_file,sep1)
            return
        else:
            print "Invalid input!"
            return

    def l_init(self,numfiles,classreg = 0,gradd = 0, costfunc = 0, epoch = 1):
        if (type(numfiles) == int) and (type(classreg) == int) and (type(gradd) == int) and (type(costfunc) == int) and (type(epoch) == int):
            self.Net.l_init(numfiles,classreg,1,gradd,costfunc,epoch)
            return
        else:
            print "Invalid input!"
            return

    def l_trainnet(self,numlatent, mode = 0, tol = -1.0):
        if (type(numlatent) == int) and (type(mode) == int) and (type(tol) == float):
            self.Net.l_trainnet(numlatent,mode,tol)
            return
        else:
            print "Invalid input, network not trained!"
            return

    def l_savenet(self):
        self.Net.l_savenet()
        return

    
    def test_data(self,in_filename, out_filename, netname, sep = ","):
        if (type(in_filename) == str) and (type(out_filename) == str) and (type(netname) == str) and (type(sep) == str):
            self.Net.test_data(in_filename, out_filename, netname, sep)
            return
        else:
            print "Invalid input!"
            return

        
    def l_trainrprop(self,numlatent,tmax = 15.0,mode = 0, tol = -1.0):
        if (type(numlatent) == int) and (type(tmax) == float) and (type(mode) == int) and (type(tol) == float):
            self.Net.l_trainrprop(numlatent,tmax,mode,tol)
            return
        else:
            print "Invalid input!"
            return

    def ld_trainrprop(self,numlatent,tmax = 15.0,mode = 0, tol = -1.0):
        if (type(numlatent) == int) and (type(tmax) == float) and (type(mode) == int) and (type(tol) == float):
            self.Net.ld_trainrprop(numlatent,tmax,mode,tol)
            return
        else:
            print "Invalid input!"
            return

    def testvoids(self,mode):
        if (type(mode) == int):
            self.Net.testvoids(mode)
            return
        else:
            print "Invalid input!"
            return

    def l_funcarch(self):
        self.Net.l_funcarch()
        return


    
             
