#### Anduril
Anduril is a neural network library for python. It is written on top of NNet.
A list of methods available to the Anduril object and their brief descriptions is given below:


#### Standard Methods:
```python
Anduril.init(sconfig, classreg = 0, numcores = 1, gradd = 0, costfunc = 0)
```

This method is used to initilize the neural network. The architecture is specified by the string `sconfig` with `-` separated numbers. Anduril automatically detects the number of input and output layers depending on the input file, thus only the architecture of the hidden layers needs to be specified for sconfig. The user is required to specify what would be the purpose of the neural network, regression or classification. The variable `classreg` should be set to 1 for regression and 0 for classification. The user can specify the number of cores they have at thier disposal, if the number of cores is greater than 1 Anduril would use that to its advantage to perform parallel calculations. The user can also decide what gradient descent algorithim they prefer to use, set `gradd` to 0 for batch gradient descent and 1 for mini-batch gradient descent. For classification the user can specify NNet to use the cross entropy cost function which can be done by specifying `costfunc` to be 1 (not available yet).



```python
Anduril.func_arch(flayer)
```

The user can specify a very specific combinations of activation functions for the network by specifying thier choice as a string. E.g
`Anduril.func_arch("030")` would use the sigmoid function for the first hidden layer, a tanh + linear function for the second hidden layer and sigmoid again for the output layer. The list of available activation functions and thier corresponding numeric values are:
* `0`: Sigmoid function
* `1`: Tangent Hyperbolic
* `2`: Rectified linear
* `3`: Tangent Hyperbolic + Linear



```python
Anduril.load(filename, mode = 0, sep1 = ",", sep2 = " ")
```

This method is used to load the file into the network. The filename is specified by `filename`. If `mode` is 0 then the data is split into training, test and validation data, if `mode` is 1 then the entire file is used for training. This method assumes that the file contains both input and output data where each component of the input and output vector is seperated by the string specified in `sep1` and the input and output vectors are seperated by the string specified in sep2. By default sep1 is a comma (",") and sep2 is a space (" ").


```python
Anduril.train_net(epoch, lrate, mode = 1, verbose = 0, logfile = ' ')
```

This methods trains the neural network using stochastic gradient descent, the learning rate is specified by the variable `lrate`. The number of epochs to be run is specified by the `epoch` variable. If `mode` is set to 1 it gives the current accuracy and/or RMSE of the neural net on the given data set. If verbose equals 1 more detailed output is produced giving the RMSE and accuracy as applicable after each training iteration. If a logfile is specified then the log of the run is stored in a .and file along with a .dat file which contains RMSE of each epoch.
*Note: logfiles are only stored if mode = 1*



```python
Anduril.train_rprop(epoch, mode = 1, verbose = 0, logfile = ' ', tmax = 15.0)
```

This method trains the neural network using resilient backpropogation. The number of epochs to be run is specified by the `epoch` variable. If `mode` is set to 1 it prints the RMSE and/or accuracy as the net is trained. The variable `tmax` sets an upper bound on the amount by which a particular weight can change. If verbose equals 1 more detailed output is produced giving the RMSE and accuracy as applicable after each training iteration. If a logfile is specified then the log of the run is stored in a .and file along with a .dat file which contains RMSE of each epoch.
*Note: logfiles are only stored if mode = 1*



```python
Anduril.test_file(filename,netname = " ", sep1 = ",", sep2 = " ")
```

This method allows the user to upload a file to test the neural network against.  The variable `netname` is used to specify the network against which the file should be tested, if left to the default value it is tested against the net already loaded. 



```python
Anduril.test_net(mode = 0, verbose = 0)
```

This method allows the user to test thier neural network on the file uploaded in load. If `mode` is set to one then the neural network is tested on the validation set. If verbose equals 1 more detailed output is produced giving the RMSE and accuracy as applicable after each training iteration.



```python
Anduril.savenet(netname)
```

Save the given network with the a name specified in `netname`.



```python
Anduril.loadnet(netname)
```

Load the neural network saved with the name specified in `netname`.



```python
Anduril.snets()
```

View all saved neural networks in the given directory.


```python
Anduril.error_stats(num_bins)
```

This method can be used to examine the performance of a network on the train and test data set. It plots a histogram of the difference between the actual value and predicted value along with their means and standard deviaitons. Currently this only works with single variable regression. The `num_bins` variable is used to specify the number of bins of the histograms.
