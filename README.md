###Anduril
Anduril is a neural network library for python. It is written on top of NNet. 
A list of methods available to the Anduril object and their brief descriptions is given below:


####Standard Methods:
```python
Anduril(sconfig, classreg = 0, numcores = 1, gradd = 0, costfunc, epoch = -1)
```

This method is used to initilize the neural network. The architecture is specified by the string `sconfig` with `-` separated numbers. Anduril automatically detects the number of input and output layers depending on the input file, thus only the architecture of the hidden layers needs to be specified for sconfig. The user is required to specify what would be the purpose of the neural network, regression or classification. The variable `classreg` should be set to 1 for regression and 0 for classification. The user can specify the number of cores they have at thier disposal, if the number of cores is greater than 1 Anduril would use that to its advantage to perform parallel calculations. The user can also decide what gradient descent algorithim they prefer to use, set `gradd` to 0 for batch gradient descent and 1 for stochastic gradient descent. For classification the user can specify NNet to use the cross entropy cost function which can be done by specifying `costfunc` to be 1 (not available yet). If the user has selected stochastic gradient descent they can specify the number of epoch to perform using the `epoch` variable



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
Anduril.train_net(lrate, mode = 0, verbose = 0)
```

This methods trains the neural network using standard backpropogation, the learning rate is specified by the variable `lrate`. If `mode` is set to 1 it gives the current accuracy and/or RMSE of the neural net on the given data set. If verbose equals 1 more detailed output is produced giving the RMSE and accuracy as applicable after each training iteration.



```python
Anduril.train_rprop(mode = 0, verbose = 0,tmax = 15.0)
```

This method trains the neural network using resilient backpropogation. If `mode` is set to 1 it prints the RMSE and/or accuracy as the net is trained. The variable `tmax` sets an upper bound on the amount by which a particular weight can change. If verbose equals 1 more detailed output is produced giving the RMSE and accuracy as applicable after each training iteration.
.



```python
Anduril.test_file(filename, verbose = 0,ffmode = -1, sep1 = ",", sep2 = " ")
```

This method allows the user to upload a file to test the neural network against. `ffmode` is best left to its value of -1, else non-optimal weights and biases would be used to test the file. If verbose equals 1 more detailed output is produced giving the RMSE and accuracy as applicable after each training iteration.



```python
Anduril.test_net(mode = 0, verbose = 0)
```

This method allows the user to test thier neural network on the file uploaded in load. If `mode` is set to one then the neural network is tested on the validation set. If verbose equals 1 more detailed output is produced giving the RMSE and accuracy as applicable after each training iteration.
.



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

