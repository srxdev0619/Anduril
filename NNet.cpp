#include<iostream>
#include<armadillo>
#include<string>
#include<cstdlib>
#include<cctype>
#include"NNet.h"
#include<cmath>
#include<boost/python.hpp>

using namespace std;
using namespace arma;

void NNet::parallel_bp(int index, int pos)
{
  backprop(xdata[index],ydata[index],pos);
} 

void NNet::l_parallelbp(int index, int pos)
{
  l_backprop(l_xvals[index],l_yvals[pos][index], pos);
}
//Contructor initilizes certain parameters
NNet::NNet()
{
  pcountx = -1;
  pcounty = -1;
  checkinit = -1;
  config = NULL;
  temp_rmse = 10000;
  min_rmse = -1;
  qmat = 0;
  trained = 0;
  l_trained = 0;
}


//Activation functions
double NNet::sigmoid(double x)
{
  return 1/(exp((-1)*x) + 1);
}

double NNet::tanh_net(double x)
{
  return tanh(x);
}

double NNet::tanh_r(double x)
{
  return tanh(x)+ 0.1*x;
  //return x;
}

double NNet::tanh_d(double x)
{
  return 1.0-(tanh(x)*tanh(x)) + 0.1;
  //return 1.0;
}

double NNet::tanh_dr(double x)
{
  return 1.0-(tanh(x)*tanh(x));
}

double NNet::reclinear(double x)
{
  if (x < 0)
    {
      return 0;
    }
  else
    {
      return x;
    }
}

double NNet::rec_D(double x)
{
  if (x < 0)
    {
      return 0;
    }
  else
    {
      return 1;
    }
}

double NNet::softplus(double x)
{
  return log(1 + exp(x));
}



//Initilization method, sets up the type and architecture of neural network to be used
void NNet::init(string sconfig, int iclassreg, int inumcores, int igradd, int icostfunc, int iepoch)
{
  int count = 0;
  int lent = sconfig.length();
  string num = "";
  //parses sconfig
  for (int i = 0; i < lent; i++)
    {
      if ((sconfig.at(i) != '-') && (isdigit(sconfig.at(i)) == 0))
	{
	  cout<<"Invalid input!\n";
	  return; 
	}
      else if ((sconfig.at(i) != '-') && (isdigit(sconfig.at(i)) != 0) && (i != (lent -1)))
	{
	  num = num + sconfig.at(i);
	}
      else if (sconfig.at(i) == '-')
	{
	  config = (int *)realloc(config,(count + 1)*sizeof(int));
	  config[count] = stoi(num,NULL);
	  count++;
	  num = "";
	}
      else if ((i == lent - 1) && (sconfig.at(i) != '-'))
	{
	  num = num + sconfig.at(i);
	  config = (int *)realloc(config,(count + 1)*sizeof(int));
	  config[count] = stoi(num,NULL);
	  count++;
	}
    }
  //sets variables
  classreg = iclassreg;
  numcores = inumcores;
  gradd = igradd;
  costfunc = icostfunc;
  epoch = iepoch;
  //checks network confgiuration
  if ((classreg > 1) || (gradd > 1) || (costfunc > 1) || (classreg < 0) || (gradd < 0))
    {
      cout<<"Invalid network configuration!\n";
      return;
    }
  if (epoch <= 0)
    {
      cout<<"Invalid configuration\nPlease choose the number of epochs to be trained";
      return;
    }
  for (int i = 0; i < count + 1; i++)
    {
      if (classreg == 0)
	{
	  funclayer.push_back(0);
	}
      else if (classreg == 1)
	{
	  funclayer.push_back(3);
	}
    }
  numhid = count;
  vector<mat> tr;
  for (int i = 0; i < numcores; i++)
    {
      activ.push_back(tr);
      sums.push_back(tr);
      grads.push_back(tr);
      dels.push_back(tr);
    }
  checkinit = 0;
  return;
}


//Defines the activation function of each layer
void NNet::func_arch(string flayer)
{
  int lent = flayer.length();
  if (lent != (numhid+1))
    {
      cout<<"Incorrect input!\n";
      return;
    }
  for (int i = 0; i < lent; i++)
    {
      if(isdigit(flayer.at(i)) == 0)
	{
	  cout<<"Incorrect input!\n";
	  return;
	}
      else
	{
	  string num = "";
	  num = num + flayer.at(i);
	  if (stoi(num) > 3)
	    {
	      cout<<"Incorrect input!\n";
	      return;
	    }
	  funclayer[i] = stoi(num);
	}
    }
  return;
}


//Load method, loads data from file to a neural network
void NNet::load(string filename,int imode, string sep1, string sep2)
{
  //check is init called, return if not
  if (checkinit == -1)
    {
      cout<<"Please initilize the Neural Network!\n";
      return;
    }
  loadmode = imode;
  if ((loadmode != 0) && (loadmode != 1))
    {
      cout<<"Load mode can only be 0 or 1"<<endl;
      return;
    }
  //open file, and initialize storage variables
  ifstream ldata(filename);
  if (!ldata.is_open())
    {
      cout<<"Error opening file!\n";
      return;
    }
  loadfile = filename;
  string temp;
  int numlines = 0;
  string decp = ".";
  string minussb = "-";
  //parse file input
  while (getline(ldata,temp))
    {
      int lent = temp.length();
      int track = 0;
      string num = "";
      int countx = 0;
      int county = 0;
      vector<double> yvals;
      vector<double> xvals;
      for(int i = 0; i < lent; i++)
	{
	  if  ((temp.at(i) != sep1.at(0)) && (temp.at(i) != sep2.at(0)) && (isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)))
	    {
	      cout<<temp.at(i)<<endl;
	      cout << "Invalid file format!\n";
	      return;
	    }
	  if (((temp.at(i) != sep1.at(0)) && (temp.at(i) != sep2.at(0)) && (i < (lent-1))) || (temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)))
	    {
	      num = num + temp.at(i);
	    }
	  else if ((temp.at(i) == sep1.at(0)) && (track == 0))
	    {
	      xvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	  else if (temp.at(i) == sep2.at(0))
	    {
	      track = 1;
	      xvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	  else if ((track == 1) && (temp.at(i) == sep1.at(0)))
	    {
	      yvals.push_back(stod(num,NULL));
	      num = "";
	      county++;
	    }
	  else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
	    {
	      num = num + temp.at(i);
	      yvals.push_back(stod(num,NULL));
	      num = "";
	      county++;
	    }
	}
      if (numlines == 0)
	{
	  pcountx = countx;
	  pcounty = county;
	}
      else if ((pcountx != countx) || (pcounty != county))
	{
	  cout<<"Invalid file format, change in size of vectors!\n";
	}
      mat mtempx(xvals);
      mat mtempy(yvals);
      xdata.push_back(mtempx);
      ydata.push_back(mtempy);
      numlines++;
    }
  if (loadmode == 0)
    {
      tests = numlines/5;
      validate = tests;
      train = numlines - tests - validate;
    }
  else if (loadmode == 1)
    {
      train = numlines;
    }
  else
    {
      cout<<"Please enter a value of 0 or 1 for loadmode\n";
      abort();
    }
  numdata = numlines;
  numlayers.push_back(pcountx);
  for (int i = 0; i < numhid; i++)
    {
      numlayers.push_back(config[i]);
    }
  numlayers.push_back(pcounty);
  //initializing parameter, bias and velocities
  for (int i = 0; i < numhid + 1; i++)
    {
      int rows = numlayers[i+1];
      int cols = numlayers[i];
      params.push_back(randn<mat>(rows,cols));
      velocity.push_back(zeros<mat>(rows,cols));
      mat a = zeros<mat>(rows,cols);
      mat b = zeros<mat>(rows,1);
      tgrads.push_back(a);
      bias.push_back(randn<mat>(rows,1));
      tdels.push_back(b);
    }
  ldata.close();
  return;
}


//Runs feedforward
void NNet::feed_forward(mat x,int gpos)
{
  int idx;
  if (gpos == -2)
    {
      idx = 0;
      if (!activ[idx].empty())
	activ[idx].clear();
      if (!sums.empty())
	sums[idx].clear();
      activ[idx].push_back(x);
      sums[idx].push_back(x);
      int count = 1;
      for (int i = 0; i < numhid+1; i++)
	{
	  activ[idx].push_back(best_params.at(i)*activ[idx].at(count-1) + best_bias.at(i));
	  sums[idx].push_back(activ[idx].at(count));
	  int lent = numlayers[count];
	  for (int j = 0; j < lent; j++)
	    {
	      if (funclayer[i] == 0)
		{
		  activ[idx].at(count)(j,0) = sigmoid(activ[idx].at(count)(j,0));
		}
	      else if (funclayer[i] == 1)
		{
		  activ[idx].at(count)(j,0) = tanh(activ[idx].at(count)(j,0));
		}
	      else if (funclayer.at(i) == 2)
		{
		  activ[idx].at(count)(j,0) = reclinear(activ[idx].at(count)(j,0));
		}
	      else
		{
		  //cout<<activ.size();
		  activ[idx].at(count)(j,0) = tanh_r(activ[idx].at(count)(j,0));
		  //cout<<activ.at(count)(0,0)<<" After\n";
		}
	    }
	  count++;
	}
      return;
    }
  if (gpos == -1)
    {
      idx = 0;
    }
  else
    {
      idx = gpos;
    }
  if (!activ[idx].empty())
    activ[idx].clear();
  if (!sums.empty())
    sums[idx].clear();
  activ[idx].push_back(x);
  sums[idx].push_back(x);
  int count = 1;
  for (int i = 0; i < numhid+1; i++)
    {
      activ[idx].push_back(params.at(i)*activ[idx].at(count-1) + bias.at(i));
      sums[idx].push_back(activ[idx].at(count));
      int lent = numlayers[count];
      for (int j = 0; j < lent; j++)
	{
	  if (funclayer[i] == 0)
	    {
	      activ[idx].at(count)(j,0) = sigmoid(activ[idx].at(count)(j,0));
	    }
	  else if (funclayer[i] == 1)
	    {
	      activ[idx].at(count)(j,0) = tanh(activ[idx].at(count)(j,0));
	    }
	  else if (funclayer.at(i) == 2)
	    {
	      activ[idx].at(count)(j,0) = reclinear(activ[idx].at(count)(j,0));
	    }
	  else
	    {
	      activ[idx].at(count)(j,0) = tanh_r(activ[idx].at(count)(j,0));
	    }
	}
      count++;
    }
  return;
}


//Backpropogaiton algorithm
void NNet::backprop(mat x, mat y, int gpos)
{
  int idx;
  if (gpos == -1)
    {
      idx = 0;
    }
  else
    {
      idx = gpos;
    }
  feed_forward(x,gpos);
  if(!grads.empty())
    { 
      grads[idx].clear();
    }
  if(!dels.empty())
    {
      dels[idx].clear();
    }
  if (costfunc == 0)
    {
      if (funclayer[numhid] == 0)
	{
	  dels.at(idx).push_back((activ.at(idx).at(numhid+1) - y)%(activ[idx][numhid+1] - (activ.at(idx).at(numhid+1)%activ.at(idx).at(numhid+1))));
	}
      else if (funclayer[numhid] == 1)
	{
	  dels[idx].push_back((y - activ[idx][numhid+1])%(ones<mat>(numlayers[numhid+1],1) - activ[idx][numhid+1]%activ[idx][numhid+1]));
	}
      else if(funclayer[numhid] == 2)
	{
	  for (int i = 0; i < numlayers[numhid+1]; i++)
	    {
	      sums[idx][numhid + 1](i,0) = rec_D(sums[idx][numhid + 1](i,0));
	    }
	  dels[idx].push_back((activ[idx][numhid+1]-y)%sums[idx][numhid+1]);
	}
      else if (funclayer[numhid] == 3)
	{
	  for (int i = 0; i < numlayers[numhid+1]; i++)
	    {
	      sums[idx][numhid + 1](i,0) = tanh_d(sums[idx][numhid + 1](i,0));
	    }
	  dels[idx].push_back((activ[idx][numhid+1]-y)%sums[idx][numhid+1]);
	}
    }
  else
    {
      //TODO: have to complete results for other cost functions
      return;
    }
  int count = 1;
  for (int i = 0; i < numhid; i++)
    {
      mat temp;
      temp = (params[numhid - i].t())*(dels[idx][i]);
      mat derv = sums[idx][numhid - i];
      //Calculates the derivatives at each layer according to the corresponding activation function
      if (funclayer[numhid - count] == 0)
	{ 
	  derv = activ[idx][numhid- i] - (activ[idx][numhid - i]%activ[idx][numhid - i]);
	}
      else if (funclayer[numhid - count] == 1)
	{
	  int n = numlayers[numhid - i];
	  for (int j = 0; j < n; j++)
	    {
	      derv(j,0) = tanh_dr(derv(j,0));
	    }
	}
      else if (funclayer[numhid - count] == 2)
	{
	  int n = numlayers[numhid - i];
	  for (int j = 0; j < n; j++)
	    {
	      derv(j,0) = rec_D(derv(j,0));
	    }
	}
      else if (funclayer[numhid-count] == 3)
	{
	  int n = numlayers[numhid - i];
	  for (int j = 0; j < n; j++)
	    {
	      derv(j,0) = tanh_d(derv(j,0));
	    }
	}
      temp = temp%derv;
      dels[idx].push_back(temp);
      count++;
    }
  for (int i = 0; i < numhid + 1; i++)
    {
      grads[idx].push_back(dels[idx][i]*(activ[idx][numhid - i].t()));
    }
  return;
}


//Train the neural network
void NNet::train_net(double lrate, int mode, int verbose)
{
  int trainmode = mode;
  vector<thread> bpthreads;
  if ((trainmode != 0) && (trainmode != 1))
    {
      cout<<"Training mode can only be 0 or 1"<<endl;
      return;
    }
  for(int i = 0; i < numhid + 1; i++)
    {
      tgrads[i].fill(0);
      tdels[i].fill(0);
    }
  double ttemp_rmse;
  int ecount = 0;
  if (gradd == 0)
    {
      double beta = 0.2;
      for (int k = 0; k < epoch; k++)
	{
	  cout<<(double)k*100/(double)epoch<<"%\n";
	  for (int i = 0; i < train; i++)
	    {
	      for (int l = 0; l < numhid + 1; l++)
		{
		  params.at(l) = params.at(l) + beta*velocity.at(l);
		}
	      backprop(xdata[i],ydata[i],-1);
	      for (int q = 0; q < numcores; q++)
		{
		  for (int j = 0; j < numhid + 1; j++)
		    {
		      tgrads[j] = tgrads[j] + grads[q][numhid-j];
		      tdels[j] = tdels[j] + dels[q][numhid - j];
		    }
		}
	    }
	  if (k > 500.0/2.0)
	    {
	      beta = 0.9;
	    }
	  for (int l = 0; l < numhid + 1; l++)
	    {
	      velocity.at(l) = beta*velocity.at(l) - (lrate/(double)train)*tgrads.at(l);
	      params.at(l) = params.at(l) + velocity.at(l);
	      bias[l] = bias[l] - (lrate/(double)train)*tdels[l];
	    }
	}
    }
  else if (gradd == 1)
    {
      cout<<"Initializing Stochastic Gradient Descent\n";
      //min_rmse = -1;
      vector<int> idxs;
      for (int i = 0; i < train; i++)
	{
	  idxs.push_back(i);
	}
      random_shuffle(idxs.begin(),idxs.end());
      double beta;
      if (classreg == 0)
	{
	  beta = 0.2;
	}
      for (int i = 0; i < epoch; i++)
	{
	  cout<<((double)i/(double)epoch)*100<<"%\n";
	  int step = 0;
	  if (classreg == 0)
	    {
	      if ( i < (epoch/2.0))
		{
		  beta = 0.2;
		}
	      else
		{
		  beta = 0.9;
		}
	    }
	  while (step < train)
	    {
	      int k = step;
	      step = min(step + 10,train);
	      if (classreg == 0)
		{
		  for (int yt = 0; yt < numhid + 1; yt++)
		    {
		      params.at(yt) = params.at(yt) + beta*velocity.at(yt);
		    }
		}
	      int ncore = min(numcores,10);
	      for(;k < step; k = k + ncore)
		{
		  //cout<<idxs.at(k)<<endl;
		  if (!bpthreads.empty())
		    {
		      bpthreads.clear();
		    }
		  for (int t = 0; t < ncore; t++)
		    {
		      if (numcores > 1)
			{ bpthreads.push_back(std::thread(&NNet::parallel_bp,this,idxs.at(k+t),t));
			}
		      else
			{
			  //Threads create an overhead which is avoided if there is only one core available
			    backprop(xdata.at(idxs.at(k+t)),ydata.at(idxs.at(k+t)),-1);
			}
		    }
		  if (numcores > 1)
		    {
		      for(int t = 0; t < ncore; t++)
			{
			  bpthreads[t].join();
			}
		    }
		  for (int q = 0; q < ncore; q++)
		    {
		      for(int j = 0; j < numhid + 1; j++)
			{
			  tgrads.at(j) = tgrads.at(j) + grads[q].at(numhid - j);
			  tdels.at(j) = tdels.at(j) + dels[q].at(numhid -j);
			}
		    }
		}
	      if (classreg == 0)
		{
		  for (int yt = 0; yt < numhid + 1; yt++)
		    {
		      params.at(yt) = params.at(yt) - beta*velocity.at(yt);
		      velocity.at(yt) = beta*velocity.at(yt) - (lrate/(double)1000.0)*tgrads.at(yt);
		    }
		}
	      double kappa = 0.001;
	      for (int j = 0; j < numhid + 1; j++)
		{
		  //the below only applies if regression is going.....
		  if (classreg == 1)
		    {
		      if (funclayer.at(j) == 3)
			{
			  kappa = 0.2;
			}
		      else
			{
			  kappa = 0.001;
			}
		      params.at(j) = params.at(j) - (lrate/(double)100.0)*tgrads.at(j) - kappa*params.at(j);
		      //tgrads.at(j).fill(0); //Doing this is the textbook method but commenting this out just works much much better
		    }
		  else
		    {
		      params.at(j) = params.at(j) + velocity.at(j) - kappa*params.at(j);
		    }
		  bias.at(j) = bias.at(j) - (lrate/(double)100.0)*tdels.at(j);
		  //tdels.at(j).fill(0);   //Doing this is the textbook method but commenting this out just works much much better
		}
	    }
	  //Below takes extra measures so that the network converges
	  if (trainmode == 1)
	    {
	      ttemp_rmse = temp_rmse;
	      if (loadmode == 1)
		{
		  test_file(loadfile,verbose);
		}
	      else 
		{
		  test_net(1,verbose);
		}
	      if (min_rmse == -1)
		{
		  min_rmse = temp_rmse;
		}
	      if (temp_rmse > ttemp_rmse)
		{
		  if (ecount <= 0)
		    {
		      double kappa = 0.001;
		      for (int j = 0; j < numhid + 1; j++)
		      {
			  //the below only applies if regression is going.....
		        if (classreg == 1)
		          {
		            if (funclayer.at(j) == 3)
			      {
				kappa = 0.099;
			      }
		            else
			      {
				kappa = 0.001;
			      }
		          }
		        params.at(j) = params.at(j) + (lrate/(double)100.0)*tgrads.at(j) + kappa*params.at(j);
		        bias.at(j) = bias.at(j) + (lrate/(double)100.0)*tdels.at(j);
		      }
		      lrate = 0.99*lrate;
		      ecount = (int)((double)epoch/(double)10);
		    }
		  else
		    {
		      ecount = ecount - 1;
		    } 
		}
	      if (temp_rmse < min_rmse)
		{
		  min_rmse = temp_rmse;
		  if (!best_params.empty())
		    {
		      best_params.clear();
		    }
		  if(!best_bias.empty())
		    {
		      best_bias.clear();
		    }
		  if (!best_velocity.empty())
		    {
		      best_velocity.clear();
		    }
		  for (int r = 0; r < numhid + 1; r++)
		    {
		      best_params.push_back(params.at(r));
		      best_bias.push_back(bias.at(r));
		      best_velocity.push_back(velocity.at(r));
		    }
		}
	      cout<<endl;
	    }
	}
    }
  return;
}


//Trains the network accroding to RPROP
void NNet::train_rprop(int mode,int verbose,double tmax)
{
  int trainmode = mode;
  vector<thread> bpthreads;
  double rmax = tmax;
  if ((trainmode != 0) && (trainmode != 1))
    {
      cout<<"Training mode can only be 0 or 1"<<endl;
      return;
    }
  for(int i = 0; i < numhid + 1; i++)
    {
      tgrads[i].fill(0);
      tdels[i].fill(0);
    }
  int rprop = 0;
  double ttemp_rmse;
  int ecount = 0;
  if (gradd == 0)
    {
      for (int k = 0; k < epoch; k++)
	{
	  cout<<(double)k*100/(double)epoch<<"%\n";
	  for (int i = 0; i < train; i++)
	    {
	      for (int t = 0; t < numcores; t++)
		{
		  bpthreads.push_back(std::thread(&NNet::parallel_bp,this,i,t));
		}
	      for (int t = 0; t < numcores; t++)
		{
		  bpthreads[t].join();
		}
	      for (int q = 0; q < numcores; q++)
		{
		  for (int j = 0; j < numhid + 1; j++)
		    {
		      tgrads[j] = tgrads[j] + grads[q][numhid-j];
		      tdels[j] = tdels[j] + dels[q][numhid - j];
		    }
		}
	    }
	  if (rprop == 0)
	    {
	      for(int q = 0; q < numhid + 1; q++)
		{
		  checkgrads.push_back(tgrads[q]);
		  checkdels.push_back(tdels[q]);
		}
	    }
	  else
	    {
	      for(int q = 0; q < numhid + 1; q++)
		{
		  //Weights update
		  int rows = checkgrads[q].n_rows;
		  int cols = checkgrads[q].n_cols;
		  for(int rw = 0; rw < rows; rw++)
		    {
		      for(int cl = 0; cl < cols; cl++)
			{
			  if (checkgrads[q](rw,cl)*tgrads[q](rw,cl) > 0) 
			    {
			      //push up weight
			      if (rprop == 1)
				{
				  double sign = tgrads[q](rw,cl)/abs(tgrads[q](rw,cl));
				  tgrads[q](rw,cl) = 0.1*1.2*(tgrads[q](rw,cl)/abs(tgrads[q](rw,cl)));
				  tgrads[q](rw,cl) = min(tgrads[q](rw,cl),rmax);
				  checkgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				  tgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				}
			      else
				{
				  double sign = tgrads[q](rw,cl)/abs(tgrads[q](rw,cl));
				  tgrads[q](rw,cl) = sign*checkgrads[q](rw,cl)*1.2;
				  tgrads[q](rw,cl) = min(tgrads[q](rw,cl),rmax);
				  checkgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				  tgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				}
			    }
			  else if ((checkgrads[q](rw,cl)*tgrads[q](rw,cl) < 0))
			    {
			      //pushdown weight
			      if (rprop == 1)
				{
				  double sign = tgrads[q](rw,cl)/abs(tgrads[q](rw,cl));
				  double temp;
				  temp = 0.1*0.5;
				  temp = max(temp,0.000001);
				  checkgrads[q](rw,cl) = sign*temp;
				  tgrads[q](rw,cl) = sign*temp;
				}
			      else
				{
				  double sign = tgrads[q](rw,cl)/abs(tgrads[q](rw,cl));
				  double temp;
				  temp = abs(checkgrads[q](rw,cl))*0.5;
				  temp = max(abs(temp),0.000001);
				  checkgrads[q](rw,cl) = sign*temp;
				  tgrads[q](rw,cl) = sign*temp;
				}
			    }
			  else if ((checkgrads[q](rw,cl)*tgrads[q](rw,cl) == 0))
			    {
			      if (rprop == 1)
				{
				  tgrads[q](rw,cl) = 0.1*1.0*(tgrads[q](rw,cl)/abs(tgrads[q](rw,cl)));
				}
			      else
				{
				  tgrads[q](rw,cl) = abs(checkgrads[q](rw,cl))*1.0*(tgrads[q](rw,cl)/abs(tgrads[q](rw,cl)));
				}
			    }
			}
		    }
		  //BIAS
		  int brows = checkdels[q].n_rows;
		  int bcols = checkdels[q].n_cols;
		  for(int rw = 0; rw < brows; rw++)
		    {
		      for(int cl = 0; cl < bcols; cl++)
			{
			  if (checkdels[q](rw,cl)*tdels[q](rw,cl) > 0)
			    {
			      //push up bias
			      if (rprop == 1)
				{
				  double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				  tdels[q](rw,cl) = 0.1*1.2*(tdels[q](rw,cl)/abs(tdels[q](rw,cl)));
				  tdels[q](rw,cl) = min(tdels[q](rw,cl),rmax);
				  checkdels[q](rw,cl) = sign*tdels[q](rw,cl);
				  tdels[q](rw,cl) = sign*tdels[q](rw,cl);
				}
			      else
				{
				  double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				  tdels[q](rw,cl) = sign*checkdels[q](rw,cl)*1.2;
				  tdels[q](rw,cl) = min(tdels[q](rw,cl),rmax);
				  checkdels[q](rw,cl) = sign*tdels[q](rw,cl);
				  tdels[q](rw,cl) = sign*tdels[q](rw,cl);
				}
			    }
			  else if ((checkdels[q](rw,cl)*tdels[q](rw,cl) < 0))
			    {
			      //pushdown bias
			      if (rprop == 1)
				{
				  double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				  double temp;
				  temp = 0.1*0.5;
				  temp = max(temp,0.000001);
				  checkdels[q](rw,cl) = sign*temp;
				  tdels[q](rw,cl) = sign*temp;
				}
			      else
				{
				  double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				  double temp;
				  temp = checkdels[q](rw,cl)*0.5;
				  temp = max(abs(temp),0.000001);
				  checkdels[q](rw,cl) = sign*temp;
				  tdels[q](rw,cl) = sign*temp;
				}
			    }
			  else if ((checkdels[q](rw,cl)*tdels[q](rw,cl) == 0))
			    {
			      if (rprop == 1)
				{
				  tdels[q](rw,cl) = 0.1*1.0*(tdels[q](rw,cl)/abs(tdels[q](rw,cl)));
				}
			      else
				{
				  tdels[q](rw,cl) = abs(checkdels[q](rw,cl))*1.0*(tdels[q](rw,cl)/abs(tdels[q](rw,cl)));
				}
			    }
			}
		    }
		}
	    }
	  for (int l = 0; l < numhid + 1; l++)
	    {
	      if (rprop == 0)
		{
		  params[l] = params[l] - (0.000001)*tgrads[l]; - 0.00001*params[l];
		  bias[l] = bias[l] - (0.0001/(double)train)*tdels[l];
		  rprop++;
		}
	      else
		{
		  params[l] = params[l] - tgrads[l];
		  bias[l] = bias[l] - tdels[l];
		  tgrads[l].fill(0);
		  tdels[l].fill(0);
		}
	    }
	  if (rprop >= 1)
	    {
	      rprop = 3;
	    }
	  else
	    {
	      rprop++;
	    }
	}
    }
  else if (gradd == 1)
    {
      cout<<"Initializing Stochastic Gradient Descent\n";
      vector<int> idxs;
      for (int i = 0; i < train; i++)
	{
	  idxs.push_back(i);
	}
      random_shuffle(idxs.begin(),idxs.end());
      for (int i = 0; i < epoch; i++)
	{
	  cout<<((double)i/(double)epoch)*100<<"%\n";
	  int step = 0;
	  while (step < train)
	    {
	      int k = step;
	      step = min(step + 10,train);
	      int ncore = min(numcores,10);
	      for(;k < step; k = k + ncore)
		{
		  if (!bpthreads.empty())
		    {
		      bpthreads.clear();
		    }
		  for (int t = 0; t < ncore; t++)
		    {
		      if (numcores > 1)
			{ bpthreads.push_back(std::thread(&NNet::parallel_bp,this,idxs.at(k+t),t));
			}
		      else
			{
			  //Threads create an overhead which is avoided if there is only one core available
			    backprop(xdata.at(idxs.at(k+t)),ydata.at(idxs.at(k+t)),-1);
			}
		    }
		  if (numcores > 1)
		    {
		      for(int t = 0; t < ncore; t++)
			{
			  bpthreads[t].join();
			}
		    }
		  for (int q = 0; q < ncore; q++)
		    {
		      for(int j = 0; j < numhid + 1; j++)
			{
			  tgrads.at(j) = tgrads.at(j) + grads[q].at(numhid - j);
			  tdels.at(j) = tdels.at(j) + dels[q].at(numhid -j);
			}
		    }
		}
	      if (rprop == 0)
		{
		  for(int q = 0; q < numhid + 1; q++)
		    {
		      checkgrads.push_back(tgrads[q]);
		      checkdels.push_back(tdels[q]);
		    }
		}
	      else
		{
		  for(int q = 0; q < numhid + 1; q++)
		    {
		      //Weights update
		      int rows = checkgrads[q].n_rows;
		      int cols = checkgrads[q].n_cols;
		      for(int rw = 0; rw < rows; rw++)
			{
			  for(int cl = 0; cl < cols; cl++)
			    {
			      if ((checkgrads[q](rw,cl)*tgrads[q](rw,cl) > 0))
				{
				  //push up weight
				  if (rprop == 1)
				    {
				      double sign = (tgrads[q](rw,cl)/abs(tgrads[q](rw,cl)));
				      tgrads[q](rw,cl) = 0.1*1.2;
				      tgrads[q](rw,cl) = min(tgrads[q](rw,cl),rmax);
				      checkgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				      tgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				    }
				  else
				    {
				      double sign = tgrads[q](rw,cl)/abs(tgrads[q](rw,cl));
				      tgrads[q](rw,cl) = sign*checkgrads[q](rw,cl)*1.2;
				      tgrads[q](rw,cl) = min(tgrads[q](rw,cl),rmax);
				      checkgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				      tgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
				    }
				}
			      else if ((checkgrads[q](rw,cl)*tgrads[q](rw,cl) < 0))
				{
				  //pushdown weight
				  if (rprop == 1)
				    {
				      double sign = tgrads[q](rw,cl)/abs(tgrads[q](rw,cl));
				      double temp;
				      temp = 0.1*0.5;
				      temp = max(temp,0.000001);
				      checkgrads[q](rw,cl) = sign*temp;
				      tgrads[q](rw,cl) = sign*temp;
				    }
				  else
				    {
				      double sign = tgrads[q](rw,cl)/abs(tgrads[q](rw,cl));
				      double temp;
				      temp = checkgrads[q](rw,cl)*0.5;
				      temp = max(abs(temp),0.0000001);
				      checkgrads[q](rw,cl) = sign*temp;
				      tgrads[q](rw,cl) = sign*temp;
				    }
				}
			      else if ((checkgrads[q](rw,cl)*tgrads[q](rw,cl) == 0))
				{
				  if (rprop == 1)
				    {
				      tgrads[q](rw,cl) = 0.1*1.0*(tgrads[q](rw,cl)/abs(tgrads[q](rw,cl)));
				    }
				  else
				    {
				      tgrads[q](rw,cl) = abs(checkgrads[q](rw,cl))*1.0*(tgrads[q](rw,cl)/abs(tgrads[q](rw,cl)));
				    }
				}
			    }
			}
		      //BIAS update
		      int brows = checkdels[q].n_rows;
		      int bcols = checkdels[q].n_cols;
		      for(int rw = 0; rw < brows; rw++)
			{
			  for(int cl = 0; cl < bcols; cl++)
			    {
			      if (checkdels[q](rw,cl)*tdels[q](rw,cl) > 0)
				{
				  //push up bias
				  if (rprop == 1)
				    {
				      double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				      tdels[q](rw,cl) = 0.1*1.2;
				      tdels[q](rw,cl) = min(tdels[q](rw,cl),rmax);
				      checkdels[q](rw,cl) = sign*tdels[q](rw,cl);
				      tdels[q](rw,cl) = sign*tdels[q](rw,cl);
				    }
				  else
				    {
				      double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				      tdels[q](rw,cl) = sign*checkdels[q](rw,cl)*1.2;
				      tdels[q](rw,cl) = min(tdels[q](rw,cl),rmax);
				      checkdels[q](rw,cl) = sign*tdels[q](rw,cl);
				      tdels[q](rw,cl) = sign*tdels[q](rw,cl);
				    }
				}
			      else if ((checkdels[q](rw,cl)*tdels[q](rw,cl) < 0))
				{
				  //pushdown weight
				  if (rprop == 1)
				    {
				      double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				      tdels[q](rw,cl) = -1*checkdels[q](rw,cl);
				      double temp;
				      temp = 0.1*0.5;
				      temp = max(temp,0.000001);
				      checkdels[q](rw,cl) = sign*temp;
				      tdels[q](rw,cl) = sign*temp;
				    }
				  else
				    {
				      double sign = tdels[q](rw,cl)/abs(tdels[q](rw,cl));
				      double temp;
				      temp = checkdels[q](rw,cl)*0.5;
				      temp = max(abs(temp),0.000001);
				      checkdels[q](rw,cl) = sign*temp;
				      tdels[q](rw,cl) = sign*temp;
				    }
				}
			      else if ((checkdels[q](rw,cl)*tdels[q](rw,cl) == 0))
				{
				  if (rprop == 1)
				    {
				      tdels[q](rw,cl) = 0.1*1.0*(tdels[q](rw,cl)/abs(tdels[q](rw,cl)));
				    }
				  else
				    {
				      tdels[q](rw,cl) = abs(checkdels[q](rw,cl))*1.0*(tdels[q](rw,cl)/abs(tdels[q](rw,cl)));
				    }
				}
			    }
			}
		    }
		}
	      for (int j = 0; j < numhid + 1; j++)
		{
		  if (rprop == 0)
		    {
		      params[j] = params[j] - (0.000001)*tgrads[j]; - 0.00001*params[j];
		      bias[j] = bias[j] - (0.00001/(double)train)*tdels[j];
		      rprop++;
		      tgrads[j].fill(0);
		      tdels[j].fill(0);
		    }
		  else
		    {
		      params[j] = params[j] - tgrads[j];
		      bias[j] = bias[j] - tdels[j];
		      tgrads[j].fill(0);
		      tdels[j].fill(0);
		    }
		}
	      if(rprop >= 1)
		{
		  rprop = 3;
		}
	      else
		{
		  rprop++;
		}
	    }
	  //Below takes extra measures so that the network converges
	  if (trainmode == 1)
	    {
	      ttemp_rmse = temp_rmse;
	      if (loadmode == 1)
		{
		  test_file(loadfile,verbose);
		}
	      else 
		{
		  test_net(1,verbose);
		}
	      if (min_rmse == -1)
		{
		  min_rmse = temp_rmse;
		}
	      if (temp_rmse > ttemp_rmse)
		{
		  //double kappa = 0.001;
		  cout<<"RMSE reversed!\n";
		  //cout<<lrate<<endl;
		  if (ecount <= 0)
		    {
		      ecount = (int)((double)epoch/(double)10);
		    }
		  else
		    {
		      //lrate = lrate;
		      ecount = ecount - 1;
		    }
		  //cout<<lrate<<endl;
		}
	      if (temp_rmse < min_rmse)
		{
		  min_rmse = temp_rmse;
		  //cout<<"Minima!\n";
		  if (!best_params.empty())
		    {
		      best_params.clear();
		    }
		  if(!best_bias.empty())
		    {
		      best_bias.clear();
		    }
		  for (int r = 0; r < numhid + 1; r++)
		    {
		      best_params.push_back(params.at(r));
		      best_bias.push_back(bias.at(r));
		      best_velocity.push_back(velocity.at(r));
		    }
		}
	      cout<<endl;
	    }
	}
    }
  return;
}


//Test the network
void NNet::test_net(int testmode, int verbose)
{
   if (loadmode != 0)
    {
      cout<<"Please use test_file(filename) to test this net!\n";
      return;
    }
   int start;
   int stop;
   if (testmode == 0)
     {
       start = train + validate;
       stop = numdata;
     }
   else if (testmode == 1)
     {
       start = validate;
       stop = train+validate;
     }
   else
     {
       cout<<"Testmode can only be 0 or 1!"<<endl;
       abort();
     }
  int passed = 0;
  int error = 0;
  for (int i = start; i < stop; i++)
    {
      feed_forward(xdata[i],-1);
      if (classreg == 0)
	{
	  int max = 0;
	  int idx = 0;
	  int lent = activ[0][numhid + 1].n_rows;
	  int alent = numhid + 1;
	  for (int j = 0; j < lent; j++)
	    {
	      if (j == 0)
		{
		  max = activ[0][alent](j,0);
		  idx = j;
		}
	      else
		{
		  if (activ[0][alent](j,0) > max)
		    {
		      max = activ[0][alent](j,0);
		      activ[0][alent](j,0)  = 0.0;
		      idx = j;
		    }
		  else
		    {
		      activ[0][alent](j,0) = 0.0;
		      continue;
		    }
		}
	    }
	  activ[0][idx](idx,0) = 1.0;
	  if (abs(activ[0][alent](idx,0) - ydata[i](idx,0)) <= 0.1)
	    {
	      passed++;
	    }
	}
      else
	{
	  int chk = 1;
	  int lent = activ[0][numhid + 1].n_rows;
	  for (int j = 0; j < lent; j++)
	    {
	      error = error + pow(ydata[i](j,0) - activ[0][numhid + 1](j,0),2);
	      if (abs(activ[0][numhid + 1](j,0) - ydata[i](j,0)) <= 0.1)
		{
		  continue;
		}
	      else
		{
		  chk = 0;
		  break;
		}
	    }
	  if (chk)
	    {
	      passed++;
	    }
	}
    }
  double hitrate = ((double)passed/(double)(stop-start))*100;
  double RMSE = sqrt((error/(double)(stop-start)));
  if (verbose == 1)
    {
      cout<<"The accuracy is: "<<hitrate<<"%\n";
    }
  if(classreg == 1)
    {
      temp_rmse = RMSE;
      if (verbose == 1)
	{
	  cout<<"RMSE: "<<RMSE<<endl;
	}
    }
  else
    {
      temp_rmse = hitrate;
      if (verbose == 1)
	{
	  cout<<"Passed: "<<passed<<endl;
	}
    }
  return;
}


//This method saves the current Neural Network
void NNet::savenet(string netname)
{
  fstream savednets;
  int ckopen = 0;
  savednets.open(".savednets",fstream::in);
  if (!savednets.is_open())
    {
      ckopen = 0;
    }
  else
    {
      ckopen = 1;
      savednets.close();
    }
  if (ckopen == 0)
    {
      savednets.open(".savednets",fstream::out);
      savednets.close();
    }
  savednets.open(".savednets",fstream::in);
  if (!savednets.is_open())
    {
      cout<<"Failed to open file 2\n";
      return;
    }
  string temp;
  string yes = "y";
  string no = "no";
  string stp = "*";
  int check = 0;
  vector<string> names;
  while(getline(savednets,temp))
    {
      int lent1 = temp.length();
      int lent2 = netname.length();
      for (int i = 0; (temp.at(i) != stp.at(0)) && (i < lent1) && (i < lent2); i++)
	{
	  if (temp.at(i) == netname.at(i))
	    {
	      check = 1;
	    }
	  else
	    {
	      check = 0;
	    }
	}
      if (check == 1)
	{
	  string ans;
	  int count = 0;
	  do
	    {
	      if (count == 0)
		{
		  cout<<"A neural network with the same name already exists, saving will overwrite the previous save. Do you wish to continue (y or n) ?: ";
		  cin>>ans;
		}
	      else
		{
		  cout<<"Continue with overwrite ? (y or n): ";
		  cin>>ans;
		}
	      count++;
	    }
	  while(((ans.at(0) != yes.at(0)) || ans.at(0) == no.at(0)) && (ans.length() != 1));
	  if (ans.at(0) == yes.at(0))
	    {
	      break;
	    }
	  else
	    {
	      cout<<"Save aborted!\n";
	      return;
	    }
	}
      else
	{
	   names.push_back(temp);
	}
    }
  savednets.close();
  savednets.open(".savednets",fstream::out);
  names.push_back(netname + "*" + to_string(numhid + 1));
  int lent = names.size();
  for (int i = 0; i < lent; i++)
    {
      savednets<<names.at(i)<<endl;
    }
  netname = "." + netname + "_";
  for (int i = 0; i < numhid + 1; i++)
    {
      if (!best_params.empty())
	{
	  best_params.at(i).save(netname + "p" + to_string(i));
	  best_bias.at(i).save(netname + "b" + to_string(i));
	}
      else
	{
	  params.at(i).save(netname + "p" + to_string(i));
	  bias.at(i).save(netname + "b" + to_string(i));
	}
    }
  cout<<"Saved\n";
  savednets.close();
  return;
}


//The below method loads a saved neural network
void NNet::loadnet(string netname)
{
  fstream savednets;
  savednets.open(".savednets",fstream::in);
  string temp;
  string num = "";
  int chk = 1;
  string spchar = "*";
  while(getline(savednets,temp))
    {
      chk = 1;
      int lent = temp.length();
      int chk1 = 0;
      int lent1 = netname.length();
      if (lent1 > lent)
	{
	  cout<<"No net exists with such a name!\n";
	  return;
	}
      for (int j = 0; j < lent; j++)
	{
	  if (temp.at(j) == spchar.at(0))
	    {
	      chk1 = 1;
	    }
	  else if ((netname.at(j) != temp.at(j)) && (chk1 == 0))
	    {
	      chk = 0;
	    }
	  else if ((chk1 == 1) && (chk == 1))
	    {
	      num = num + temp.at(j);
	    }
	}
      if (chk == 1)
	{
	  break;
	}
    }
  if (chk == 0)
    {
      cout<<netname<<" not found!\n";
      return;
    }
  else
    {
      int numparams = stoi(num,NULL);
      numhid = numparams - 1;
      netname = "." + netname + "_";
      for (int i = 0; i < numparams; i++)
	{
	  bool stat1 = params.at(i).load(netname + "p" + to_string(i));
	  bool stat2 = bias.at(i).load(netname + "b" + to_string(i));
	  if ((stat1 != true) || (stat2 != true))
	    {
	      cout<<"Load failed!\n";
	      abort();
	      return;
	    }
	}
    }
  checkinit = 0;
  return;
}


//This method prints saved neural networks
void NNet::snets(void)
{
  ifstream savednets;
  savednets.open(".savednets");
  string name = "";
  string temp;
  string spchar = "*";
  cout<<"Saved Neural Networks\n";
  while (getline(savednets,temp))
    {
      for (int i = 0; temp.at(i) != spchar.at(0); i++)
	{
	  name = name + temp.at(i);
	}
      cout<<name<<endl;
      name = "";
    }
  return;
}


//This method is to test data of a specific file
void NNet::test_file(string filename,int verbose,int ffmode, string sep1, string sep2)
{
  if (loadmode != 1)
    {
      cout<<"Please use test_net() to test this neural net!\n";
      return;
    }
  ifstream ldata(filename);
  if (!ldata.is_open())
    {
      cout<<"Error opening file!\n";
      return;
    }
  string temp;
  int numlines = 0;
  string decp = ".";
  string minussb = "-";
  int feedforwardmode = ffmode;
  //parse file input
  while (getline(ldata,temp))
    {
      int lent = temp.length();
      int track = 0;
      string num = "";
      int countx = 0;
      int county = 0;
      vector<double> yvals;
      vector<double> xvals;
      for(int i = 0; i < lent; i++)
	{
	  if  ((temp.at(i) != sep1.at(0)) && (temp.at(i) != sep2.at(0)) && (isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)))
	    {
	      cout << "Invalid file format!\n";
	      return;
	    }
	  if (((temp.at(i) != sep1.at(0)) && (temp.at(i) != sep2.at(0)) && (i < (lent-1))) || (temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)))
	    {
	      num = num + temp.at(i);
	    }
	  else if ((temp.at(i) == sep1.at(0)) && (track == 0))
	    {
	      xvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	  else if (temp.at(i) == sep2.at(0))
	    {
	      track = 1;
	      xvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	  else if ((track == 1) && (temp.at(i) == sep1.at(0)))
	    {
	      yvals.push_back(stod(num,NULL));
	      num = "";
	      county++;
	    }
	  else if (i == (lent - 1))
	    {
	      num = num + temp.at(i);
	      yvals.push_back(stod(num,NULL));
	      num = "";
	      county++;
	    }
	}
      if ((pcountx != countx) || (pcounty != county))
	{
	  cout<<"Invalid file format, change in size of vectors!\n";
	}
      mat mtempx(xvals);
      mat mtempy(yvals);
      testxdata.push_back(mtempx);
      testydata.push_back(mtempy);
      numlines++;
    }
  int passed = 0;
  int error = 0;
  for (int i = 0; i < numlines; i++)
    {
      feed_forward(testxdata[i],feedforwardmode);
      if (classreg == 0)
	{
	  int max = 0;
	  int idx = 0;
	  int lent = numlayers[numhid + 1];
	  int alent = numhid + 1;
	  for (int j = 0; j < lent; j++)
	    {
	      if (j == 0)
		{
		  max = activ[0][alent](j,0);
		  idx = j;
		}
	      else
		{
		  if (activ[0][alent](j,0) > max)
		    {
		      max = activ[0][alent](j,0);
		      activ[0][alent](j,0)  = 0.0;
		      idx = j;
		    }
		  else
		    {
		      activ[0][alent](j,0) = 0.0;
		      continue;
		    }
		}
	    }
	  activ[0][idx](idx,0) = 1.0;
	  if (abs(activ[0][alent](idx,0) - testydata[i](idx,0)) <= 0.1)
	    {
	      passed++;
	    }
	}
      else
	{
	  int chk = 1;
	  int lent = activ[0][numhid + 1].n_rows;
	  for (int j = 0; j < lent; j++)
	    {
	      error = error + pow(testydata[i](j,0) - activ[0][numhid + 1](j,0),2);
	      if (abs(activ[0][numhid + 1](j,0) - testydata[i](j,0)) <= 0.1)
		{
		  continue;
		}
	      else
		{
		  chk = 0;
		  break;
		}
	    }
	  if (chk)
	    {
	      passed++;
	    }
	}
    }
  double hitrate = ((double)passed/(double)numlines)*100;
  if (verbose == 1)
    {
      cout<<"The accuracy is: "<<hitrate<<"%\n";
    }
  double RMSE = sqrt((error/(double)numlines));
  if(classreg == 1)
    {
      temp_rmse = RMSE;
      if (verbose == 1)
	{
	  cout<<"RMSE: "<<RMSE<<endl;
	}
    }
  else
    {
      temp_rmse = hitrate;
      if (verbose == 1)
	{
	  cout<<"Passed: "<<passed<<endl;
	}
    }
  return; 
}


/*
 *DEFINED BELOW ARE METHODS TO TEST AN EXPERIMENTAL ALGORITHM FOR LEARNING, ONLY 
 *FOR THE BRAVE AND COURAGEOUS
 *
 *TODO: CHECK ASSIGNMENT OF ALL NETWORK ARCHITECTURES
 */



void NNet::ls_init(string nconfig, int iclassreg, int igradd, int icostfunc, int iepoch)
{
  numfiles = 0;
  if (!l_numlayers.empty())
    {
      l_numlayers.clear();
    }
  if(!l_numhids.empty())
    {
      l_numhids.clear();
    }
  vector<string> nconfigs;
  int nlent = nconfig.length();
  for (int i = 0; i < nlent; i++)
    {
      string cmm = ",";
      string septr = "-";
      string num = "";
      if ((nconfig[i] != cmm[0]) && (nconfig.at(i) != septr[0]) && (isdigit(nconfig.at(i)) == 0))
	{
	  cout<<"Invalid configuration!"<<endl;
	  abort();
	}
      else if (nconfig[i] == cmm[0])
	{
	  nconfigs.push_back(num);
	  num = "";
	  numfiles++;
	}
      else if (i == (nlent -1))
	{
	  nconfigs.push_back(num);
	  num = "";
	  numfiles++;
	}
      else
	{
	  num = num + nconfig[i];
	}
    }
  for(int j = 0; j < numfiles; j++)
    {
      string sconfig = nconfigs[j];
      vector<int> itn;
      l_numlayers.push_back(itn);
      int lent = sconfig.length();
      string num = "";
      //parses sconfig
      for (int i = 0; i < lent; i++)
	{
	  if ((sconfig.at(i) != '-') && (isdigit(sconfig.at(i)) == 0))
	    {
	      cout<<"Invalid input!\n";
	      return; 
	    }
	  else if ((sconfig.at(i) != '-') && (isdigit(sconfig.at(i)) != 0) && (i != (lent -1)))
	    {
	      num = num + sconfig.at(i);
	    }
	  else if (sconfig.at(i) == '-')
	    {
	      //config = (int *)realloc(config,(count + 1)*sizeof(int));
	      //config[count] = stoi(num,NULL);
	      //count++;
	      l_numlayers[j].push_back(stoi(num,NULL));
	      num = "";
	    }
	  else if ((i == lent - 1) && (sconfig.at(i) != '-'))
	    {
	      num = num + sconfig.at(i);
	      //config = (int *)realloc(config,(count + 1)*sizeof(int));
	      //config[count] = stoi(num,NULL);
	      l_numlayers[j].push_back(stoi(num,NULL));
	    }
	}
      l_numhids.push_back(l_numlayers[j].size());
    }
  //sets variables
  classreg = iclassreg;
  gradd = igradd;
  costfunc = icostfunc;
  epoch = iepoch;
  //checks network confgiuration
  if ((classreg > 1) || (gradd > 1) || (costfunc > 1) || (classreg < 0) || (gradd < 0))
    {
      cout<<"Invalid network configuration!\n";
      return;
    }
  if (epoch <= 0)
    {
      cout<<"Invalid configuration\nPlease choose the number of epochs to be trained";
      return;
    }
  for (int j = 0; j < numfiles; j++)
    {
      vector<int> tr;
      l_funclayer.push_back(tr);
      int count = l_numhids[j];
      for (int i = 0; i < count + 1; i++)
	{
	  if (classreg == 0)
	    {
	      l_funclayer[j].push_back(0);
	    }
	  else if (classreg == 1)
	    {
	      l_funclayer[j].push_back(3);
	    }
	}
    }
  //numhid = count;
  vector<mat> tr;
  l_dels.push_back(tr);
  l_tdels.push_back(tr);
  for (int i = 0; i < numfiles; i++)
    {
      l_activ.push_back(tr);
      l_sums.push_back(tr);
      l_grads.push_back(tr);
      l_dels.push_back(tr);
      l_tgrads.push_back(tr);
      l_tdels.push_back(tr);
      l_checkgrads.push_back(tr);
      l_checkdels.push_back(tr);
    }
  checkinit = 0;
  return;
}



//A
void NNet::ls_load(string outputfiles, string Qmatrix, int lmode, string input_file, string sep1)
{
  if (checkinit == -1)
    {
      cout<<"Please initilize the Neural Network!\n";
      return;
    }
  l_numx = 0;
  if (!filenames.empty())
    {
      filenames.clear();
    }
  int nlent = outputfiles.length();
  int fcount = 0;
  for (int i = 0; i < nlent; i++)
    {
      string num = "";
      string cmma = ",";
      if(outputfiles[i] == cmma[0])
	{
	  filenames.push_back(num);
	  num = "";
	  fcount++;
	}
      else if (i == (nlent - 1))
	{
	  filenames.push_back(num);
	  num = "";
	  fcount++;
	}
      else
	{
	  num = num + outputfiles[i];
	}
    }
  if (fcount != numfiles)
    {
      cout<<"Incorrect number of files given!"<<endl;
      abort();
    }
  for (int i = 0; i < numfiles; i++)
    {
      vector<mat> t1;
      l_params.push_back(t1);
      l_bias.push_back(t1);
      l_yvals.push_back(t1);
    }
  string decp = ".";
  string minussb = "-";
  string empt = " ";
  //int templines = 0;
  int numlines = 0;
  for (int j = 0; j < numfiles; j++)
    {
      ifstream ldata(filenames.at(j));
      if (!ldata.is_open())
	{
	  cout<<"Error opening file!\n";
	  abort();
	  return;
	}
      string temp;
      //templines = numlines;
      numlines = 0;
      //parse file input
      int tempcount = 0;
      int county = 0;
      while (getline(ldata,temp))
	{
	  int lent = temp.length();
	  string num = "";
	  if((tempcount != county) && (tempcount != 0))
	    {
	      cout<<temp;
	      cout<<"Change in length of output!\n";
	      abort();
	      return;
	    }
	  tempcount = county;
	  county = 0;
	  vector<double> yvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  //cout<<temp.at(i)<<endl;
		  cout << "Invalid file format!\n";
		  return;
		}
	      if (((i < (lent-1))) && ((temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)) || (isdigit(temp.at(i)) != 0)))
		{
		  num = num + temp.at(i);
		}
	      else if (temp.at(i) == sep1.at(0))
	       {
		 yvals.push_back(stod(num,NULL));
		 num = "";
		 county++;
	       }
	      else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
		{
		  num = num + temp.at(i);
		  yvals.push_back(stod(num,NULL));
		  num = "";
		  county++;
		}
	    }
	  mat mtempy(yvals);
	  l_yvals[j].push_back(mtempy);
	  numlines++;
	}
      l_numlayers[j].push_back(county);
      ldata.close();
    }
  file_nlines = numlines;
  //cout<<numlines<<endl;
  
  //LOADING INPUT FILE
  if (input_file.at(0) != empt.at(0))
    {
      //LOADING QMATRIX
      int qnumlines = 0;
      qmat = 1;
      cout<<"QMAT: "<<qmat<<endl;
      ifstream qdata(Qmatrix);
      if (!qdata.is_open())
	{
	  cout<<"Error opening file!\n";
	  abort();
	  return;
	}
      string temp;
      int qtempcount = 0;
      int qcounty = 0;
      string qsep = ",";
      while (getline(qdata,temp))
	{
	  int lent = temp.length();
	  string num = "";
	  if((qtempcount != qcounty) && (qtempcount != 0))
	    {
	      cout<<temp;
	      cout<<"Change in length of output!\n";
	      abort();
	      return;
	    }
	  qtempcount = qcounty;
	  qcounty = 0;
	  vector<int> qvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != qsep.at(0)))
		{
	      cout<<temp.at(i)<<endl;
	      cout << "Invalid file format!\n";
	      return;
		}
	      if (((i < (lent-1))) && ((temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)) || (isdigit(temp.at(i)) != 0)))
		{
		  num = num + temp.at(i);
		}
	      else if (temp.at(i) == qsep.at(0))
		{
		  qvals.push_back(stoi(num,NULL));
		  num = "";
		  qcounty++;
		}
	      else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
		{
		  num = num + temp.at(i);
		  qvals.push_back(stoi(num,NULL));
		  num = "";
		  qcounty++;
		}
	    }
	  Q_mat.push_back(qvals);
	  qnumlines++;
	}
      qdata.close();
      ifstream ldata(input_file);
      if (!ldata.is_open())
	{
	  cout<<"Error opening file!\n";
	  return;
	}
      //string temp;
      int xnumlines = 0;
      int tempcx = 0;
      int countx = 0;
      //parse file input
      while (getline(ldata,temp))
	{
	  int lent = temp.length();
	  string num = "";
	  if ((tempcx != countx) && (tempcx != 0))
	    {
	      cout<<"Change in length of input!"<<endl;
	      abort();
	      return;
	    }
	  tempcx = countx;
	  countx = 0;
	  vector<double> xvals;
	  for(int i = 0; i < lent; i++)
	    {
	      //cout<<sep1.at(0);
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  //cout<<temp.at(i)<<endl;
		  cout << "Invalid file format!\n";
		  abort();
		  return;
		}
	      if (((i < (lent-1))) && ((temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)) || (isdigit(temp.at(i)) != 0)))
		{
		  num = num + temp.at(i);
		}
	      else if (temp.at(i) == sep1.at(0))
		{
		  xvals.push_back(stod(num,NULL));
		  num = "";
		  countx++;
		}
	      else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
		{
		  num = num + temp.at(i);
		  xvals.push_back(stod(num,NULL));
		  num = "";
		  countx++;
		}
	    }
	  //cout<<xvals.size();
	  mat mtempx(xvals);
	  l_xvals.push_back(mtempx);
	  xnumlines++;
	}
      if (xnumlines != qnumlines)
	{
	  cout<<"Missing Q-matrix entries for certain data points!"<<endl;
	  abort();
	  return;
	}
      l_numx = countx;
    }
  if (lmode == 0)
    {
      l_test = numlines/5;
      l_validate = tests;
      l_train = numlines - tests - validate;
    }
  else if(lmode == 1)
    {
      l_train = numlines;
    }
  else
    {
      cout<<"Loading configuration can only be 1 or 0"<<endl;
    }
  return;
}



//The below method initilizes the neural network
void NNet::l_init(int num_files, int iclassreg, int inumcores, int igradd, int icostfunc, int iepoch)
{
  if (num_files <= 0)
    {
      cout<<"Number of files must be non-zero and positive!"<<endl;
      return;
    }
  numfiles = num_files;
  if (!l_numlayers.empty())
    {
      l_numlayers.clear();
    }
  if(!l_numhids.empty())
    {
      l_numhids.clear();
    }
  for(int j = 0; j < numfiles; j++)
    {
      string sconfig;
      vector<int> itn;
      l_numlayers.push_back(itn);
      cout<<"Please enter the configuration of NN "<<to_string(j + 1)<<": ";
      //int count = 0;
      cin>>sconfig;
      int lent = sconfig.length();
      string num = "";
      //parses sconfig
      for (int i = 0; i < lent; i++)
	{
	  if ((sconfig.at(i) != '-') && (isdigit(sconfig.at(i)) == 0))
	    {
	      cout<<"Invalid input!\n";
	      return; 
	    }
	  else if ((sconfig.at(i) != '-') && (isdigit(sconfig.at(i)) != 0) && (i != (lent -1)))
	    {
	      num = num + sconfig.at(i);
	    }
	  else if (sconfig.at(i) == '-')
	    {
	      //config = (int *)realloc(config,(count + 1)*sizeof(int));
	      //config[count] = stoi(num,NULL);
	      //count++;
	      l_numlayers[j].push_back(stoi(num,NULL));
	      num = "";
	    }
	  else if ((i == lent - 1) && (sconfig.at(i) != '-'))
	    {
	      num = num + sconfig.at(i);
	      //config = (int *)realloc(config,(count + 1)*sizeof(int));
	      //config[count] = stoi(num,NULL);
	      l_numlayers[j].push_back(stoi(num,NULL));
	    }
	}
      l_numhids.push_back(l_numlayers[j].size());
    }
  //sets variables
  classreg = iclassreg;
  numcores = inumcores;
  gradd = igradd;
  costfunc = icostfunc;
  epoch = iepoch;
  //checks network confgiuration
  if ((classreg > 1) || (gradd > 1) || (costfunc > 1) || (classreg < 0) || (gradd < 0))
    {
      cout<<"Invalid network configuration!\n";
      return;
    }
  if (epoch <= 0)
    {
      cout<<"Invalid configuration\nPlease choose the number of epochs to be trained";
      return;
    }
  for (int j = 0; j < numfiles; j++)
    {
      vector<int> tr;
      l_funclayer.push_back(tr);
      int count = l_numhids[j];
      for (int i = 0; i < count + 1; i++)
	{
	  if (classreg == 0)
	    {
	      l_funclayer[j].push_back(0);
	    }
	  else if (classreg == 1)
	    {
	      l_funclayer[j].push_back(3);
	    }
	}
    }
  //numhid = count;
  vector<mat> tr;
  l_dels.push_back(tr);
  l_tdels.push_back(tr);
  for (int i = 0; i < numfiles; i++)
    {
      l_activ.push_back(tr);
      l_sums.push_back(tr);
      l_grads.push_back(tr);
      l_dels.push_back(tr);
      l_tgrads.push_back(tr);
      l_tdels.push_back(tr);
      l_checkgrads.push_back(tr);
      l_checkdels.push_back(tr);
    }
  checkinit = 0;
  return;
}


//This is a special load method for latent parameters
void NNet::l_load(string Qmatrix, int lmode, string input_file, string sep1)
{
  if (checkinit == -1)
    {
      cout<<"Please initilize the Neural Network!\n";
      return;
    }
  l_numx = 0;
  string temp1;
  if (!filenames.empty())
    {
      filenames.clear();
    }
  cout<<"Please enter the names of the files."<<endl;
  //LOADING OUTPUT FILES
  for (int i = 0; i < numfiles; i++)
    {
      cout<<"File "<<to_string(i + 1)<<": ";
      cin>>temp1;
      vector<mat> t1;
      filenames.push_back(temp1);
      l_params.push_back(t1);
      l_bias.push_back(t1);
      l_yvals.push_back(t1);
    }
  string decp = ".";
  string minussb = "-";
  string empt = " ";
  //int templines = 0;
  int numlines = 0;
  for (int j = 0; j < numfiles; j++)
    {
      ifstream ldata(filenames.at(j));
      if (!ldata.is_open())
	{
	  cout<<"Error opening file!\n";
	  abort();
	  return;
	}
      string temp;
      //templines = numlines;
      numlines = 0;
      //parse file input
      int tempcount = 0;
      int county = 0;
      while (getline(ldata,temp))
	{
	  int lent = temp.length();
	  string num = "";
	  if((tempcount != county) && (tempcount != 0))
	    {
	      cout<<temp;
	      cout<<"Change in length of output!\n";
	      abort();
	      return;
	    }
	  tempcount = county;
	  county = 0;
	  vector<double> yvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  //cout<<temp.at(i)<<endl;
		  cout << "Invalid file format!\n";
		  return;
		}
	      if (((i < (lent-1))) && ((temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)) || (isdigit(temp.at(i)) != 0)))
		{
		  num = num + temp.at(i);
		}
	      else if (temp.at(i) == sep1.at(0))
	       {
		 yvals.push_back(stod(num,NULL));
		 num = "";
		 county++;
	       }
	      else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
		{
		  num = num + temp.at(i);
		  yvals.push_back(stod(num,NULL));
		  num = "";
		  county++;
		}
	    }
	  mat mtempy(yvals);
	  l_yvals[j].push_back(mtempy);
	  numlines++;
	}
      l_numlayers[j].push_back(county);
      ldata.close();
    }
  file_nlines = numlines;
  //cout<<numlines<<endl;
  
  //LOADING INPUT FILE
  if (input_file.at(0) != empt.at(0))
    {
      //LOADING QMATRIX
      int qnumlines = 0;
      qmat = 1;
      cout<<"QMAT: "<<qmat<<endl;
      ifstream qdata(Qmatrix);
      if (!qdata.is_open())
	{
	  cout<<"Error opening file!\n";
	  abort();
	  return;
	}
      string temp;
      int qtempcount = 0;
      int qcounty = 0;
      string qsep = ",";
      while (getline(qdata,temp))
	{
	  int lent = temp.length();
	  string num = "";
	  if((qtempcount != qcounty) && (qtempcount != 0))
	    {
	      cout<<temp;
	      cout<<"Change in length of output!\n";
	      abort();
	      return;
	    }
	  qtempcount = qcounty;
	  qcounty = 0;
	  vector<int> qvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != qsep.at(0)))
		{
		  cout<<temp.at(i)<<endl;
		  cout << "Invalid file format!\n";
		  return;
		}
	      if (((i < (lent-1))) && ((temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)) || (isdigit(temp.at(i)) != 0)))
		{
		  num = num + temp.at(i);
		}
	      else if (temp.at(i) == qsep.at(0))
		{
		  qvals.push_back(stoi(num,NULL));
		  num = "";
		  qcounty++;
		}
	      else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
		{
		  num = num + temp.at(i);
		  qvals.push_back(stoi(num,NULL));
		  num = "";
		  qcounty++;
		}
	    }
	  Q_mat.push_back(qvals);
	  qnumlines++;
	}
      qdata.close();
      ifstream ldata(input_file);
      if (!ldata.is_open())
	{
	  cout<<"Error opening file!\n";
	  return;
	}
      //string temp;
      int xnumlines = 0;
      int tempcx = 0;
      int countx = 0;
      //parse file input
      while (getline(ldata,temp))
	{
	  int lent = temp.length();
	  string num = "";
	  if ((tempcx != countx) && (tempcx != 0))
	    {
	      cout<<"Change in length of input!"<<endl;
	      abort();
	      return;
	    }
	  tempcx = countx;
	  countx = 0;
	  vector<double> xvals;
	  for(int i = 0; i < lent; i++)
	    {
	      //cout<<sep1.at(0);
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  //cout<<temp.at(i)<<endl;
		  cout << "Invalid file format!\n";
		  abort();
		  return;
		}
	      if (((i < (lent-1))) && ((temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)) || (isdigit(temp.at(i)) != 0)))
		{
		  num = num + temp.at(i);
		}
	      else if (temp.at(i) == sep1.at(0))
		{
		  xvals.push_back(stod(num,NULL));
		  num = "";
		  countx++;
		}
	      else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
		{
		  num = num + temp.at(i);
		  xvals.push_back(stod(num,NULL));
		  num = "";
		  countx++;
		}
	    }
	  //cout<<xvals.size();
	  mat mtempx(xvals);
	  l_xvals.push_back(mtempx);
	  xnumlines++;
	}
      if (xnumlines != qnumlines)
	{
	  cout<<"Missing Q-matrix entries for certain data points!"<<endl;
	  abort();
	  return;
	}
      l_numx = countx;
    }
  if (lmode == 0)
    {
      l_test = numlines/5;
      l_validate = tests;
      l_train = numlines - tests - validate;
    }
  else if(lmode == 1)
    {
      l_train = numlines;
    }
  else
    {
      cout<<"Loading configuration can only be 1 or 0"<<endl;
    }
  return;
}


//
void NNet::l_feedforward(mat x,int gpos)
{
  int idx;
  //TODO: ASPA decide if l_best_params is needed or not
  //if (gpos == -2)
  //{
  //  idx = 0;
  //  if (!activ[idx].empty())
  //	activ[idx].clear();
  //  if (!sums.empty())
  //	sums[idx].clear();
  //  activ[idx].push_back(x);
  //  sums[idx].push_back(x);
  //  int count = 1;
  //  for (int i = 0; i < numhid+1; i++)
  //	{
  //	  activ[idx].push_back(best_params.at(i)*activ[idx].at(count-1) + best_bias.at(i));
  //	  sums[idx].push_back(activ[idx].at(count));
  //	  int lent = numlayers[count];
  //	  for (int j = 0; j < lent; j++)
  //	    {
  //	      if (funclayer[i] == 0)
  //		{
  //		  activ[idx].at(count)(j,0) = sigmoid(activ[idx].at(count)(j,0));
  //		}
  //	      else if (funclayer[i] == 1)
  //		{
  //		  activ[idx].at(count)(j,0) = tanh(activ[idx].at(count)(j,0));
  //		}
  //	      else if (funclayer.at(i) == 2)
  //		{
  //		  activ[idx].at(count)(j,0) = reclinear(activ[idx].at(count)(j,0));
  //		}
  //	      else
  //		{
  		  //cout<<activ.size();
  //		  activ[idx].at(count)(j,0) = tanh_r(activ[idx].at(count)(j,0));
		  //cout<<activ.at(count)(0,0)<<" After\n";
  //		}
  //	    }
  //	  count++;
  //	}
  //  return;
  //}
  if (gpos == -1)
    {
      idx = 0;
    }
  else
    {
      idx = gpos;
    }
  if (!l_activ[idx].empty())
    l_activ[idx].clear();
  if (!l_sums.empty())
    l_sums[idx].clear();
  l_activ[idx].push_back(x);
  l_sums[idx].push_back(x);
  int count = 1;
  int lnumhid = l_numhids[idx];
  for (int i = 0; i < lnumhid+1; i++)
    {
      l_activ[idx].push_back(l_params[idx].at(i)*l_activ[idx].at(count-1) + l_bias[idx].at(i));
      l_sums[idx].push_back(l_activ[idx].at(count));
      int lent = l_numlayers[idx][count];
      for (int j = 0; j < lent; j++)
	{
	  if (l_funclayer[idx][i] == 0)
	    {
	      l_activ[idx].at(count)(j,0) = sigmoid(l_activ[idx].at(count)(j,0));
	    }
	  else if (l_funclayer[idx][i] == 1)
	    {
	      l_activ[idx].at(count)(j,0) = tanh(l_activ[idx].at(count)(j,0));
	    }
	  else if (l_funclayer[idx].at(i) == 2)
	    {
	      l_activ[idx].at(count)(j,0) = reclinear(l_activ[idx].at(count)(j,0));
	    }
	  else
	    {
	      //cout<<activ.size();
	      l_activ[idx].at(count)(j,0) = tanh_r(l_activ[idx].at(count)(j,0));
	      //cout<<activ.at(count)(0,0)<<" After\n";
	    }
	}
      count++;
    }
  //cout<<"FACT: "<<l_activ[0][2](0,0)<<endl;
  return;
}


//This is a slightly modified variant of backprop so learning latent parameters is also incorporated
void NNet::l_backprop(mat x, mat y, int gpos)
{
  //cout<<"backproping"<<endl;
  int idx;
  if (gpos == -1)
    {
      idx = 0;
    }
  else
    {
      idx = gpos;
    }
  //paracheck[0] = 21;
  int l_numhid = l_numhids.at(idx);
  l_feedforward(x,idx);
  if(!l_grads[idx].empty())
    { 
      l_grads[idx].clear();
    }
  if(!l_dels[idx].empty())
    {
      l_dels[idx].clear();
    }
  if (costfunc == 0)
    {
      if (l_funclayer[idx][l_numhid] == 0)
	{
	  l_dels[idx].push_back((l_activ.at(idx).at(l_numhid+1) - y)%(l_activ[idx][l_numhid+1] - (l_activ.at(idx).at(l_numhid+1)%l_activ.at(idx).at(l_numhid+1))));
	}
      else if (l_funclayer[idx][l_numhid] == 1)
	{
	  l_dels[idx].push_back((y - l_activ[idx][l_numhid+1])%(ones<mat>(numlayers[l_numhid+1],1) - l_activ[idx][l_numhid+1]%l_activ[idx][l_numhid+1]));
	}
      else if (l_funclayer[idx][l_numhid] == 2)
	{
	  for (int i = 0; i < l_numlayers[idx][l_numhid+1]; i++)
	    {
	      l_sums[idx][l_numhid + 1](i,0) = rec_D(l_sums[idx][l_numhid + 1](i,0));
	    }
	  l_dels[idx].push_back((l_activ[idx][l_numhid+1]-y)%l_sums[idx][l_numhid+1]);
	}
      else if (l_funclayer[idx][l_numhid] == 3)
	{
	  for (int i = 0; i < l_numlayers[idx][l_numhid+1]; i++)
	    {
	      l_sums[idx][l_numhid + 1](i,0) = tanh_d(l_sums[idx][l_numhid + 1](i,0));
	    }
	  l_dels[idx].push_back((l_activ[idx][l_numhid+1]-y)%l_sums[idx][l_numhid+1]);
	}
    }
  else
    {
      //TODO: have to complete results for other cost functions
      return;
    }
  int count = 1;
  for (int i = 0; i < l_numhid + 1; i++)
    {
      mat temp;
      temp = (l_params[idx][l_numhid - i].t())*(l_dels[idx][i]);
      mat derv = l_sums[idx][l_numhid - i];
      if (i < l_numhid)
	{
	  if (l_funclayer[idx][l_numhid - count] == 0)
	    { 
	      derv = activ[idx][l_numhid- i] - (l_activ[idx][l_numhid - i]%l_activ[idx][l_numhid - i]);
	    }
	  else if (l_funclayer[idx][l_numhid-count] == 1)
	    {
	      int n = l_numlayers[idx][l_numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  derv(j,0) = tanh_dr(derv(j,0));
		}
	    }
	  else if (l_funclayer[idx][l_numhid - count] == 2)
	    {
	      int n = l_numlayers[idx][l_numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  derv(j,0) = rec_D(derv(j,0));
		}
	    }
	  else if (l_funclayer[idx][l_numhid-count] == 3)
	    {
	      int n = l_numlayers[idx][l_numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  derv(j,0) = tanh_d(derv(j,0));
		}
	    }
	}
      else
	{
	  derv = ones<mat>(l_numlayers[idx][0],1);
	}
      temp = temp%derv;
      l_dels[idx].push_back(temp);
      count++;
    }
  for (int i = 0; i < l_numhid + 1; i++)
    {
      l_grads[idx].push_back(l_dels[idx][i]*(l_activ[idx][l_numhid - i].t()));
    }
  return;
}


//Training weights and latent parameters
void NNet::l_trainnet(int numlatent, int mode)
{
  int trainmode = mode;
  vector<thread> l_bpthreads;
  if ((trainmode != 0) && (trainmode != 1))
    {
      cout<<"Training mode can only be 0 or 1"<<endl;
      return;
    }
  if ((numlatent >= 0) && (l_trained == 0))
    {
      if (l_xvals.empty())
	{
	  for (int i = 0; i < file_nlines; i++)
	    {
	      l_xvals.push_back(randn<mat>(numlatent,1));
	    }
	}
      else
	{
	  for (int i = 0; i < file_nlines; i++)
	    {
	      for (int j = 0; j < numlatent; j++)
		{
		  mat trnum = randn<mat>(1,1);
		  (l_xvals.at(i)).insert_rows(0,trnum);
		}
	    }
	}
      l_numx = l_numx + numlatent;
      for(int i = 0; i < numfiles; i++)
	{
	  l_numlayers[i].insert(l_numlayers[i].begin(),l_numx);
	  l_tdels[i].push_back(zeros<mat>(l_numx,1));
	  for(int j = 0; j < l_numhids[i] + 1; j++)
	    {
	      int rows = l_numlayers[i][j+1];
	      int cols = l_numlayers[i][j];
	      l_params[i].push_back(randn<mat>(rows,cols));
	      l_tgrads[i].push_back(zeros<mat>(rows,cols));
	      l_bias[i].push_back(randn<mat>(rows,1));
	      l_tdels[i].push_back(zeros<mat>(rows,1));
	    }
	}
    }
  else
    {
      if (numlatent < 0)
	{
	  cout<<"Number of latent parameters must be greater than 1"<<endl;
	  return;
	}
      else
	{
	  for(int i = 0; i < numfiles; i++)
	    {
	      l_tdels[i][0].fill(0);
	      for (int j = 0; j < l_numhids[i] + 1; j++)
		{
		  l_tgrads[i][j].fill(0);
		  l_tdels[i][j+1].fill(0);
		}
	    }
	}
    }
  //double ttemp_rmse;
  //int ecount = 0;
  vector<double> lrates;
  for(int i = 0; i < numfiles; i++)
    {
      double rate;
      cout<<"Please enter the desired learning rate for NN "<<to_string(i+1)<<": ";
      cin>>rate;
      lrates.push_back(rate);
    }
  if (gradd == 0)
    {
      //int stop = 0;
      //double beta = 0.2;
      for (int k = 0; k < epoch; k++)
	{
	  cout<<(double)k*100/(double)epoch<<"%\n";
	  for (int i = 0; i < train; i++)
	    {
	      //cout<<train;
	      //if (classreg == 0)
	      //{
	      //  for (int j = 0; j < numfiles; j++)
	      //    {
	      //      for (int l = 0; l < numhid + 1; l++)
	      //	{
	      //	  l_params[j].at(l) = l_params[j].at(l) + beta*l_velocity[j].at(l);
	      //	}
	      //    }
	      //}
	      //backprop(xdata[i],ydata[i],-1);
	      int threadcount = 0;
	      for (int j = 0; j < numfiles; j++)
		{
		  if (qmat == 1)
		    {
		      if (Q_mat[i][j] == 1)
			{
			  l_bpthreads.push_back(std::thread(&NNet::l_parallelbp,this,i,j));
		      threadcount++;
			}
		    }
		  else
		    {
		    l_bpthreads.push_back(std::thread(&NNet::l_parallelbp,this,i,j));
		      threadcount++;
		    }
		}
	      for(int j = 0; j < threadcount; j++)
		{
		  l_bpthreads[j].join();
		}
	      for (int q = 0; q < numfiles; q++)
		{
		  if (qmat == 1)
		    {
		      if (Q_mat[i][q] == 1)
			{
			  int lnumhid = l_numhids[q];
			  l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
			  for (int j = 0; j < lnumhid + 1; j++)
			    {
			      l_tgrads[q][j] = tgrads[q][j] + grads[q][lnumhid-j];
			      l_tdels[q][j+1] = l_tdels[q][j+1] + l_dels[q][lnumhid - j];
			    }
			}
		    }
		  else
		    {
		      int lnumhid = l_numhids[q];
		      l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
		      for (int j = 0; j < lnumhid + 1; j++)
			{
			  l_tgrads[q][j] = tgrads[q][j] + grads[q][lnumhid-j];
			  l_tdels[q][j+1] = l_tdels[q][j+1] + l_dels[q][lnumhid - j];
			}
		    }
		}
	      for(int q = 0; q < numfiles; q++)
		{
		  if( qmat == 1)
		    {
		      if (Q_mat[i][q] == 1)
			{
			  for(int j = 0; j < numlatent; j++)
			    {
			      l_xvals[i](j,0) = l_xvals[i](j,0) - (lrates[q]/4.0)*l_tdels[q][0](j,0);
			      l_tdels[q][0].fill(0); //Doing this is the textbook method but commenting this out just works much much better
			    }
			}
		    }
		  else
		    {
		      for(int j = 0; j < numlatent; j++)
			{
			  l_xvals[i](j,0) = l_xvals[i](j,0) - (lrates[j]/4.0)*l_tdels[q][0](j,0);
			  l_tdels[q][0].fill(0); //Doing this is the textbook method but commenting this out just works much much better
			}
		    }
		}
	    }
	  //if (k > 500.0/2.0)
	  //{
	  //  beta = 0.9;
	  //}
	  for(int t = 0; t < numfiles; t++)
	    {
	      int lnumhid = l_numhids[t];
	      for (int l = 0; l < lnumhid + 1; l++)
		{
		  //cout<<velocity.at(i)<<"\n";
		  //velocity.at(l) = beta*velocity.at(l) - (lrate/(double)train)*tgrads.at(l);
		  l_params[t][l] = params[t][l] - (lrates[t]/(double)train)*tgrads[t][l] - 0.00001*params[t][l];
		  //params.at(l) = params.at(l) + velocity.at(l);
		  l_bias[t][l] = l_bias[t][l] - (lrates[t]/(double)train)*l_tdels[t][l];
		}
	    }
	}
    }
  //TODO:MODIFY SGD FOR LATENT PARAMS
  else if (gradd == 1)
    {
      cout<<"Initializing Stochastic Gradient Descent L\n";
      //min_rmse = -1;
      vector<int> idxs;
      for (int i = 0; i < l_train; i++)
	{
	  idxs.push_back(i);
	}
      random_shuffle(idxs.begin(),idxs.end());
      //double beta;
      //if (classreg == 0)
      //{
      //  beta = 0.2;
      //}
      for (int i = 0; i < epoch; i++)
	{
	  cout<<((double)i/(double)epoch)*100<<"%\n";
	  if (trainmode == 1)
	    {
	      l_testall();
	      cout<<endl;
	    }
	  int step = 0;
	  for (int lr = 0; lr < numfiles; lr++)
	    {
	      lrates[lr] = 0.99*lrates[lr];
	    }
	  while (step < l_train)
	    {
	      int k = step;
	      step = min(step + 10,l_train);
	      for(;k < step; k++)
		{
		  if (!l_bpthreads.empty())
		    {
		      l_bpthreads.clear();
		    }
		  int threadcount = 0;
		  for (int t = 0; t < numfiles; t++)
		    {
		      //cout<<"Q-val: "<<Q_mat[idxs.at(k)][t]<<endl;
		      if(qmat == 1)
			{
			  if (Q_mat[idxs.at(k)][t] == 1)
			    {
			      //backprop(xdata.at(idxs.at(k+t)),ydata.at(idxs.at(k+t)),-1);
			      l_bpthreads.push_back(thread(&NNet::l_parallelbp,this,idxs.at(k),t));
			      threadcount++;
			    }
			}
		      else
			{
			  l_bpthreads.push_back(thread(&NNet::l_parallelbp,this,idxs.at(k),t));
			  threadcount++;
			}
		    }
		  for(int t = 0; t < threadcount; t++)
		    {
		      l_bpthreads[t].join();
		    }
		  for (int q = 0; q < numfiles; q++)
		    {
		      if (qmat == 1)
			{
			  if (Q_mat[idxs.at(k)][q] == 1)
			    {
			      int lnumhid = l_numhids[q];
			      l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
			      for(int j = 0; j < lnumhid + 1; j++)
				{
				  l_tgrads[q].at(j) = l_tgrads[q].at(j) + l_grads[q].at(lnumhid - j);
				  l_tdels[q].at(j+1) = l_tdels[q].at(j+1) + l_dels[q].at(lnumhid -j);
				}
			    }
			}
		      else
			{
			  int lnumhid = l_numhids[q];
			  l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
			  for(int j = 0; j < lnumhid + 1; j++)
			    {
			      l_tgrads[q].at(j) = l_tgrads[q].at(j) + l_grads[q].at(lnumhid - j);
			      l_tdels[q].at(j+1) = l_tdels[q].at(j+1) + l_dels[q].at(lnumhid -j);
			    }
			}
		    }
		  for(int q = 0; q < numfiles; q++)
		    {
		      if(qmat == 1)
			{
			  if (Q_mat[idxs.at(k)][q] == 1)
			    {
			      for(int j = 0; j < numlatent; j++)
				{
				  l_xvals[idxs.at(k)](j,0) = l_xvals[idxs.at(k)](j,0) - (lrates[q]/(double)numfiles)*l_tdels[q][0](j,0);
				  l_tdels[q][0].fill(0); //Doing this is the textbook method but commenting this out just works much much better
				}
			    }
			}
		      else
			{
			  for(int j = 0; j < numlatent; j++)
			    {
			      l_xvals[idxs.at(k)](j,0) = l_xvals[idxs.at(k)](j,0) - (lrates[q]/(double)numfiles)*l_tdels[q][0](j,0);
			      l_tdels[q][0].fill(0); //Doing this is the textbook method but commenting this out just works much much better
			    }
			}
		    }
		}
	      //if (classreg == 0)
	      //{
	      //  for (int yt = 0; yt < numhid + 1; yt++)
	      //    {
	      //      l_params.at(yt) = l_params.at(yt) - beta*l_velocity.at(yt);
	      //      l_velocity.at(yt) = beta*l_velocity.at(yt) - (lrate/(double)1000.0)*tgrads.at(yt);
	      //    }
	      //}
	      double kappa = 0.001;
	      for(int q = 0; q < numfiles; q++)
		{
		  int lnumhid = l_numhids[q];
		  for (int j = 0; j < lnumhid + 1; j++)
		    {
		      //the below only applies if regression is going.....
		      if ((classreg == 1) || (classreg == 0))
			{
			  if (l_funclayer[q].at(j) == 3)
			    {
			      kappa = 0.2;
			    }
			  else
			    {
			      kappa = 0.001;
			    }
			  l_params[q].at(j) = l_params[q].at(j) - (lrates[q]/(double)100.0)*l_tgrads[q].at(j) - kappa*l_params[q].at(j);
			  //l_tgrads[q].at(j).fill(0); //Doing this is the textbook method but commenting this out just works much much better
			}
		  //else
		  //{
		  //  l_params.at(j) = l_params.at(j) + l_velocity.at(j) - kappa*l_params.at(j);
		  //}
		      l_bias[q].at(j) = l_bias[q].at(j) - (lrates[q]/(double)100.0)*l_tdels[q].at(j+1);
		      //l_tdels[q].at(j+1).fill(0);    //Doing this is the textbook method but commenting this out just works much much better
		    }
		}
	    }
	}
    }
  l_trained++;
  return;
}


//saves a particular neural network
void NNet::lsavenets(string netname, int index)
{
  fstream savednets;
  int ckopen = 0;
  savednets.open(".savednets",fstream::in);
  if (!savednets.is_open())
    {
      ckopen = 0;
    }
  else
    {
      ckopen = 1;
      savednets.close();
    }
  if (ckopen == 0)
    {
      savednets.open(".savednets",fstream::out);
      savednets.close();
    }
  savednets.open(".savednets",fstream::in);
  if (!savednets.is_open())
    {
      cout<<"Failed to open file 2\n";
      return;
    }
  string temp;
  string yes = "y";
  string no = "no";
  string stp = "*";
  int check = 0;
  vector<string> names;
  while(getline(savednets,temp))
    {
      int lent1 = temp.length();
      int lent2 = netname.length();
      for (int i = 0; (temp.at(i) != stp.at(0)) && (i < lent1) && (i < lent2); i++)
	{
	  if (temp.at(i) == netname.at(i))
	    {
	      check = 1;
	    }
	  else
	    {
	      check = 0;
	    }
	}
      if (check == 1)
	{
	  string ans;
	  int count = 0;
	  do
	    {
	      if (count == 0)
		{
		  cout<<"A neural network with the same name already exists, saving will overwrite the previous save. Do you wish to continue (y or n) ?: ";
		  cin>>ans;
		}
	      else
		{
		  cout<<"Continue with overwrite ? (y or n): ";
		  cin>>ans;
		}
	      count++;
	    }
	  while(((ans.at(0) != yes.at(0)) || ans.at(0) == no.at(0)) && (ans.length() != 1));
	  if (ans.at(0) == yes.at(0))
	    {
	      break;
	    }
	  else
	    {
	      cout<<"Save aborted!\n";
	      return;
	    }
	}
      else
	{
	   names.push_back(temp);
	}
    }
  savednets.close();
  savednets.open(".savednets",fstream::out);
  int l_numhid = l_numhids[index];
  names.push_back(netname + "*" + to_string(l_numhid + 1));
  int lent = names.size();
  for (int i = 0; i < lent; i++)
    {
      savednets<<names.at(i)<<endl;
    }
  netname = "." + netname + "_";
  for (int i = 0; i < l_numhid + 1; i++)
    {
      l_params[index].at(i).save(netname + "p" + to_string(i));
      l_bias[index].at(i).save(netname + "b" + to_string(i));
    }
  cout<<"Saved\n";
  savednets.close();
  return;
}


//saves network params and latent params
void NNet::l_savenet(void)
{
  vector<string> netnames;
  cout<<"You have loaded "<<to_string(numfiles)<<" files into the program."<<endl;
  for(int i = 0; i < numfiles; i++)
    {
      string temp;
      cout<<"Please enter the name of the NN associated with file "<<filenames.at(i)<<" :";
      cin>>temp;
      netnames.push_back(temp);
      lsavenets(netnames.at(i),i);
    }
  string input_name;
  string ans = "y";
  cout<<"Please enter the name of the input file: ";
  cin>>input_name;
  fstream saveinput;
  string yes = "y";
  string no = "n";
  saveinput.open(input_name,fstream::in);
  int chkopen = 0;
  if (saveinput.is_open())
    {
      chkopen = 1;
    }
  else
    {
      chkopen = 0;
    }
  saveinput.close();
  if (chkopen == 1)
    {
      do
	{
	  cout<<input_name<<" already exists, overwrite ? (y or n) : ";
	  cin>>ans;
	}
      while(((ans.at(0) != yes.at(0)) || (ans.at(0) != no.at(0))) && (ans.length() != 1));
    }
  if(ans.at(0) != yes.at(0))
    {
      cout<<"Save aborted!\n";
      return;
    }
  else
    {
      saveinput.open(input_name,fstream::out);
      for(int i = 0; i < file_nlines; i++)
	{
	  for(int j = 0; j < l_numx; j++)
	    {
	      saveinput<<l_xvals[i](j,0);
	      if (j < l_numx - 1)
		{
		  saveinput<<",";
		}
	    }
	  saveinput<<endl;
	}
    }
  saveinput.close();
  return;
}


//This methods loads a file to be tested against another file with the given parameters
void NNet::test_data(string in_filename, string out_filename, string netname, string sep)
{
  ifstream ldata(in_filename);
  if (!ldata.is_open())
    {
      cout<<"Error opening file!\n";
      return;
    }
  string temp;
  int xnumlines = 0;
  string decp = ".";
  string minussb = "-";
  int countx = 0;
  int xtemp = 0;
  int county = 0;
  int ytemp = 0;
  //parse file input
  while (getline(ldata,temp))
    {
      int lent = temp.length();
      string num = "";
      if ((xtemp != countx) && (xtemp != 0))
	{
	  cout<<"Invalid file format!"<<endl;
	  return;
	}
      xtemp = countx;
      countx = 0;
      vector<double> xvals;
      for(int i = 0; i < lent; i++)
	{
	  if  ((temp.at(i) != sep.at(0)) && (isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)))
	    {
	      cout<<temp.at(i)<<endl;
	      cout << "Invalid file format!\n";
	      return;
	    }
	  if (((temp.at(i) != sep.at(0)) && (i < (lent-1))) || (temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)))
	    {
	      num = num + temp.at(i);
	    }
	  else if (temp.at(i) == sep.at(0))
	    {
	      xvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	  else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
	    {
	      num = num + temp.at(i);
	      xvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	}
      mat mtempx(xvals);
      testxdata.push_back(mtempx);
      xnumlines++;
    }
  ldata.close();
  ldata.open(out_filename);
  int ynumlines = 0;
  while (getline(ldata,temp))
    {
      int lent = temp.length();
      //int track = 0;
      string num = "";
      if (ytemp != county)
	{
	  cout<<"Invalid file format!"<<endl;
	  return;
	}
      xtemp = countx;
      countx = 0;
      vector<double> yvals;
      for(int i = 0; i < lent; i++)
	{
	  if  ((temp.at(i) != sep.at(0)) && (isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)))
	    {
	      cout<<temp.at(i)<<endl;
	      cout << "Invalid file format!\n";
	      return;
	    }
	  if (((temp.at(i) != sep.at(0)) && (i < (lent-1))) || (temp.at(i) == decp.at(0)) || (temp.at(i) == minussb.at(0)))
	    {
	      num = num + temp.at(i);
	    }
	  else if (temp.at(i) == sep.at(0))
	    {
	      yvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	  else if ((i == (lent - 1)) && (isdigit(temp.at(i)) != 0))
	    {
	      num = num + temp.at(i);
	      yvals.push_back(stod(num,NULL));
	      num = "";
	      countx++;
	    }
	}
      //mat mtempx(xvals);
      mat mtempy(yvals);
      //xdata.push_back(mtempx);
      testydata.push_back(mtempy);
      ynumlines++;
    }
  ldata.close();
  if (xnumlines != ynumlines)
    {
      cout<<"The amount of input data is different from the amount of output data, line numbers in both files do not match!"<<endl;
      return;
    }
  double error = 0;
  //int passes = 0;
  for(int i = 0; i < xnumlines; i++)
    {
      feed_forward(xdata[i],-1);
      if (classreg == 1)
	{
	  int lent = activ[0][numhid + 1].n_rows;
	  for (int j = 0; j < lent; j++)
	    {
	      error = error + pow(activ[0][numhid+1](j,0) - testydata[i](j,0),2);
	    }
	}
    }
  error = sqrt(error/(double)xnumlines);
  cout<<"RMSE :"<<error<<endl;
  return;
}


//
void NNet::testvoids(int mode)
{
  vector<double> LRMSE;
  vector<int> counts;
  for (int i = 0; i < numfiles; i++)
    {
      LRMSE.push_back(0);
      counts.push_back(0);
    }
  //int count = 0;
  for(int i = 0; i < l_train; i++)
    {
      for (int j = 0; j < numfiles; j++)
	{
	  int l_numhid;
	  if(qmat == 1)
	    {
	      if (Q_mat[i][j] == 0)
		{
		  l_feedforward(l_xvals[i],j);
		  l_numhid = l_numhids[j];
		  int lent = l_numlayers[j][l_numhid + 1];
		  double err = 0;
		  for(int t = 0; t < lent; t++)
		    {
		      if((mode == 1) && (j == 0)) //temporary just to check why the NN is performing so badly on volume
			{
			  cout<<l_activ[j][l_numhid + 1][t]<<" "<<l_yvals[j][i][t]<<endl;
			}
		      err = err + pow(l_activ[j][l_numhid + 1][t] - l_yvals[j][i][t],2);
		    }
		  LRMSE[j] = LRMSE[j] + err;
		  counts[j]++;
		}
	    }
	  else
	    {
	       l_feedforward(l_xvals[i],j);
	       l_numhid = l_numhids[j];
	       int lent = l_numlayers[j][l_numhid + 1];
	       double err = 0;
	       for(int t = 0; t < lent; t++)
		 {
		   err = err + pow(l_activ[j][l_numhid + 1][t] - l_yvals[j][i][t],2);
		 }
	       LRMSE[j] = LRMSE[j] + err;
	       counts[j]++;
	    }
	}
    }
  //cout<<"COUNT: "<<count<<endl;
  for(int i = 0; i < numfiles; i++)
    {
      double frmse = sqrt(LRMSE[i]/(double)counts[i]);
      //cout<<"RMSE of file "<<filenames[i]<<" is: "<<frmse<<endl;
      double averr = (sqrt(LRMSE[i])/(double)counts[i]);
      if (mode == 1)
	{
	  cout<<"Error of file "<<filenames[i]<<" is: "<<averr<<endl;
	  cout<<"RMSE of file "<<filenames[i]<<" is: "<<frmse<<endl;
	}
    }
}



//RPROP for latent parameter learning
void NNet::l_trainrprop(int numlatent, double tmax, int mode)
{
  int trainmode = mode;
  int rprop = 0;
  vector<thread> l_bpthreads;
  //int revertp = 0;
  //int revertb = 0;
  double rmax = tmax;
  int lat_rprop = 0;
  if ((trainmode != 0) && (trainmode != 1))
    {
      cout<<"Training mode can only be 0 or 1"<<endl;
      return;
    }
  if ((numlatent >= 0) && (l_trained == 0))
    {
      if (l_xvals.empty())
	{
	  for (int i = 0; i < file_nlines; i++)
	    {
	      l_xvals.push_back(randn<mat>(numlatent,1));
	    }
	}
      else
	{
	  for (int i = 0; i < file_nlines; i++)
	    {
	      for (int j = 0; j < numlatent; j++)
		{
		  mat trnum = randn<mat>(1,1);
		  (l_xvals.at(i)).insert_rows(0,trnum);
		}
	    }
	}
      l_numx = l_numx + numlatent;
      for(int i = 0; i < numfiles; i++)
	{
	  l_numlayers[i].insert(l_numlayers[i].begin(),l_numx);
	  l_tdels[i].push_back(zeros<mat>(l_numx,1));
	  for(int j = 0; j < l_numhids[i] + 1; j++)
	    {
	      int rows = l_numlayers[i][j+1];
	      int cols = l_numlayers[i][j];
	      l_params[i].push_back(randn<mat>(rows,cols));
	      l_tgrads[i].push_back(zeros<mat>(rows,cols));
	      l_bias[i].push_back(randn<mat>(rows,1));
	      l_tdels[i].push_back(zeros<mat>(rows,1));
	    }
	}
    }
  else
    {
      if (numlatent < 0)
	{
	  cout<<"Number of latent parameters must be greater than 1"<<endl;
	  return;
	}
      else
	{
	  for(int i = 0; i < numfiles; i++)
	    {
	      l_tdels[i][0].fill(0);
	      for (int j = 0; j < l_numhids[i] + 1; j++)
		{
		  l_tgrads[i][j].fill(0);
		  l_tdels[i][j+1].fill(0);
		}
	    }
	}
    }
  vector<double> lrates;
  for(int i = 0; i < numfiles; i++)
    {
      double rate = 0.0001;
      //cout<<"Please enter the desired learning rate for NN "<<to_string(i+1)<<": ";
      //cin>>rate;
      lrates.push_back(rate);
    }
  if (gradd == 0)
    {
      //int stop = 0;
      //double beta = 0.2;
      for (int k = 0; k < epoch; k++)
	{
	  cout<<(double)k*100/(double)epoch<<"%\n";
	  if (trainmode == 1)
	    {
	      l_testall();
	      cout<<endl;
	    }
	  for (int i = 0; i < train; i++)
	    {
	      int threadcount = 0;
	      for (int j = 0; j < numfiles; j++)
		{
		  if (qmat == 1)
		    {
		      if (Q_mat[i][j] == 1)
			{
			  l_bpthreads.push_back(std::thread(&NNet::l_parallelbp,this,i,j));
		      threadcount++;
			}
		    }
		  else
		    {
		    l_bpthreads.push_back(std::thread(&NNet::l_parallelbp,this,i,j));
		      threadcount++;
		    }
		}
	      for(int j = 0; j < threadcount; j++)
		{
		  l_bpthreads[j].join();
		}
	      for (int q = 0; q < numfiles; q++)
		{
		  if (qmat == 1)
		    {
		      if (Q_mat[i][q] == 1)
			{
			  int lnumhid = l_numhids[q];
			  l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
			  for (int j = 0; j < lnumhid + 1; j++)
			    {
			      l_tgrads[q][j] = l_tgrads[q][j] + l_grads[q][lnumhid-j];
			      l_tdels[q][j+1] = l_tdels[q][j+1] + l_dels[q][lnumhid - j];
			    }
			}
		    }
		  else
		    {
		      int lnumhid = l_numhids[q];
		      l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
		      for (int j = 0; j < lnumhid + 1; j++)
			{
			  l_tgrads[q][j] = l_tgrads[q][j] + l_grads[q][lnumhid-j];
			  l_tdels[q][j+1] = l_tdels[q][j+1] + l_dels[q][lnumhid - j];
			}
		    }
		}
	      for(int q = 0; q < numfiles; q++)
		{
		  if( qmat == 1)
		    {
		      if (Q_mat[i][q] == 1)
			{
			  for(int j = 0; j < numlatent; j++)
			    {
			      l_xvals[i](j,0) = l_xvals[i](j,0) - (lrates[q]/4.0)*l_tdels[q][0](j,0);
			      cout<<"LATENT PARAMS: "<<l_xvals[i](j,0)<<endl;
			      l_tdels[q][0].fill(0); //Doing this is the textbook method but commenting this out just works much much better
			    }
			}
		    }
		  else
		    {
		      for(int j = 0; j < numlatent; j++)
			{
			  l_xvals[i](j,0) = l_xvals[i](j,0) - (lrates[j]/4.0)*l_tdels[q][0](j,0);
			  l_tdels[q][0].fill(0); //Doing this is the textbook method but commenting this out just works much much better
			}
		    }
		}
	    }
	  if (rprop == 0)
	    {
	      for (int fl = 0; fl < numfiles; fl++)
		{
		  int l_numhid = l_numhids[fl];
		  for(int q = 0; q < l_numhid + 1; q++)
		    {
		      l_checkgrads[fl].push_back(l_tgrads[fl][q]);
		      l_checkdels[fl].push_back(l_tdels[fl][q]);
		    }
		}
	    }
	  else
	    {
	      for (int fl = 0; fl < numfiles; fl++)
		{
		  int l_numhid = l_numhids[fl];
		  for(int q = 0; q < l_numhid + 1; q++)
		    {
		      int rows = l_checkgrads[fl][q].n_rows;
		      int cols = l_checkgrads[fl][q].n_cols;
		      for(int rw = 0; rw < rows; rw++)
			{
			  for(int cl = 0; cl < cols; cl++)
			    {
			      if (l_checkgrads[fl][q](rw,cl)*l_tgrads[fl][q](rw,cl) > 0) 
				{
				  //push up weight
				  if (rprop == 1)
				    {
				      double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
				      l_tgrads[fl][q](rw,cl) = 0.1*1.2;
				      l_tgrads[fl][q](rw,cl) = min(l_tgrads[fl][q](rw,cl),rmax);
				      l_checkgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
				    }
				  else
				    {
				      double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
				      l_tgrads[fl][q](rw,cl) = sign*l_checkgrads[fl][q](rw,cl)*1.2;
				      l_tgrads[fl][q](rw,cl) = min(l_tgrads[fl][q](rw,cl),rmax);
				      l_checkgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
				    }
				}
			      else if ((l_checkgrads[fl][q](rw,cl)*l_tgrads[fl][q](rw,cl) < 0))
				{
				  //pushdown weight
				  if (rprop == 1)
				    {
				      double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
				      //l_tgrads[fl][q](rw,cl) = -1*l_checkgrads[fl][q](rw,cl);
				      double temp;
				      temp = 0.1*0.5*(l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl)));
				      temp = max(abs(temp),0.000001);
				      l_checkgrads[fl][q](rw,cl) = sign*temp;
				      l_tgrads[fl][q](rw,cl) = sign*temp;
				    }
				  else
				    {
				      double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
				      //l_tgrads[fl][q](rw,cl) = -1*l_checkgrads[fl][q](rw,cl);
				      double temp;
				      temp = l_checkgrads[fl][q](rw,cl)*0.5;
				      temp = max(abs(temp),0.000001);
				      l_checkgrads[fl][q](rw,cl) = sign*temp;
				      l_tgrads[fl][q](rw,cl) = sign*temp;
				    }
				}
			      else if ((l_checkgrads[fl][q](rw,cl)*l_tgrads[fl][q](rw,cl) == 0))
				{
				  if (rprop == 1)
				    {
				      l_tgrads[fl][q](rw,cl) = 0.1*1.0*(l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl)));
				    }
				  else
				    {
				      l_tgrads[fl][q](rw,cl) = abs(l_checkgrads[fl][q](rw,cl))*1.0*(l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl)));
				    }
				}
			    }
			}
		      //BIAS
		      int brows = l_checkdels[fl][q].n_rows;
		      int bcols = l_checkdels[fl][q].n_cols;
		      for(int rw = 0; rw < brows; rw++)
			{
			  for(int cl = 0; cl < bcols; cl++)
			    {
			      if (l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) > 0)
				{
				  //push up weight
				  if (rprop == 1)
				    {
				      double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
				      l_tdels[fl][q+1](rw,cl) = 0.1*1.2*(l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl)));
				      l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),rmax);
				      l_checkdels[fl][q](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
				    }
				  else
				    {
				      //double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
				      double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
				      l_tdels[fl][q+1](rw,cl) = sign*l_checkdels[fl][q](rw,cl)*1.2;
				      l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),rmax);
				      l_checkdels[fl][q](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
				    }
				}
			      else if ((l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) < 0))
				{
				  //pushdown weight
				  if (rprop == 1)
				    {
				      double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
				      l_tdels[fl][q+1](rw,cl) = sign*0.1*0.5;
				      double temp;
				      temp = 0.1*0.5;
				      temp = max(temp,0.000001);
				      l_checkdels[fl][q](rw,cl) = sign*temp;
				    }
				  else
				    {
				      double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
				      //l_tdels[fl][q+1](rw,cl) = sign*abs(l_checkdels[fl][q](rw,cl))*0.5;
				      double temp;
				      temp = l_checkdels[fl][q](rw,cl)*0.5;
				      temp = max(abs(temp),0.000001);
				      l_checkdels[fl][q](rw,cl) = sign*temp;
				      l_tdels[fl][q+1](rw,cl) = sign*temp;
				    }
				}
			      else if ((l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) == 0))
				{
				  if (rprop == 1)
				    {
				      l_tdels[fl][q+1](rw,cl) = 0.1*1.0*(l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl)));
				      //l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),20.0);
				      //l_checkdels[fl][q](rw,cl) = l_tdels[fl][q+1](rw,cl);
				    }
				  else
				    {
				      l_tdels[fl][q+1](rw,cl) = abs(l_checkdels[fl][q](rw,cl))*1.0*(l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl)));
				      //l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),20.0);
				      //l_checkdels[fl][q](rw,cl) = l_tdels[fl][q+1](rw,cl);
				    }
				}
			    }
			}
		    }
		}
	    }
	  for(int t = 0; t < numfiles; t++)
	    {
	      int lnumhid = l_numhids[t];
	      for (int l = 0; l < lnumhid + 1; l++)
		{
		  if (rprop == 0)
		    {
		      l_params[t][l] = params[t][l] - (lrates[t]/(double)train)*tgrads[t][l] - 0.00001*params[t][l];
		      l_bias[t][l] = l_bias[t][l] - (lrates[t]/(double)train)*l_tdels[t][l+1];
		      l_tgrads[t][l].fill(0);
		      l_tdels[t][l+1].fill(0);
		    }
		  else
		    {
		      l_params[t][l] = l_params[t][l] - l_tgrads[t][l];
		      l_bias[t][l] = l_bias[t][l] - l_tdels[t][l+1];
		      l_tgrads[t][l].fill(0);
		      l_tdels[t][l+1].fill(0);
		    }
		}
	    }
	   if (rprop > 1)
	     {
	       rprop = 3;
	     }
	   else
	     {
	       rprop++;
	     }
	}
    }
  //TODO:MODIFY SGD FOR LATENT PARAMS
  else if (gradd == 1)
    {
      cout<<"Initializing Stochastic Gradient Descent L\n";
      vector<int> idxs;
      for (int i = 0; i < l_train; i++)
	{
	  idxs.push_back(i);
	}
      random_shuffle(idxs.begin(),idxs.end());
      for (int i = 0; i < epoch; i++)
	{
	  cout<<((double)i/(double)epoch)*100<<"%\n";
	  if (trainmode == 1)
	    {
	      l_testall();
	      cout<<endl;
	    }
	  int step = 0;
	  while (step < l_train)
	    {
	      int k = step;
	      step = min(step + 20,l_train);
	      for(;k < step; k++)
		{
		  if (!l_bpthreads.empty())
		    {
		      l_bpthreads.clear();
		    }
		  int threadcount = 0;
		  for (int t = 0; t < numfiles; t++)
		    {
		      //cout<<"Q-val: "<<Q_mat[idxs.at(k)][t]<<endl;
		      if(qmat == 1)
			{
			  if (Q_mat[idxs.at(k)][t] == 1)
			    {
			      l_bpthreads.push_back(thread(&NNet::l_parallelbp,this,idxs.at(k),t));
			      threadcount++;
			    }
			}
		      else
			{
			  l_bpthreads.push_back(thread(&NNet::l_parallelbp,this,idxs.at(k),t));
			  threadcount++;
			}
		    }
		  for(int t = 0; t < threadcount; t++)
		    {
		      l_bpthreads[t].join();
		    }
		  for (int q = 0; q < numfiles; q++)
		    {
		      if (qmat == 1)
			{
			  if (Q_mat[idxs.at(k)][q] == 1)
			    {
			      int lnumhid = l_numhids[q];
			      l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
			      for(int j = 0; j < lnumhid + 1; j++)
				{
				  l_tgrads[q].at(j) = l_tgrads[q].at(j) + l_grads[q].at(lnumhid - j);
				  l_tdels[q].at(j+1) = l_tdels[q].at(j+1) + l_dels[q].at(lnumhid -j);
				}
			    }
			}
		      else
			{
			  int lnumhid = l_numhids[q];
			  l_tdels[q][0] = l_tdels[q][0] + l_dels[q][lnumhid + 1];
			  for(int j = 0; j < lnumhid + 1; j++)
			    {
			      l_tgrads[q].at(j) = l_tgrads[q].at(j) + l_grads[q].at(lnumhid - j);
			      l_tdels[q].at(j+1) = l_tdels[q].at(j+1) + l_dels[q].at(lnumhid -j);
			    }
			}
		    }
		  mat lat_grads = zeros<mat>(numlatent,1);
		  for (int q = 0; q < numfiles; q++)
		    {
		      if(qmat == 1)
			{
			  if (Q_mat[idxs.at(k)][q] == 1)
			    {
			      for(int j = 0; j < numlatent; j++)
				{
				  lat_grads(j,0) = lat_grads(j,0) + l_tdels[q][0](j,0);
				}
			      l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
			    }
			}
		      else
			{
			  for(int j = 0; j < numlatent; j++)
			    {
			      lat_grads(j,0) = lat_grads(j,0) + l_tdels[q][0](j,0);
			    }
			  l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
			} 
		    }
		  if (lat_rprop == 0)
		    {
		      lat_checkgrads = lat_grads;
		    }
		  else
		    {
		      //int revertlat = 0;
		      for(int j = 0; j < numlatent; j++)
			{
			  if (lat_checkgrads(j,0)*lat_grads(j,0) > 0)
			    {
			      if (lat_rprop == 1)
				{
				  //cout<<"P"<<endl;
				  double sign = lat_grads(j,0)/abs(lat_grads(j,0));
				  lat_grads(j,0) = sign*1.2*0.1;
				  lat_checkgrads(j,0) = lat_grads(j,0);
				}
			      else
				{
				  double sign = lat_grads(j,0)/abs(lat_grads(j,0));
				  lat_grads(j,0) = 1.2*abs(lat_checkgrads(j,0));
				  lat_grads(j,0) = min(lat_grads(j,0),rmax);
				  lat_checkgrads(j,0) = sign*lat_grads(j,0);
				  lat_grads(j,0) = sign*lat_grads(j,0);
				}
			    }
			  else if ((lat_checkgrads(j,0)*lat_grads(j,0) < 0))
			    {
			      if(lat_rprop == 1)
				{
				  double sign = lat_grads(j,0)/abs(lat_grads(j,0));
				  lat_grads(j,0) = sign*0.5*0.1;
				  lat_checkgrads(j,0) = lat_grads(j,0);
				}
			      else
				{
				  double sign = lat_grads(j,0)/abs(lat_grads(j,0));
				  //lat_grads(j,0) = sign*abs(lat_checkgrads(j,0))*0.5;
				  double temp = 0.5*abs(lat_checkgrads(j,0));
				  temp = max(temp,0.0000001);
				  lat_checkgrads(j,0) = sign*temp;
				  lat_grads(j,0) = sign*temp;
				}
			    }
			  else if ((lat_checkgrads(j,0)*lat_grads(j,0) == 0))
			    {
			      if(lat_rprop == 1)
				{
				  double sign = lat_grads(j,0)/abs(lat_grads(j,0));
				  lat_grads(j,0) =  sign*0.1;
				}
			      else
				{
				  double sign = lat_grads(j,0)/abs(lat_grads(j,0));
				  lat_grads(j,0) =  sign*abs(lat_checkgrads(j,0));
				}
			    }
			}
		    }
		  if (lat_rprop == 0)
		    {
		      for(int j = 0; j < numlatent; j++)
			{
			  l_xvals[idxs.at(k)](j,0) = l_xvals[idxs.at(k)](j,0) - (0.00001)*lat_grads(j,0);
			  //l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
			}
		    }
		  else
		    {
		      for(int j = 0; j < numlatent; j++)
			{
			  l_xvals[idxs.at(k)](j,0) = l_xvals[idxs.at(k)](j,0) - lat_grads(j,0);
			  //l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
			}
		    }
		  if (lat_rprop >= 1)
		    {
		      lat_rprop = 3;
		    }
		  else
		    {
		      lat_rprop++;
		    }
		}
	      if (rprop == 0)
		{
		  for (int fl = 0; fl < numfiles; fl++)
		    {
		      int l_numhid = l_numhids[fl];
		      for(int q = 0; q < l_numhid + 1; q++)
			{
			  l_checkgrads[fl].push_back(l_tgrads[fl][q]);
			  l_checkdels[fl].push_back(l_tdels[fl][q+1]);
			}
		    }
		}
	      else
		{
		  for (int fl = 0; fl < numfiles; fl++)
		    {
		      //revertp = 0;
		      //revertb = 0;
		      int l_numhid = l_numhids[fl];
		      for(int q = 0; q < l_numhid + 1; q++)
			{
			  int rows = l_checkgrads[fl][q].n_rows;
			  int cols = l_checkgrads[fl][q].n_cols;
			  for(int rw = 0; rw < rows; rw++)
			    {
			      for(int cl = 0; cl < cols; cl++)
				{
				  if (l_checkgrads[fl][q](rw,cl)*l_tgrads[fl][q](rw,cl) > 0) 
				    {
				      //push up weight
				      if (rprop == 1)
					{
					  double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
					  l_tgrads[fl][q](rw,cl) = 0.1*1.2;
					  l_tgrads[fl][q](rw,cl) = min(l_tgrads[fl][q](rw,cl),rmax);
					  l_checkgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
					  l_tgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
					}
				      else
					{
					  double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
					  l_tgrads[fl][q](rw,cl) = sign*l_checkgrads[fl][q](rw,cl)*1.2;
					  l_tgrads[fl][q](rw,cl) = min(l_tgrads[fl][q](rw,cl),rmax);
					  l_checkgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
					  l_tgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
					}
				    }
				  else if ((l_checkgrads[fl][q](rw,cl)*l_tgrads[fl][q](rw,cl) < 0))
				    {
				      //pushdown weight
				      if (rprop == 1)
					{
					  double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
					  l_tgrads[fl][q](rw,cl) = sign*abs(l_checkgrads[fl][q](rw,cl))*0.5;
					  double temp;
					  temp = 0.1*0.5;
					  temp = max(temp,0.000001);
					  temp = 0.001;
					  l_checkgrads[fl][q](rw,cl) = sign*(temp);
					}
				      else
					{
					  double sign = l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl));
					  l_tgrads[fl][q](rw,cl) = sign*abs(l_checkgrads[fl][q](rw,cl))*0.5;
					  double temp;
					  temp = l_checkgrads[fl][q](rw,cl)*0.5;
					  temp = max(abs(temp),0.000001);
					  //temp = 0.001;
					  l_checkgrads[fl][q](rw,cl) = sign*(temp);
					}
				    }
				  else if ((l_checkgrads[fl][q](rw,cl)*l_tgrads[fl][q](rw,cl) == 0))
				    {
				      if (rprop == 1)
					{
					  l_tgrads[fl][q](rw,cl) = 0.1*1.0*(l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl)));
					}
				      else
					{
					  l_tgrads[fl][q](rw,cl) = abs(l_checkgrads[fl][q](rw,cl))*1.0*(l_tgrads[fl][q](rw,cl)/abs(l_tgrads[fl][q](rw,cl)));
					}
				    }
				}
			    }
			  //BIAS
			  int brows = l_checkdels[fl][q].n_rows;
			  int bcols = l_checkdels[fl][q].n_cols;
			  for(int rw = 0; rw < brows; rw++)
			    {
			      for(int cl = 0; cl < bcols; cl++)
				{
				  if (l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) > 0)
				    {
				      if (rprop == 1)
					{
					  
					  double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
					  l_tdels[fl][q+1](rw,cl) = 0.1*1.2;
					  l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),rmax);
					  l_checkdels[fl][q](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
					  l_tdels[fl][q+1](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
					}
				      else
					{
					  double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
					  l_tdels[fl][q+1](rw,cl) = sign*l_checkdels[fl][q](rw,cl)*1.2;
					  l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),rmax);
					  l_checkdels[fl][q](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
					  l_tdels[fl][q+1](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
					}
				    }
				  else if ((l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) < 0))
				    {
				      //pushdown weight
				      if (rprop == 1)
					{
					  double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
					  l_tdels[fl][q+1](rw,cl) = sign*abs(l_checkdels[fl][q](rw,cl))*0.5;
					  double temp;
					  temp = 0.1*0.5;
					  temp = max(abs(temp),0.000001);
					  //temp = 0.0000001;
					  l_checkdels[fl][q](rw,cl) = sign*(temp);
					}
				      else
					{
					  double sign = l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl));
					  l_tdels[fl][q+1](rw,cl) = sign*abs(l_checkdels[fl][q](rw,cl))*0.5;
					  double temp;
					  temp = l_checkdels[fl][q](rw,cl)*0.5;
					  temp = max(abs(temp),0.000001);
					  //temp = 0.0000001;
					  l_checkdels[fl][q](rw,cl) = sign*(temp);
					}
				    }
				  else if ((l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) == 0))
				    {
				      if (rprop == 1)
					{
					  l_tdels[fl][q+1](rw,cl) = 0.1*1.0*(l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl)));
					  //l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),20.0);
					  //l_checkdels[fl][q](rw,cl) = l_tdels[fl][q+1](rw,cl);
					}
				      else
					{
					  l_tdels[fl][q+1](rw,cl) = abs(l_checkdels[fl][q](rw,cl))*1.0*(l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl)));
					  //l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),20.0);
					  //l_checkdels[fl][q](rw,cl) = l_tdels[fl][q+1](rw,cl);
					}
				    }
				}
			    }
			}
		    }
		}
	      for(int q = 0; q < numfiles; q++)
		{
		  int lnumhid = l_numhids[q];
		  for (int j = 0; j < lnumhid + 1; j++)
		    {
		      //the below only applies if regression is going.....
		      if (rprop == 0)
			{
			  l_params[q][j] = l_params[q][j] - (0.00001/(double)l_train)*l_tgrads[q][j] - 0.0001*l_params[q][j];
			  l_bias[q][j] = l_bias[q][j] - (0.00001/(double)l_train)*l_tdels[q][j+1];
			  //cout<<l_tgrads[q][j]<<endl<<endl;
			  l_tgrads[q][j].fill(0.0);
			  l_tdels[q][j+1].fill(0.0);
			}
		      else
			{
			  l_params[q][j] = l_params[q][j] - l_tgrads[q][j];
			  l_bias[q][j] = l_bias[q][j] - l_tdels[q][j+1];
			  l_tgrads[q][j].fill(0.0);
			  l_tdels[q][j+1].fill(0.0);
			}
		    }
		}
	      if (rprop >= 1)
		{
		  rprop = 3;
		}
	      else
		{
		  rprop++;
		}
	      //cout<<"RPROP: "<<rprop<<endl;
	    }
	}
    }
  l_trained++;
  return;
}

void NNet::l_funcarch(void)
{
  if (checkinit == 0)
    {
      for(int i = 0; i < numfiles; i++)
	{
	  string temp;
	  cout<<"Please enter the desired function architecture for "<<filenames[i]<<": ";
	  cin>>temp;
	  if (!l_funclayer[i].empty())
	    {
	      l_funclayer[i].clear();
	    }
	  int lent = temp.length();
	  for(int j = 0; j < lent; j++)
	    {
	      if (isdigit(temp[j]) == 0)
		{
		  cout<<"Invalid input!"<<endl;
		  abort();
		}
	      else
		{
		  string num = "";
		  num = num + temp.at(j);
		  l_funclayer[i].push_back(stoi(num,NULL));
		}
	    }
	}
    }
  else
    {
      cout<<"Please initialize the neural network!"<<endl;
      abort();
    }
  return;
}


//
void NNet::l_testall(void)
{
  vector<double> TRRMSE;
  vector<double> TSRMSE;
  vector<int> tscounts;
  vector<int> trcounts;
  for (int i = 0; i < numfiles; i++)
    {
      TRRMSE.push_back(0);
      TSRMSE.push_back(0);
      tscounts.push_back(0);
      trcounts.push_back(0);
    }
  //int count = 0;
  for(int i = 0; i < l_train; i++)
    {
      for (int j = 0; j < numfiles; j++)
	{
	  int l_numhid;
	  if(qmat == 1)
	    {
	      if (Q_mat[i][j] == 0)
		{
		  l_feedforward(l_xvals[i],j);
		  l_numhid = l_numhids[j];
		  int lent = l_numlayers[j][l_numhid + 1];
		  double err = 0;
		  for(int t = 0; t < lent; t++)
		    {
		      err = err + pow(l_activ[j][l_numhid + 1][t] - l_yvals[j][i][t],2);
		    }
		  TSRMSE[j] = TSRMSE[j] + err;
		  tscounts[j]++; 
		}
	      else
		{
		  l_feedforward(l_xvals[i],j);
		  l_numhid = l_numhids[j];
		  int lent = l_numlayers[j][l_numhid + 1];
		  double err = 0;
		  for(int t = 0; t < lent; t++)
		    {
		      err = err + pow(l_activ[j][l_numhid + 1][t] - l_yvals[j][i][t],2);
		    }
		  TRRMSE[j] = TRRMSE[j] + err;
		  trcounts[j]++; 
		}
	    } 
	  else
	    {
	       l_feedforward(l_xvals[i],j);
	       l_numhid = l_numhids[j];
	       int lent = l_numlayers[j][l_numhid + 1];
	       double err = 0;
	       for(int t = 0; t < lent; t++)
		 {
		   err = err + pow(l_activ[j][l_numhid + 1][t] - l_yvals[j][i][t],2);
		 }
	       TRRMSE[j] = TRRMSE[j] + err;
	       trcounts[j]++;
	    }
	}
    }
  //cout<<"COUNT: "<<count<<endl; 
  for(int i = 0; i < numfiles; i++)
    {
      //cout<<"COUNTS: "<<counts[i]<<endl;
      double frmse = sqrt(TRRMSE[i]/(double)trcounts[i]);
      //cout<<"RMSE of file "<<filenames[i]<<" is: "<<frmse<<endl;
      double averr = (sqrt(TRRMSE[i])/(double)trcounts[i]);
      double tsrmse = sqrt(TSRMSE[i]/(double)tscounts[i]);
      double tserror = sqrt(TSRMSE[i]/(double)tscounts[i]);
      cout<<"Training error"<<endl;
      cout<<"Error of file "<<filenames[i]<<" is: "<<averr<<endl;
      cout<<"RMSE of file "<<filenames[i]<<" is: "<<frmse<<endl;
      if (qmat == 1)
	{
	  cout<<"Test Error"<<endl;
	  cout<<"Error of file "<<filenames[i]<<" is: "<<tserror<<endl;
	  cout<<"RMSE of file "<<filenames[i]<<" is: "<<tsrmse<<endl;
	}
    }
}

