#include<iostream>
#include<armadillo>
#include<string>
#include<cstdlib>
#include<cctype>
#include"NNet.h"
#include<cmath>
#include<boost/python.hpp>
#include<iomanip>

using namespace std;
using namespace arma;

void NNet::parallel_bp(int index, int pos)
{
  backprop(xdata[index],ydata[index],pos);
} 


void NNet::d_parallelbp(int index, int pos)
{
  d_backprop(xdata[index],ydata[index],pos);
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
  temp_rmse = 10000;
  min_rmse = -1;
  qmat = 0;
  trained = 0;
  l_trained = 0;
  tmode = 0;
  to_log = 0;
}


//Activation functions
double NNet::sigmoid(double x)
{
  return 1/(exp((-1)*x) + 1);
}

double NNet::sigmoid_d(double x)
{
  return exp(x)/(pow((exp(x) + 1),2));
}

double NNet::sigmoid_dd(double x)
{
  return -1*((exp(x)*(exp(x) - 1))/(pow((exp(x) + 1),3)));
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

double NNet::tanh_dd(double x)
{
  double sech = 1/cosh(x);
  return -1*(2*(tanh(x))*(sech*sech));
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


double NNet::softplus_D(double x)
{
  return exp(x)/(1 + exp(x));
}


//Initilization method, sets up the type and architecture of neural network to be used
void NNet::init(string sconfig, int iclassreg, int inumcores, int igradd, int icostfunc, int iepoch)
{
  int count = 0;
  int lent = sconfig.length();
  string num = "";
  if (!config.empty())
    {
      config.clear();
    }
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
	  config.push_back(stoi(num,NULL));
	  count++;
	  num = "";
	}
      else if ((i == lent - 1) && (sconfig.at(i) != '-'))
	{
	  num = num + sconfig.at(i);
	  config.push_back(stoi(num,NULL));
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
  if (!funclayer.empty())
    funclayer.clear();
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
  if (!activ.empty())
    {
      activ.clear();
      sums.clear();
      grads.clear();
      dels.clear();
    }
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
  string ex = "e";
  if (!xdata.empty())
    {
      xdata.clear();
      ydata.clear();
    }
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
	  if  ((temp.at(i) != ex.at(0)) && (temp.at(i) != sep1.at(0)) && (temp.at(i) != sep2.at(0)) && (isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)))
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
      train = numlines - tests;
    }
  else if (loadmode == 1)
    {
      train = numlines;
    }
  else
    {
      cout<<"Please enter a value of 0 or 1 for loadmode\n";
      xdata.clear();
      ydata.clear();
      return;
    }
  numdata = numlines;
  if (!numlayers.empty())
    numlayers.clear();
  numlayers.push_back(pcountx);
  for (int i = 0; i < numhid; i++)
    {
      numlayers.push_back(config[i]);
    }
  numlayers.push_back(pcounty);
  //initializing parameter, bias and velocities
  if (!params.empty())
    {
      params.clear();
      velocity.clear();
      tgrads.clear();
      bias.clear();
      tdels.clear();
    }
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
      if (!sums[idx].empty())
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
  if (!sums[idx].empty())
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



void NNet::d_backprop(mat x, mat y, int gpos)
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
  feed_forward(x,idx);
  vector<mat> le_dels;
  if (costfunc == 0)
    {
      le_dels.push_back((activ.at(idx).at(numhid+1) - y));
    }
  else
    {
      //TODO: have to complete results for other cost functions
      return;
    }
  for (int i = 0; i < numhid; i++)
    {
      mat temp;
      mat tle_del = le_dels[i];
      mat derv = sums[idx][numhid + 1 - i];
      if (funclayer[numhid - i] == 3)
	{
	  int n = numlayers[numhid + 1 - i];
	  for (int j = 0; j < n; j++)
	    {
	      if (funclayer[numhid - i] == 0)
		{
		  tle_del(j,0) = tle_del(j,0)*sigmoid_d(derv(j,0));
		}
	      else if(funclayer[numhid - i] == 1)
		{
		  tle_del(j,0) = tle_del(j,0)*tanh_dr(derv(j,0));
		}
	      else if (funclayer[numhid - i] == 3)
		{
		  tle_del(j,0) = tle_del(j,0)*tanh_d(derv(j,0));
		}
	    }
	}
      temp = (params[numhid - i].t())*(tle_del);
      le_dels.push_back(temp);
    }
  if(!d_grads[idx].empty())
    { 
      d_grads[idx].clear();
    }
  if(!d_dels[idx].empty())
    {
      d_dels[idx].clear();
    }
  if (costfunc == 0)
    {
      if (funclayer[numhid] == 0)
	{
	  for (int i = 0; i < numlayers[numhid+1]; i++)
	    {
	      le_dels[0](i,0) = le_dels[0](i,0)*sigmoid_dd(sums[idx][numhid + 1](i,0));
	    }
	  d_dels[idx].push_back(((activ[idx][numhid+1] - (activ.at(idx).at(numhid+1)%activ.at(idx).at(numhid+1)))%(activ[idx][numhid+1] - (activ.at(idx).at(numhid+1)%activ.at(idx).at(numhid+1)))) + le_dels[0]);
	}
      else if (funclayer[numhid] == 1)
	{
	  for (int i = 0; i < numlayers[numhid+1]; i++)
	    {
	      le_dels[0](i,0) = le_dels[0](i,0)*tanh_dd(sums[idx][numhid + 1](i,0));
	    }
	  d_dels[idx].push_back(((ones<mat>(numlayers[numhid+1],1) - activ[idx][numhid+1]%activ[idx][numhid+1])%(ones<mat>(numlayers[numhid+1],1) - activ[idx][numhid+1]%activ[idx][numhid+1])) + le_dels[0]);
	}
      else if (funclayer[numhid] == 2)
	{
	  for (int i = 0; i < numlayers[numhid+1]; i++)
	    {
	      le_dels[0](i,0) = 0.0*le_dels[0](i,0);
	      sums[idx][numhid + 1](i,0) = rec_D(sums[idx][numhid + 1](i,0));
	    }
	  d_dels[idx].push_back((sums[idx][numhid+1]%sums[idx][numhid+1]) + le_dels[0]);
	}
      else if (funclayer[numhid] == 3)
	{
	  for (int i = 0; i < numlayers[numhid+1]; i++)
	    {
	      le_dels[0](i,0) = le_dels[0](i,0)*tanh_dd(sums[idx][numhid + 1](i,0));
	      sums[idx][numhid + 1](i,0) = tanh_d(sums[idx][numhid + 1](i,0));
	    }
	  d_dels[idx].push_back((sums[idx][numhid+1]%sums[idx][numhid+1]) + le_dels[0]);
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
      temp = ((params[numhid - i]%params[numhid - i]).t())*(d_dels[idx][i]);
      mat derv = sums[idx][numhid - i];
      if (i < numhid)
	{
	  if (funclayer[numhid - count] == 0)
	    { 
	      int rows = le_dels[i+1].n_rows;
	      for(int rw = 0; rw < rows; rw++)
		{
		  if (funclayer[numhid-count] == 0)
		    {
		      le_dels[i+1](rw,0) = le_dels[i+1](rw,0)*sigmoid_dd(derv(rw,0));
		    }
		}
	      derv = activ[idx][numhid- i] - (activ[idx][numhid - i]%activ[idx][numhid - i]);
	    }
	  else if (funclayer[numhid-count] == 1)
	    {
	      int n = numlayers[numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  le_dels[i+1](j,0) = le_dels[i+1](j,0)*tanh_dd(derv(j,0));
		  derv(j,0) = tanh_dr(derv(j,0));
		}
	    }
	  else if (funclayer[numhid - count] == 2)
	    {
	      int n = numlayers[numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  le_dels[i+1](j,0) = le_dels[i+1](j,0)*0.0;
		  derv(j,0) = rec_D(derv(j,0));
		}
	    }
	  else if (funclayer[numhid-count] == 3)
	    {
	      int n = numlayers[numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  le_dels[i+1](j,0) = le_dels[i+1](j,0)*tanh_dd(derv(j,0));
		  derv(j,0) = tanh_d(derv(j,0));
		}
	    }
	}
      temp = temp%(derv%derv);
      d_dels[idx].push_back(temp + le_dels[i+1]);
      count++;
    }
  for (int i = 0; i < numhid + 1; i++)
    {
      d_grads[idx].push_back(d_dels[idx][i]*((activ[idx][numhid - i]%activ[idx][numhid - i]).t()));
    }
  return;
}



//Train the neural network
void NNet::train_net(double lrate, int mode, int verbose, string logfile)
{
  if (ydata.empty())
    {
      cout<<"Please load files into the network!"<<endl;
      return;
    }
  int trainmode = mode;
  tmode = trainmode;
  string empt = " ";
  ofstream loggerf;
  ofstream logf;
  if ((logfile[0] != empt[0]) && (trainmode == 1))
    {
      to_log = 1;
      f_log = logfile + ".and";
      f_logger = logfile + "_log.dat";
      loggerf.open(f_logger);
      if (!loggerf.is_open())
	{
	  cout<<"Failed to open file!"<<endl;
	  return;
	}
    }
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
      int cthreads = 0;
      double beta = 0.2;
      for (int k = 0; k < epoch; k++)
	{
	  if (verbose == 0)
	    {
	      cout<<setprecision(5);
	      cout<<"\r"<<((double)k/(double)epoch)*100<<"%"<<flush;
	    }
	  else
	    {
	      if (to_log == 0)
		{
		  cout<<setprecision(5);
		  cout<<((double)k/(double)epoch)*100<<"%"<<endl;
		}
	      else
		{
		  logf.open(f_log, ios::app);
		  if (!logf.is_open())
		    {
		      cout<<"Failed to open log file !"<<endl;
		      return;
		    }
		  cout<<setprecision(2)<<fixed;
		  cout<<"\r"<<((double)k/(double)epoch)*100<<"%"<<flush;
		  logf<<"Epoch No: "<<k<<endl;
		  logf.close();
		  loggerf<<k<<" ";
		}
	    }
	  for (int i = 0; i < train; i = i + numcores)
	    {
	      for(int l = 0; l < numhid + 1; l++)
		{
		  params[l] = params[l] + beta*velocity[l];
		}
	      if (numcores > 1)
		{
		  if (!bpthreads.empty())
		    {
		      bpthreads.clear();
		      cthreads = 0;
		    }
		  for (int t = 0; t < numcores; t++)
		    {
		      if ((i+t) < train)
			{
			  bpthreads.push_back(std::thread(&NNet::parallel_bp,this,i + t,t));
			  cthreads++;
			}
		    }
		  for (int t = 0; t < cthreads; t++)
		    {
		      bpthreads[t].join();
		    }
		}
	      else
		{
		  backprop(xdata[i],ydata[i],0);
		  cthreads = 1;
		}
	      for (int q = 0; q < cthreads; q++)
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
	  if (trainmode == 1)
	    {
	      ttemp_rmse = temp_rmse;
	      if (loadmode == 1)
		{
		  testfile(verbose);
		}
	      else 
		{
		  test_net(verbose);
		}
	      if (min_rmse == -1)
		{
		  min_rmse = temp_rmse;
		}
	      if (to_log == 1)
		{
		  loggerf<<temp_rmse<<"\n";
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
		        params.at(j) = params.at(j) + (lrate)*tgrads.at(j) + kappa*params.at(j);
		        bias.at(j) = bias.at(j) + (lrate)*tdels.at(j);
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
	  if (verbose == 0)
	    {
	      cout<<setprecision(5);
	      cout<<"\r"<<((double)i/(double)epoch)*100<<"%"<<flush;
	    }
	  else
	    {
	      if (to_log == 0)
		{
		  cout<<setprecision(5);
		  cout<<((double)i/(double)epoch)*100<<"%"<<endl;
		}
	      else
		{
		  logf.open(f_log, ios::app);
		  if (!logf.is_open())
		    {
		      cout<<"Failed to open log file !"<<endl;
		      return;
		    }
		  cout<<setprecision(2)<<fixed;
		  cout<<"\r"<<((double)i/(double)epoch)*100<<"%"<<flush;
		  logf<<"Epoch No: "<<i<<endl;
		  logf.close();
		  loggerf<<i<<" ";
		}
	    }
	  int step = 0;
	  int cthreads = 0;
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
		      cthreads = 0;
		    }
		  for (int t = 0; t < ncore; t++)
		    {
		      if (numcores > 1)
			{ 
			  if ((k+t) < train)
			    {
			      bpthreads.push_back(std::thread(&NNet::parallel_bp,this,idxs.at(k+t),t));
			      cthreads++;
			    }
			}
		      else
			{
			  //Threads create an overhead which is avoided if there is only one core available
			    backprop(xdata.at(idxs.at(k+t)),ydata.at(idxs.at(k+t)),-1);
			    cthreads = 1;
			}
		    }
		  if (numcores > 1)
		    {
		      for(int t = 0; t < cthreads; t++)
			{
			  bpthreads[t].join();
			}
		    }
		  for (int q = 0; q < cthreads; q++)
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
		      params.at(j) = params.at(j) - (lrate)*tgrads.at(j) - kappa*params.at(j);
		      //tgrads.at(j).fill(0); //Doing this is the textbook method but commenting this out just works much much better
		    }
		  else
		    {
		      params.at(j) = params.at(j) + velocity.at(j) - kappa*params.at(j);
		    }
		  bias.at(j) = bias.at(j) - (lrate)*tdels.at(j);
		  //tdels.at(j).fill(0);   //Doing this is the textbook method but commenting this out just works much much better
		}
	    }
	  //Below takes extra measures so that the network converges
	  if (trainmode == 1)
	    {
	      ttemp_rmse = temp_rmse;
	      if (loadmode == 1)
		{
		  testfile(verbose);
		}
	      else 
		{
		  test_net(verbose);
		}
	      if (min_rmse == -1)
		{
		  min_rmse = temp_rmse;
		}
	      if (to_log == 1)
		{
		  loggerf<<temp_rmse<<"\n";
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
		        params.at(j) = params.at(j) + (lrate)*tgrads.at(j) + kappa*params.at(j);
		        bias.at(j) = bias.at(j) + (lrate)*tdels.at(j);
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
	    }
	}
    }
  cout<<endl;
  trained = 1;
  if (to_log == 1)
    {
      loggerf.close();
      to_log = 0;
    }
  return;
}


void NNet::update(int r_prop, double r_max)
{
  int rprop = r_prop;
  double rmax = r_max;
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
			  double sign = copysign(1,tgrads[q](rw,cl));
			  tgrads[q](rw,cl) = 0.1*1.2*(tgrads[q](rw,cl)/abs(tgrads[q](rw,cl)));
			  tgrads[q](rw,cl) = min(tgrads[q](rw,cl),rmax);
			  checkgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
			  tgrads[q](rw,cl) = sign*tgrads[q](rw,cl);
			}
		      else
			{
			  double sign = copysign(1,tgrads[q](rw,cl));
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
			  double sign = copysign(1,tgrads[q](rw,cl));
			  double temp;
			  temp = 0.1*0.5;
			  temp = max(temp,0.0000001);
			  checkgrads[q](rw,cl) = sign*temp;
			  tgrads[q](rw,cl) = sign*temp;
			}
		      else
			{
			  double sign = copysign(1,tgrads[q](rw,cl));
			  double temp;
			  temp = checkgrads[q](rw,cl)*0.5;
			  temp = max(abs(temp),0.0000001);
			  checkgrads[q](rw,cl) = sign*(temp);
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
			  double sign = copysign(1,tdels[q](rw,cl));
			  tdels[q](rw,cl) = 0.1*1.2*(tdels[q](rw,cl)/abs(tdels[q](rw,cl)));
			  tdels[q](rw,cl) = min(tdels[q](rw,cl),rmax);
			  checkdels[q](rw,cl) = sign*tdels[q](rw,cl);
			  tdels[q](rw,cl) = sign*tdels[q](rw,cl);
			}
		      else
			{
			  double sign = copysign(1,tdels[q](rw,cl));
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
			  double sign = copysign(1,tdels[q](rw,cl));
			  double temp;
			  temp = 0.1*0.5;
			  temp = max(abs(temp),0.0000001);
			  checkdels[q](rw,cl) = sign*(temp);
			  tdels[q](rw,cl) = sign*temp;
			}
		      else
			{
			  double sign = copysign(1,tdels[q](rw,cl));
			  double temp;
			  temp = checkdels[q](rw,cl)*0.5;
			  temp = max(abs(temp),0.0000001);
			  checkdels[q](rw,cl) = sign*(temp);
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
}



void NNet::rms_prop(int r_prop)
{
  int rprop = r_prop;
  if (rprop == 0)
    {
      for(int q = 0; q < numhid + 1; q++)
	{
	  checkgrads.push_back(tgrads[q]);
	  checkdels.push_back(tdels[q]);
	  int rows = checkgrads[q].n_rows;
	  int cols = checkdels[q].n_cols;
	  for(int rw = 0; rw < rows; rw++)
	    {
	      for(int cl = 0; cl < cols; cl++)
		{
		  checkgrads[q](rw,cl) = abs(checkgrads[q](rw,cl));
		}
	    }
	  int brows = checkdels[q].n_rows;
	  int bcols = checkdels[q].n_cols;
	  for(int rw = 0; rw < brows; rw++)
	    {
	      for(int cl = 0; cl < bcols; cl++)
		{
		  checkdels[q](rw,cl) = abs(checkdels[q](rw,cl));
		}
	    }
	}
    }
  else
    {
      for(int q = 0; q < numhid + 1; q++)
	{
	  checkgrads[q] = 0.9*(checkgrads[q]%checkgrads[q]) + 0.1*(tgrads[q]%tgrads[q]);
	  checkdels[q] = 0.9*(checkdels[q]%checkdels[q]) + 0.1*(tdels[q]%tdels[q]);
	  int rows = checkgrads[q].n_rows;
	  int cols = checkgrads[q].n_cols;
	  for(int rw = 0; rw < rows; rw++)
	    {
	      for(int cl = 0; cl < cols; cl++)
		    {
		      checkgrads[q](rw,cl) = sqrt(checkgrads[q](rw,cl));
		    }
		}
	  int brows = checkdels[q].n_rows;
	  int bcols = checkdels[q].n_cols;
	  for(int rw = 0; rw < brows; rw++)
	    {
	      for(int cl = 0; cl < bcols; cl++)
		{
		  checkdels[q](rw,cl) = sqrt(checkdels[q](rw,cl));
		}
	    }
	}
    }
  return;
}



void NNet::optimalBD(void)
{
  if (!d_tgrads.empty())
    {
      d_tgrads.clear();
    }
  for(int j = 0; j < numhid + 1; j++)
    {
      int rows = numlayers[j+1];
      int cols = numlayers[j];
      d_tgrads.push_back(zeros<mat>(rows,cols));
    }
  vector<thread> dbp_threads;
  int l_count = 0;
  double factc;
  for (int i = 0; i < train; i = i + numcores)
    {
      if (numcores > 1)
	{
	  if (!dbp_threads.empty())
	    {
	      dbp_threads.clear();
	    }
	  for(int j = 0; j < numcores; j++)
	    {
	      dbp_threads.push_back(std::thread(&NNet::d_parallelbp,this,i+j,j));
	    }
	  for(int j = 0; j < numcores; j++)
	    {
	      dbp_threads[j].join();
	    }
	}
      else
	{
	  d_backprop(xdata[i],ydata[i],0);
	}
      l_count++;
      for (int k = 0; k < numcores; k++)
	{
	  reverse(d_grads[k].begin(),d_grads[k].end());
	  for(int j = 0; j < numhid + 1; j++)
	    {
	      d_tgrads[j] = d_tgrads[j] + d_grads[k][j];
	    }
	}
    }
  factc = 1.0/(double)l_count;
  if (!saliencies.empty())
    {
      checksals = saliencies;
      saliencies.clear();
    }
  for(int i = 0; i < numhid + 1; i++)
    {
      saliencies.push_back(factc*0.5*((params[i]%params[i])%d_tgrads[i]));
    }
  double min_s = 1000.0 + saliencies[0](0,0);
  for(int i = 0; i < numhid + 1; i++)
    {
      int rows = saliencies[i].n_rows;
      int cols = saliencies[i].n_cols;
      for(int rw = 0; rw < rows; rw++)
	{
	  for(int cl = 0; cl < cols; cl++)
	    {
	      if (checksals.empty())
		{
		  if (saliencies[i](rw,cl) < min_s)
		    {
		      min_s = saliencies[i](rw,cl);
		    }
		  else
		    {
		      continue;
		    }
		}
	      else
		{
		  if ((saliencies[i](rw,cl) < min_s) && (checksals[i](rw,cl) != 0))
		    {
		      min_s = saliencies[i](rw,cl);
		    }
		  else
		    {
		      continue;
		    } 
		}
	    }
	}
    }
  for(int i = 0; i < numhid + 1; i++)
    {
      int rows = saliencies[i].n_rows;
      int cols = saliencies[i].n_cols;
      double s_tol = min_s/2.0;
      for(int rw = 0; rw < rows; rw++)
	{
	  for(int cl = 0; cl < cols; cl++)
	    {
	      if (abs(saliencies[i](rw,cl) - min_s) < s_tol)
		{
		  saliencies[i](rw,cl) = 0.0;
		}
	      else
		{
		  saliencies[i](rw,cl) = 1.0*checksals[i](rw,cl);
		}
	    }
	}
    }
  return;
}




//Trains the network accroding to RPROP
void NNet::train_rprop(int mode,int verbose, string logfile, double tmax)
{
  if (ydata.empty())
    {
      cout<<"Please load files into the network!"<<endl;
      return;
    }
  if (!checkgrads.empty())
    {
      checkgrads.clear();
      checkdels.clear();
    }
  int trainmode = mode;
  tmode = trainmode;
  string empt = " ";
  ofstream loggerf;
  ofstream logf;
  if ((logfile[0] != empt[0]) && (trainmode == 1))
    {
      to_log = 1;
      f_log = logfile + ".and";
      f_logger = logfile + "_log.dat";
      loggerf.open(f_logger);
      if (!loggerf.is_open())
	{
	  cout<<"Failed to open file!"<<endl;
	  return;
	}
    }
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
  if (gradd == 0)
    {
      int cthreads = 0;
      for (int k = 0; k < epoch; k++)
	{
	  if (verbose == 0)
	    {
	      cout<<setprecision(5);
	      cout<<"\r"<<((double)k/(double)epoch)*100<<"%"<<flush;
	    }
	  else
	    {
	      if (to_log == 0)
		{
		  cout<<setprecision(5);
		  cout<<((double)k/(double)epoch)*100<<"%"<<endl;
		}
	      else
		{
		  logf.open(f_log, ios::app);
		  if (!logf.is_open())
		    {
		      cout<<"Failed to open log file !"<<endl;
		      return;
		    }
		  cout<<setprecision(2)<<fixed;
		  cout<<"\r"<<((double)k/(double)epoch)*100<<"%"<<flush;
		  logf<<"Epoch No: "<<k<<endl;
		  logf.close();
		  loggerf<<k<<" ";
		}
	    }
	  for (int i = 0; i < train; i = i + numcores)
	    {
	      if (numcores > 1)
		{
		  if (!bpthreads.empty())
		    {
		      bpthreads.clear();
		      cthreads = 0;
		    }
		  for (int t = 0; t < numcores; t++)
		    {
		      if ((i+t) < train)
			{
			  bpthreads.push_back(std::thread(&NNet::parallel_bp,this,i + t,t));
			  cthreads++;
			}
		    }
		  for (int t = 0; t < cthreads; t++)
		    {
		      bpthreads[t].join();
		    }
		}
	      else
		{
		  backprop(xdata[i],ydata[i],0);
		  cthreads = 1;
		}
	      for (int q = 0; q < cthreads; q++)
		{
		  for (int j = 0; j < numhid + 1; j++)
		    {
		      tgrads[j] = tgrads[j] + grads[q][numhid-j];
		      tdels[j] = tdels[j] + dels[q][numhid - j];
		    }
		}
	    }
	  update(rprop,rmax);
	  for (int l = 0; l < numhid + 1; l++)
	    {
	      if (rprop == 0)
		{
		  params[l] = params[l] - (0.0001/(double)train)*tgrads[l]; - 0.00001*params[l];
		  bias[l] = bias[l] - (0.0001/(double)train)*tdels[l];
		  tgrads[l].fill(0);
		  tdels[l].fill(0);
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
	  //Below takes extra measures so that the network converges
	  if (trainmode == 1)
	    {
	      if (loadmode == 1)
		{
		  testfile(verbose);
		}
	      else 
		{
		  test_net(verbose);
		}
	      if (min_rmse == -1)
		{
		  min_rmse = temp_rmse;
		}
	      if (to_log == 1)
		{
		  loggerf<<temp_rmse<<"\n";
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
	  if (verbose == 0)
	    {
	      cout<<setprecision(5);
	      cout<<"\r"<<((double)i/(double)epoch)*100<<"%"<<flush;
	    }
	  else
	    {
	     if (to_log == 0)
	       {
		 cout<<setprecision(5);
		 cout<<((double)i/(double)epoch)*100<<"%"<<endl;
	       }
	     else
	       {
		 logf.open(f_log, ios::app);
		 if (!logf.is_open())
		   {
		     cout<<"Failed to open log file !"<<endl;
		     return;
		   }
		 cout<<setprecision(2)<<fixed;
		 cout<<"\r"<<((double)i/(double)epoch)*100<<"%"<<flush;
		 logf<<"Epoch No: "<<i<<endl;
		 logf.close();
		 loggerf<<i<<" ";
	       }
	    }
	  int step = 0;
	  int cthreads = 0;
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
		      cthreads = 0;
		    }
		  for (int t = 0; t < ncore; t++)
		    {
		      if (numcores > 1)
			{ 
			  if ((k+t) < train)
			    {
			      bpthreads.push_back(std::thread(&NNet::parallel_bp,this,idxs.at(k+t),t));
			      cthreads++;
			    }
			}
		      else
			{
			  //Threads create an overhead which is avoided if there is only one core available
			  backprop(xdata.at(idxs.at(k+t)),ydata.at(idxs.at(k+t)),-1);
			  cthreads = 1;
			}
		    }
		  if (numcores > 1)
		    {
		      for(int t = 0; t < cthreads; t++)
			{
			  bpthreads[t].join();
			}
		    }
		  for (int q = 0; q < cthreads; q++)
		    {
		      for(int j = 0; j < numhid + 1; j++)
			{
			  tgrads.at(j) = tgrads.at(j) + grads[q].at(numhid - j);
			  tdels.at(j) = tdels.at(j) + dels[q].at(numhid -j);
			}
		    }
		}
	      rms_prop(rprop);
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
		      params[j] = params[j] - 0.001*(tgrads[j]/checkgrads[j]);
		      bias[j] = bias[j] - 0.001*(tdels[j]/checkdels[j]);
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
	      if (loadmode == 1)
		{
		  testfile(verbose);
		}
	      else 
		{
		  test_net(verbose);
		}
	      if (min_rmse == -1)
		{
		  min_rmse = temp_rmse;
		}
	      if (to_log == 1)
		{
		  loggerf<<temp_rmse<<"\n";
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
	    }
	}
    }
  cout<<endl;
  trained = 1;
  if (to_log == 1)
    {
      loggerf.close();
      to_log = 0;
    }
  return;
}



//Trains the network accroding to RPROP
void NNet::d_trainrprop(int mode, int verbose, string logfile, double tmax)
{
  if (ydata.empty())
    {
      cout<<"Please load files into the network!"<<endl;
      abort();
    }
  if (!checkgrads.empty())
    {
      checkgrads.clear();
      checkdels.clear();
    }
  int trainmode = mode;
  tmode = trainmode;
  string empt = " ";
  ofstream loggerf;
  ofstream logf;
  if ((logfile[0] != empt[0]) && (trainmode == 1))
    {
      to_log = 1;
      f_log = logfile + ".and";
      f_logger = logfile + "_log.dat";
      loggerf.open(f_logger);
      if (!loggerf.is_open())
	{
	  cout<<"Failed to open file!"<<endl;
	  return;
	}
    }
  vector<thread> bpthreads;
  double rmax = tmax;
  vector<mat> tr;
  for (int i = 0; i < numcores; i++)
    {
      d_grads.push_back(tr);
      d_dels.push_back(tr);
    }
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
  if (gradd == 0)
    {
      int cthreads = 0;
      for(int obd = 0; obd < 3; obd++)
	{
	  if (obd == 0)
	    {
	      for(int j = 0; j < numhid + 1; j++)
		{
		  int rows = numlayers[j+1];
		  int cols = numlayers[j];
		  saliencies.push_back(ones<mat>(rows,cols));
		}
	    }
	  for (int k = 0; k < epoch; k++)
	    {
	      if (verbose == 0)
		{
		  cout<<setprecision(5);
		  cout<<"\r"<<((double)k/(double)epoch)*100<<"%"<<flush; 
		}
	      else
		{
		  if (to_log == 0)
		    {
		      cout<<setprecision(5);
		      cout<<((double)k/(double)epoch)*100<<"%"<<endl;
		    }
		  else
		    {
		      logf.open(f_log,ios::app);
		      if (!logf.is_open())
			{
			  cout<<"Failed to open log file !"<<endl;
			  return;
			}
		      cout<<setprecision(2)<<fixed;
		      cout<<"\r"<<((double)k/(double)epoch)*100<<"%"<<flush;
		      logf<<"Epoch No: "<<k<<endl;
		      logf.close();
		      loggerf<<(k + obd*epoch)<<" ";
		    }
		}
	      for (int i = 0; i < train; i = i + numcores)
		{
		  if (numcores > 1)
		    {
		      if (!bpthreads.empty())
			{
			  bpthreads.clear();
			  cthreads = 0;
			}
		      for (int t = 0; t < numcores; t++)
			{
			  if ((i+t) < train)
			    {
			      bpthreads.push_back(std::thread(&NNet::parallel_bp,this,i + t,t));
			      cthreads++;
			    }
			}
		      for (int t = 0; t < cthreads; t++)
			{
			  bpthreads[t].join();
			}
		    }
		  else
		    {
		      backprop(xdata[i],ydata[i],0);
		      cthreads = 1;
		    }
		  for (int q = 0; q < cthreads; q++)
		    {
		      for (int j = 0; j < numhid + 1; j++)
			{
			  tgrads[j] = tgrads[j] + grads[q][numhid-j];
			  tdels[j] = tdels[j] + dels[q][numhid - j];
			}
		    }
		}
	      update(rprop,rmax);
	      for (int l = 0; l < numhid + 1; l++)
		{
		  if (rprop == 0)
		    {
		      params[l] = params[l] - (0.000001)*tgrads[l]; - 0.00001*params[l];
		      bias[l] = bias[l] - (0.0001/(double)train)*tdels[l];
		      params[l] = params[l]%saliencies[l];
		      rprop++;
		      tgrads[l].fill(0);
		      tdels[l].fill(0);
		    }
		  else
		    {
		      params[l] = params[l] - tgrads[l];
		      bias[l] = bias[l] - tdels[l];
		      params[l] = params[l]%saliencies[l];
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
	      //Below takes extra measures so that the network converges
	      if (trainmode == 1)
		{
		  if (loadmode == 1)
		    {
		      testfile(verbose);
		    }
		  else 
		    {
		      test_net(verbose);
		    }
		  if (min_rmse == -1)
		    {
		      min_rmse = temp_rmse;
		    }
		  if (to_log == 1)
		    {
		      loggerf<<temp_rmse<<"\n";
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
		      for (int r = 0; r < numhid + 1; r++)
			{
			  best_params.push_back(params.at(r));
			  best_bias.push_back(bias.at(r));
			  best_velocity.push_back(velocity.at(r));
			}
		    }
		}
	    }
	  optimalBD();
	}
    }
  else if (gradd == 1)
    {
      cout<<"Initializing Stochastic Gradient Descent\n";
      vector<int> idxs;
      int cthreads = 0;
      for (int i = 0; i < train; i++)
	{
	  idxs.push_back(i);
	}
      random_shuffle(idxs.begin(),idxs.end());
      for(int obd = 0; obd < 3; obd++)
	{
	  if (obd == 0)
	    {
	      for(int j = 0; j < numhid + 1; j++)
		{
		  int rows = numlayers[j+1];
		  int cols = numlayers[j];
		  saliencies.push_back(ones<mat>(rows,cols));
		}
	    }
	  for (int i = 0; i < epoch; i++)
	    {
	      if (verbose == 0)
		{
		  cout<<setprecision(5);
		  cout<<"\r"<<((double)i/(double)epoch)*100<<"%"<<flush; 
		}
	      else
		{
		  if (to_log == 0)
		    {
		      cout<<setprecision(5);
		      cout<<((double)i/(double)epoch)*100<<"%"<<endl;
		    }
		  else
		    {
		      logf.open(f_log,ios::app);
		      if (!logf.is_open())
			{
			  cout<<"Failed to open log file !"<<endl;
			  return;
			}
		      cout<<setprecision(2)<<fixed;
		      cout<<"\r"<<((double)i/(double)epoch)*100<<"%"<<flush;
		      logf<<"Epoch No: "<<i<<endl;
		      logf.close();
		      loggerf<<(i + obd*epoch)<<" ";
		    }
		}
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
			  cthreads = 0;
			}
		      for (int t = 0; t < ncore; t++)
			{
			  if (numcores > 1)
			    { 
			      if ((k+t) < train)
				{
				  bpthreads.push_back(std::thread(&NNet::parallel_bp,this,idxs.at(k+t),t));
				  cthreads++;
				}
			    }
			  else
			    {
			      //Threads create an overhead which is avoided if there is only one core available
			      backprop(xdata.at(idxs.at(k+t)),ydata.at(idxs.at(k+t)),-1);
			      cthreads = 1;
			    }
			}
		      if (numcores > 1)
			{
			  for(int t = 0; t < cthreads; t++)
			    {
			      bpthreads[t].join();
			    }
			}
		      for (int q = 0; q < cthreads; q++)
			{
			  for(int j = 0; j < numhid + 1; j++)
			    {
			      tgrads.at(j) = tgrads.at(j) + grads[q].at(numhid - j);
			      tdels.at(j) = tdels.at(j) + dels[q].at(numhid -j);
			    }
			}
		    }
		  rms_prop(rprop);
		  for (int j = 0; j < numhid + 1; j++)
		    {
		      if (rprop == 0)
			{
			  params[j] = params[j] - (0.000001)*tgrads[j]; - 0.00001*params[j];
			  bias[j] = bias[j] - (0.00001/(double)train)*tdels[j];
			  params[j] = params[j]%saliencies[j];
			  rprop++;
			  tgrads[j].fill(0);
			  tdels[j].fill(0);
			}
		      else
			{
			  params[j] = params[j] - 0.001*(tgrads[j]/checkgrads[j]);
			  bias[j] = bias[j] - 0.001*(tdels[j]/checkdels[j]);
			  params[j] = params[j]%saliencies[j];
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
		  if (loadmode == 1)
		    {
		      testfile(verbose);
		    }
		  else 
		    {
		      test_net(verbose);
		    }
		  if (min_rmse == -1)
		    {
		      min_rmse = temp_rmse;
		    }
		  if (to_log == 1)
		    {
		      loggerf<<temp_rmse<<"\n";
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
		      for (int r = 0; r < numhid + 1; r++)
			{
			  best_params.push_back(params.at(r));
			  best_bias.push_back(bias.at(r));
			  best_velocity.push_back(velocity.at(r));
			}
		    }
		}
	    }
	  optimalBD();
	}
    }
  cout<<endl;
  trained = 1;
  if (to_log == 1)
    {
      loggerf.close();
      to_log = 0;
    }
  return;
}




//Test the network
void NNet::test_net(int verbose)
{
  if (loadmode != 0)
    {
      cout<<"Please use test_file(filename) to test this net!\n";
      return;
    }
  int ffmode = 0;
  if ((trained == 1) && (tmode == 1))
    {
      ffmode = -2;
    }
  else
    {
      ffmode = -1;
    }
  int start;
  int stop;
  start = train;
  stop = numdata;
  int passed = 0;
  double error = 0;
  for (int i = start; i < stop; i++)
    {
      feed_forward(xdata[i],ffmode);
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
  double averror = (sqrt(error)/((double)(stop-start)));
  if(classreg == 1)
    {
      temp_rmse = RMSE;
      if (verbose == 1)
	{
	  if (to_log == 0)
	    {
	      cout<<setprecision(5);
	      cout<<"RMSE: "<<RMSE<<endl;
	      cout<<"Average error: "<<averror<<endl;
	    }
	  else
	    {
	      ofstream logf;
	      logf.open(f_log,ios::app);
	      if (!logf.is_open())
		{
		  cout<<"Failed to open log file !"<<endl;
		  return;
		}
	      logf<<setprecision(5);
	      logf<<"RMSE: "<<RMSE<<endl;
	      logf<<"Average error: "<<averror<<endl<<endl;
	      logf.close();
	    }
	}
    }
  else
    {
      temp_rmse = hitrate;
      if (verbose == 1)
	{
	  if (to_log == 0)
	    {
	      cout<<setprecision(5);
	      cout<<"Passed: "<<passed<<endl;
	      cout<<"The accuracy is: "<<hitrate<<"%\n"<<endl;
	    }
	  else
	    {
	      ofstream logf;
	      logf.open(f_log,ios::app);
	      if (!logf.is_open())
		{
		  cout<<"Failed to open log file !"<<endl;
		  return;
		}
	      logf<<setprecision(5);
	      logf<<"Passed: "<<passed<<endl;
	      logf<<"The accuracy is: "<<hitrate<<"%\n"<<endl;
	      logf.close();
	    }
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
  int inc;
  vector<string> names;
  while(getline(savednets,temp))
    {
      inc = 1;
      string tempname = "";
      for (int i = 0; (temp.at(i) != stp.at(0)); i++)
	{
	  tempname = tempname + temp[i];
	}
      if (netname == tempname)
	{
	  check = 1;
	}
      else
	{
	  check = 0;
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
	      inc = 0;
	    }
	  else
	    {
	      cout<<"Save aborted!\n";
	      return;
	    }
	}
      if (inc == 1)
	{
	   names.push_back(temp);
	}
    }
  savednets.close();
  savednets.open(".savednets",fstream::out);
  string funcstring = "";
  for (int i = 0; i < numhid + 1; i++)
    {
      funcstring = funcstring + to_string(funclayer[i]);
    }
  names.push_back(netname +"*"+ to_string(classreg) + "*" + to_string(numhid + 1) + "*" + funcstring);
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
  vector<int> tempfunc;
  while(getline(savednets,temp))
    {
      chk = 0;
      int lent = temp.length();
      int chk1 = 0;
      string tempname = "";
      for (int j = 0; j < lent; j++)
	{
	  if (temp.at(j) == spchar.at(0))
	    {
	      chk1++;
	      continue;
	    }
	  if (chk1 == 0)
	    {
	      tempname = tempname + temp.at(j);
	    }
	  if ((chk1 > 0) && (tempname == netname))
	    {
	      if (chk1 == 1)
		{
		  string tclass = "";
		  tclass = tclass + temp.at(j);
		  classreg = stoi(tclass,NULL);
		}
	      else if (chk1 == 2)
		{
		  num = num + temp.at(j);
		}
	      else if (chk1 == 3)
		{
		  string fstr = "";
		  fstr = fstr + temp.at(j);
		  tempfunc.push_back(stoi(fstr,NULL));
		}
	      chk = 1;
	    }
	}
      if (chk == 1)
	{
	  break;
	}
    }
  if (chk == 0)
    {
      cout<<netname<<" not found!"<<endl;
      return;
    }
  else
    {
      int numparams = stoi(num,NULL);
      numhid = numparams - 1;
      netname = "." + netname + "_";
      funclayer = tempfunc;
      if (!params.empty())
	{
	  params.clear();
	  bias.clear();
	  numlayers.clear();
	}
      for (int i = 0; i < numparams + 1; i++)
	{
	  if (i < numparams)
	    {
	      mat p;
	      mat b;
	      bool stat1 = p.load(netname + "p" + to_string(i));
	      bool stat2 = b.load(netname + "b" + to_string(i));
	      if ((stat1 != true) || (stat2 != true))
		{
		  cout<<"Load failed!\n";
		  abort();
		  return;
		}
	      else
		{
		  params.push_back(p);
		  bias.push_back(b);
		  numlayers.push_back(params[i].n_cols);
		}
	    }
	  else
	    {
	      numlayers.push_back(params[i-1].n_rows);
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
void NNet::test_file(string filename, string netname, string sep1, string sep2)
{
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
  int feedforwardmode;
  string empt = " ";
  int verbose = 1;
  string ex = "e";
  if (netname != empt)
    {
      loadnet(netname);
      feedforwardmode = -1;
       if (activ.empty())
	 {
	   vector<mat> tr;
	   activ.push_back(tr);
	   sums.push_back(tr);
	 }
    }
  else
    {
      if (!best_params.empty())
	{
	  feedforwardmode = -2;
	}
      else
	{
	  feedforwardmode = -1;
	}
    }
  if (!testxdata.empty())
    {
      testxdata.clear();
      testydata.clear();
    }
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
	  if  ((temp.at(i) != ex.at(0)) && (temp.at(i) != sep1.at(0)) && (temp.at(i) != sep2.at(0)) && (isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)))
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
      if (numlines == 0)
	{
	  pcountx = countx;
	  pcounty = county;
	}
      else if ((pcountx != countx) || (pcounty != county))
	{
	  cout<<"Invalid file format, change in size of vectors!\n";
	  return;
	}
      mat mtempx(xvals);
      mat mtempy(yvals);
      testxdata.push_back(mtempx);
      testydata.push_back(mtempy);
      numlines++;
    }
  int passed = 0;
  double error = 0;
  for (int i = 0; i < numlines; i++)
    {
      feed_forward(testxdata.at(i),feedforwardmode);
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
	  int lent = activ[0][numhid + 1].n_rows;
	  for (int j = 0; j < lent; j++)
	    {
	      error = error + pow(testydata[i](j,0) - activ[0][numhid + 1](j,0),2);
	    }
	}
    }
  if(classreg == 1)
    {
      double RMSE = sqrt((error/(double)numlines));
      temp_rmse = RMSE;
      if (verbose == 1)
	{
	  double averror =  (sqrt(error))/(double)numlines;
	  cout<<setprecision(5);
	  cout<<"RMSE: "<<RMSE<<endl;
	  cout<<"Average error: "<<averror<<endl;
	}
    }
  else
    {
      double hitrate = ((double)passed/(double)numlines)*100;
      if (verbose == 1)
	{
	  cout<<passed<<endl;
	  cout<<"The accuracy is: "<<hitrate<<"%"<<endl;
	}
    }
  return;
}




void NNet::testfile(int verbose)
{
  if (loadmode != 1)
    {
      cout<<"Please use test_net() to test this neural net!\n";
      return;
    }
  int feedforwardmode = -1;
  int passed = 0;
  double error = 0;
  for (int i = 0; i < train; i++)
    {
      feed_forward(xdata[i],feedforwardmode);
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
  double hitrate = ((double)passed/(double)train)*100;
  if ((verbose == 1) && (classreg == 0))
    {
      if (to_log == 0)
	{
	  cout<<passed<<endl;
	  cout<<setprecision(5);
	  cout<<"The accuracy is: "<<hitrate<<"%\n"<<endl;
	}
      else 
	{
	  ofstream logf;
	  logf.open(f_log,ios::app);
	  if (!logf.is_open())
	    {
	      cout<<"Failed to open log file !"<<endl;
	      return;
	    }
	  logf<<passed<<endl;
	  logf<<setprecision(5);
	  logf<<"The accuracy is: "<<hitrate<<"%\n"<<endl;
	  logf.close();
	}
    }
  double RMSE = sqrt((error/(double)train));
  if(classreg == 1)
    {
      temp_rmse = RMSE;
      if (verbose == 1)
	{
	  if (to_log == 0)
	    {
	      double averror =  (sqrt(error)/(double)train);
	      cout<<setprecision(5);
	      cout<<"RMSE: "<<RMSE<<endl;
	      cout<<"Average error: "<<averror<<endl<<endl;
	    }
	  else
	    {
	      ofstream logf;
	      logf.open(f_log,ios::app);
	      if (!logf.is_open())
		{
		  cout<<"Failed to open log file !"<<endl;
		  return;
		}
	      double averror =  (sqrt(error)/(double)train);
	      logf<<setprecision(5);
	      logf<<"RMSE: "<<RMSE<<endl;
	      logf<<"Average error: "<<averror<<endl<<endl;
	      logf.close();
	    }
	}
    }
  else
    {
      temp_rmse = hitrate;
    }
  return;
}





/*
 *DEFINED BELOW ARE METHODS TO TEST AN EXPERIMENTAL ALGORITHM FOR LEARNING, ONLY 
 *FOR THE BRAVE AND COURAGEOUS
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
  string num = "";
  for (int i = 0; i < nlent; i++)
    {
      string cmm = ",";
      string septr = "-";
      if ((nconfig[i] != cmm[0]) && (nconfig.at(i) != septr[0]) && (isdigit(nconfig.at(i)) == 0))
	{
	  cout<<"Invalid configuration!, net not initialized!"<<endl;
	  return;
	}
      else if (nconfig[i] == cmm[0])
	{
	  nconfigs.push_back(num);
	  num = "";
	  numfiles++;
	}
      else if (i == (nlent -1))
	{
	  num = num + nconfig[i];
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
	      l_numlayers[j].push_back(stoi(num,NULL));
	      num = "";
	    }
	  else if ((i == lent - 1) && (sconfig.at(i) != '-'))
	    {
	      num = num + sconfig.at(i);
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
  if (!l_funclayer.empty())
    {
      l_funclayer.clear();
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
  if (!l_activ.empty())
    {
      l_activ.clear();
      l_sums.clear();
      l_grads.clear();
      l_dels.clear();
      l_tgrads.clear();
      l_tdels.clear();
      l_checkgrads.clear();
      l_checkdels.clear();
    }
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
  l_trained = 0;
  return;
}



//A
void NNet::ls_load(string outputfiles, string Qmatrix, string input_file, string sep1)
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
  string num = "";
  for (int i = 0; i < nlent; i++)
    {
      string cmma = ",";
      if(outputfiles[i] == cmma[0])
	{
	  filenames.push_back(num);
	  num = "";
	  fcount++;
	}
      else if (i == (nlent - 1))
	{
	  num = num + outputfiles[i];
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
      filenames.clear();
      return;
    }
  if (!l_params.empty())
    {
      l_params.clear();
      l_bias.clear();
      l_yvals.clear();
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
  int numlines = 0;
  for (int j = 0; j < numfiles; j++)
    {
      ifstream ldata(filenames.at(j));
      if (!ldata.is_open())
	{
	  cout<<"Error opening file!\n";
	  l_params.clear();
	  l_bias.clear();
	  l_yvals.clear();
	  return;
	}
      string temp;
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
	      cout<<"Change in length of output!\n";
	      filenames.clear();
	      l_params.clear();
	      l_bias.clear();
	      l_yvals.clear();
	      return;
	    }
	  tempcount = county;
	  county = 0;
	  vector<double> yvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  cout << "Invalid file format!\n";
		  l_params.clear();
		  l_bias.clear();
		  l_yvals.clear();
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
  if (!l_xvals.empty())
    {
      l_xvals.clear();
      Q_mat.clear();
    }
  //LOADING INPUT FILE
  if (input_file.at(0) != empt.at(0))
    {
      //LOADING QMATRIX
      int qnumlines = 0;
      qmat = 1;
      ifstream qdata(Qmatrix);
      if (!qdata.is_open())
	{
	  cout<<"Error opening file!\n";
	  filenames.clear();
	  l_yvals.clear();
	  l_params.clear();
	  l_bias.clear();
	  return;
	}
      string temp;
      int qtempcount = 0;
      int qcounty = 0;
      string qsep = sep1;
      while (getline(qdata,temp))
	{
	  int lent = temp.length();
	  string num = "";
	  if((qtempcount != qcounty) && (qtempcount != 0))
	    {
	      cout<<"Change in length of output!\n";
	      filenames.clear();
	      l_yvals.clear();
	      l_params.clear();
	      l_bias.clear();
	      return;
	    }
	  qtempcount = qcounty;
	  qcounty = 0;
	  vector<int> qvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != qsep.at(0)))
		{
		  cout << "Invalid file format!\n";
		  filenames.clear();
		  l_yvals.clear();
		  l_params.clear();
		  l_bias.clear();
		  Q_mat.clear();
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
	  filenames.clear();
	  Q_mat.clear();
	  l_yvals.clear();
	  l_params.clear();
	  l_bias.clear();
	  return;
	}
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
	      filenames.clear();
	      Q_mat.clear();
	      l_yvals.clear();
	      l_params.clear();
	      l_bias.clear();
	      return;
	    }
	  tempcx = countx;
	  countx = 0;
	  vector<double> xvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  cout << "Invalid file format!\n";
		  filenames.clear();
		  Q_mat.clear();
		  l_yvals.clear();
		  l_params.clear();
		  l_bias.clear();
		  l_xvals.clear();
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
	  mat mtempx(xvals);
	  l_xvals.push_back(mtempx);
	  xnumlines++;
	}
      if (xnumlines != qnumlines)
	{
	  cout<<"Missing Q-matrix entries for certain data points!"<<endl;
	  filenames.clear();
	  Q_mat.clear();
	  l_yvals.clear();
	  l_params.clear();
	  l_bias.clear();
	  l_xvals.clear();
	  return;
	}
      l_numx = countx;
    }
  l_train = numlines; 
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
	      l_numlayers[j].push_back(stoi(num,NULL));
	      num = "";
	    }
	  else if ((i == lent - 1) && (sconfig.at(i) != '-'))
	    {
	      num = num + sconfig.at(i);
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
  if (!l_funclayer.empty())
    {
      l_funclayer.clear();
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
  if (!l_activ.empty())
    {
      l_activ.clear();
      l_sums.clear();
      l_grads.clear();
      l_dels.clear();
      l_tgrads.clear();
      l_tdels.clear();
      l_checkgrads.clear();
      l_checkdels.clear();
    }
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
  l_trained = 0;
  return;
}


//This is a special load method for latent parameters
void NNet::l_load(string Qmatrix, string input_file, string sep1)
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
  if (!l_params.empty())
    {
      l_params.clear();
      l_bias.clear();
      l_yvals.clear();
    }
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
  int numlines = 0;
  for (int j = 0; j < numfiles; j++)
    {
      ifstream ldata(filenames.at(j));
      if (!ldata.is_open())
	{
	  cout<<"Error opening file!\n";
	  filenames.clear();
	  l_params.clear();
	  l_bias.clear();
	  l_yvals.clear();
	  return;
	}
      string temp;
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
	      filenames.clear();
	      l_params.clear();
	      l_bias.clear();
	      l_yvals.clear();
	      return;
	    }
	  tempcount = county;
	  county = 0;
	  vector<double> yvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  cout << "Invalid file format!\n";
		  filenames.clear();
		  l_params.clear();
		  l_bias.clear();
		  l_yvals.clear();
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
  if (!l_xvals.empty())
    {
      l_xvals.clear();
      Q_mat.clear();
    }
  //LOADING INPUT FILE
  if (input_file.at(0) != empt.at(0))
    {
      //LOADING QMATRIX
      int qnumlines = 0;
      qmat = 1;
      ifstream qdata(Qmatrix);
      if (!qdata.is_open())
	{
	  cout<<"Error opening file!\n";
	  filenames.clear();
	  l_params.clear();
	  l_bias.clear();
	  l_yvals.clear();
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
	      filenames.clear();
	      l_params.clear();
	      l_bias.clear();
	      l_yvals.clear();
	      Q_mat.clear();
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
	  filenames.clear();
	  Q_mat.clear();
	  l_yvals.clear();
	  l_params.clear();
	  l_bias.clear();
	  return;
	}
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
	      filenames.clear();
	      l_params.clear();
	      l_bias.clear();
	      l_yvals.clear();
	      Q_mat.clear();
	      l_xvals.clear();
	      return;
	    }
	  tempcx = countx;
	  countx = 0;
	  vector<double> xvals;
	  for(int i = 0; i < lent; i++)
	    {
	      if  ((isdigit(temp.at(i)) == 0) && (temp.at(i) != decp.at(0)) && (temp.at(i) != minussb.at(0)) && (temp.at(i) != sep1.at(0)))
		{
		  cout << "Invalid file format!\n";
		  filenames.clear();
		  l_params.clear();
		  l_bias.clear();
		  l_yvals.clear();
		  Q_mat.clear();
		  l_xvals.clear();
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
	  mat mtempx(xvals);
	  l_xvals.push_back(mtempx);
	  xnumlines++;
	}
      if (xnumlines != qnumlines)
	{
	  cout<<"Missing Q-matrix entries for certain data points!"<<endl;
	  filenames.clear();
	  l_params.clear();
	  l_bias.clear();
	  l_yvals.clear();
	  Q_mat.clear();
	  l_xvals.clear();
	  return;
	}
      l_numx = countx;
    }
  l_train = numlines;
  return;
}


//
void NNet::l_feedforward(mat x,int gpos)
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
	      l_activ[idx].at(count)(j,0) = tanh_r(l_activ[idx].at(count)(j,0));
	    }
	}
      count++;
    }
  return;
}


//This is a slightly modified variant of backprop so learning latent parameters is also incorporated
void NNet::l_backprop(mat x, mat y, int gpos)
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
void NNet::l_trainnet(int numlatent, int mode, double tol)
{
    if (l_yvals.empty())
    {
      cout<<"Please load the files first!"<<endl;
      return;
    }
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
  vector<double> lrates;
  for(int i = 0; i < numfiles; i++)
    {
      double rate;
      cout<<"Please enter the desired learning rate for NN "<<to_string(i+1)<<": ";
      cin>>rate;
      lrates.push_back(rate);
    }
  int tolcheck;
  if (epoch < 0)
    {
      if (tol < 0)
	{
	  cout<<"Please enter a tolerence value!"<<endl;
	  for (int i = 0; i < numfiles; i++)
	    {
	      l_params[i].clear();
	      l_tgrads[i].clear();
	      l_bias[i].clear();
	      l_tdels[i].clear();
	    }
	  return;
	}
      tolcheck = 1;
      epoch = 10;
    }
  else
    {
      tolcheck = 0;
    }
  if (gradd == 0)
    {
      int c_epoch = 0;
      for (int k = 0; k < epoch; k++)
	{
	  if (k == 0 && (trainmode == 1))
	    {
	      cout<<"Initial error"<<endl;
	      l_testall();
	      cout<<endl;
	    }
	  if (tolcheck == 1)
	    {
	      k = -2;
	    }
	  for (int i = 0; i < l_train; i++)
	    {
	      int threadcount = 0;
	      if (!l_bpthreads.empty())
		{
		  l_bpthreads.clear();
		}
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
	  for(int t = 0; t < numfiles; t++)
	    {
	      int lnumhid = l_numhids[t];
	      for (int l = 0; l < lnumhid + 1; l++)
		{
		  l_params[t][l] = params[t][l] - (lrates[t]/(double)train)*tgrads[t][l] - 0.00001*params[t][l];
		  l_bias[t][l] = l_bias[t][l] - (lrates[t]/(double)train)*l_tdels[t][l];
		}
	    }
	  if (tolcheck == 0)
	    {
	      if (trainmode == 0)
		{
		  double pc = ((double)k/(double)epoch)*100;
		  cout<<setprecision(5);
		  cout<<"\r"<<pc<<"%"<<flush;
		}
	      else if (trainmode == 1)
		{
		  cout<<setprecision(5);
		  cout<<((double)k/(double)epoch)*100<<"%"<<endl;
		  l_testall();
		  cout<<endl;
		}
	    }
	  else
	    {
	      if (trainmode == 0)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  l_testall(1);
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		}
	      else if (trainmode == 1)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  l_testall();
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		  cout<<endl;
		}
	    }
	}
    }
  else if (gradd == 1)
    {
      int c_epoch = 0;
      cout<<"Initializing Stochastic Gradient Descent L\n";
      vector<int> idxs;
      for (int i = 0; i < l_train; i++)
	{
	  idxs.push_back(i);
	}
      random_shuffle(idxs.begin(),idxs.end());
      for (int i = 0; i < epoch; i++)
	{
	  if (i == 0 && (trainmode == 1))
	    {
	      cout<<"Initial error"<<endl;
	      l_testall();
	      cout<<endl;
	    }
	  if (tolcheck == 1)
	    {
	      i = -2;
	    }
	  int step = 0;
	  for (int lr = 0; lr < numfiles; lr++)
	    {
	      lrates[lr] = 0.999*lrates[lr];
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
	      double kappa = 0.001;
	      for(int q = 0; q < numfiles; q++)
		{
		  int lnumhid = l_numhids[q];
		  for(int j = 0; j < lnumhid + 1; j++)
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
		      l_bias[q].at(j) = l_bias[q].at(j) - (lrates[q]/(double)100.0)*l_tdels[q].at(j+1);
		      //l_tdels[q].at(j+1).fill(0);    //Doing this is the textbook method but commenting this out just works much much better
		    }
		}
	    }
	  if (tolcheck == 0)
	    {
	      if (trainmode == 0)
		{
		  double pc = ((double)i/(double)epoch)*100;
		  cout<<setprecision(5);
		  cout<<"\r"<<pc<<"%"<<flush;
		}
	      else if (trainmode == 1)
		{
		  cout<<setprecision(5);
		  cout<<((double)i/(double)epoch)*100<<"%"<<endl;
		  l_testall();
		  cout<<endl;
		}
	    }
	  else
	    {
	      if (trainmode == 0)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  l_testall(1);
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		}
	      else if (trainmode == 1)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  l_testall();
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		  cout<<endl;
		}
	    }
	}
    }
  l_trained++;
  cout<<endl;
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
  int inc;
  vector<string> names;
  while(getline(savednets,temp))
    {
      inc = 1;
      string tempname = "";
      for (int i = 0; (temp.at(i) != stp.at(0)); i++)
	{
	  tempname = tempname + temp[i];
	}
      if (netname == tempname)
	{
	  check = 1;
	}
      else
	{
	  check = 0;
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
	      inc = 0;
	    }
	  else
	    {
	      cout<<"Save aborted!\n";
	      return;
	    }
	}
      if (inc == 1)
	{
	   names.push_back(temp);
	}
    }
  savednets.close();
  savednets.open(".savednets",fstream::out);
  int l_numhid = l_numhids[index];
  string funcstring = "";
  for (int i = 0; i < l_numhid + 1; i++)
    {
      funcstring = funcstring + to_string(l_funclayer[index][i]);
    }
  names.push_back(netname + "*" + to_string(classreg) +"*" + to_string(l_numhid + 1) + "*" + funcstring);
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
	      saveinput<<fixed<<l_xvals[i](j,0);
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


void NNet::ls_savenet(string names, string in_name)
{
  vector<string> netnames;
  int nlent = names.length();
  string num = "";
  for (int i = 0; i < nlent; i++)
    {
      string cmm = ",";
      if (names[i] == cmm[0])
	{
	  netnames.push_back(num);
	  num = "";
	}
      else if (i == (nlent -1))
	{
	  num = num + names[i];
	  netnames.push_back(num);
	  num = "";
	}
      else
	{
	  num = num + names[i];
	}
    }
  for(int i = 0; i < numfiles; i++)
    {
      lsavenets(netnames.at(i),i);
    }
  string input_name = in_name;
  string ans = "y";
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
  if (!testxdata.empty())
    {
      testxdata.clear();
      testydata.clear();
    }
  if (activ.empty())
     {
       vector<mat> tr;
       activ.push_back(tr);
       sums.push_back(tr);
     }
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
      mat mtempy(yvals);
      testydata.push_back(mtempy);
      ynumlines++;
    }
  ldata.close();
  if (xnumlines != ynumlines)
    {
      cout<<"The amount of input data is different from the amount of output data, line numbers in both files do not match!"<<endl;
      return;
    }
  loadnet(netname);
  double error = 0;
  for(int i = 0; i < xnumlines; i++)
    {
      feed_forward(testxdata[i],-1);
      if (classreg == 1)
	{
	  int lent = activ[0][numhid + 1].n_rows;
	  for (int j = 0; j < lent; j++)
	    {
	      error = error + pow(activ[0][numhid+1](j,0) - testydata[i](j,0),2);
	    }
	}
    }
  double rmse = sqrt(error/(double)xnumlines);
  error = sqrt(error)/(double)xnumlines;
  cout<<setprecision(5);
  cout<<"Average error: "<<error<<endl;
  cout<<"RMSE :"<<rmse<<endl;
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
		      err = err + pow(l_activ[j][l_numhid + 1](t,0) - l_yvals[j][i](t,0),2);
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
		   err = err + pow(l_activ[j][l_numhid + 1](t,0) - l_yvals[j][i](t,0),2);
		 }
	       LRMSE[j] = LRMSE[j] + err;
	       counts[j]++;
	    }
	}
    }
  for(int i = 0; i < numfiles; i++)
    {
      double frmse = sqrt(LRMSE[i]/(double)counts[i]);
      double averr = (sqrt(LRMSE[i])/(double)counts[i]);
      if (mode == 1)
	{
	  cout<<setprecision(5);
	  cout<<"Error of file "<<filenames[i]<<" is: "<<averr<<endl;
	  cout<<"RMSE of file "<<filenames[i]<<" is: "<<frmse<<endl;
	}
    }
}



void NNet::l_update(int r_prop, double r_max)
{
  double rmax = r_max;
  int rprop = r_prop;
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
			       double sign = copysign(1,l_tgrads[fl][q](rw,cl));
			       l_tgrads[fl][q](rw,cl) = 0.1*1.2;
			       l_tgrads[fl][q](rw,cl) = min(l_tgrads[fl][q](rw,cl),rmax);
			       l_checkgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
			       l_tgrads[fl][q](rw,cl) = sign*l_tgrads[fl][q](rw,cl);
			     }
			   else
			     {
			       double sign = copysign(1,l_tgrads[fl][q](rw,cl));
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
			       double sign = copysign(1,l_tgrads[fl][q](rw,cl));
			       double temp;
			       temp = 0.1*0.5;
			       temp = max(temp,0.0000001);
			       l_checkgrads[fl][q](rw,cl) = sign*(temp);
			       l_tgrads[fl][q](rw,cl) = sign*temp;
			     }				    
			   else
			     {
			       double sign = copysign(1,l_tgrads[fl][q](rw,cl));
			       double temp;
			       temp = l_checkgrads[fl][q](rw,cl)*0.5;
			       temp = max(abs(temp),0.0000001);
			       l_checkgrads[fl][q](rw,cl) = sign*(temp);
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
			   //push up bias
			   if (rprop == 1)
			     {
			       double sign = copysign(1,l_tdels[fl][q+1](rw,cl));
			       l_tdels[fl][q+1](rw,cl) = 0.1*1.2;
			       l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),rmax);
			       l_checkdels[fl][q](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
			       l_tdels[fl][q+1](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
			     }
			   else
			     {
			       double sign = copysign(1,l_tdels[fl][q+1](rw,cl));
			       l_tdels[fl][q+1](rw,cl) = sign*l_checkdels[fl][q](rw,cl)*1.2;
			       l_tdels[fl][q+1](rw,cl) = min(l_tdels[fl][q+1](rw,cl),rmax);
			       l_checkdels[fl][q](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
			       l_tdels[fl][q+1](rw,cl) = sign*l_tdels[fl][q+1](rw,cl);
			     }
			 }
		       else if ((l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) < 0))
			 {
			   //pushdown bias
			   if (rprop == 1)
			     {
			       double sign = copysign(1,l_tdels[fl][q+1](rw,cl));
			       double temp;
			       temp = 0.1*0.5;
			       temp = max(abs(temp),0.0000001);
			       l_checkdels[fl][q](rw,cl) = sign*(temp);
			       l_tdels[fl][q+1](rw,cl) = sign*temp;
			     }
			   else
			     {
			       double sign = copysign(1,l_tdels[fl][q+1](rw,cl));
			       double temp;
			       temp = l_checkdels[fl][q](rw,cl)*0.5;
			       temp = max(abs(temp),0.0000001);
			       l_checkdels[fl][q](rw,cl) = sign*(temp);
			       l_tdels[fl][q+1](rw,cl) = sign*temp;
			     }
			 }
		       else if ((l_checkdels[fl][q](rw,cl)*l_tdels[fl][q+1](rw,cl) == 0))
			 {
			   if (rprop == 1)
			     {
			       l_tdels[fl][q+1](rw,cl) = 0.1*1.0*(l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl)));
			     }
			   else
			     {
			       l_tdels[fl][q+1](rw,cl) = abs(l_checkdels[fl][q](rw,cl))*1.0*(l_tdels[fl][q+1](rw,cl)/abs(l_tdels[fl][q+1](rw,cl)));
			     }
			 }
		     }
		 }
	     }
	 }
     }
}



//RMSPROP for SGD 
void NNet::l_rmsprop(int r_prop)
{
  int rprop = r_prop;
  if (rprop == 0)
    {
      for (int fl = 0; fl < numfiles; fl++)
	{
	  int l_numhid = l_numhids[fl];
	  for(int q = 0; q < l_numhid + 1; q++)
	    {
	      l_checkgrads[fl].push_back(l_tgrads[fl][q]);
	      l_checkdels[fl].push_back(l_tdels[fl][q+1]);
	      int rows = l_checkgrads[fl][q].n_rows;
	      int cols = l_checkgrads[fl][q].n_cols;
	      for(int rw = 0; rw < rows; rw++)
		{
		  for(int cl = 0; cl < cols; cl++)
		    {
		      l_checkgrads[fl][q](rw,cl) = abs(l_checkgrads[fl][q](rw,cl));
		    }
		}
	      int brows = l_checkdels[fl][q].n_rows;
	      int bcols = l_checkdels[fl][q].n_cols;
	      for(int rw = 0; rw < brows; rw++)
		{
		  for(int cl = 0; cl < bcols; cl++)
		    {
		      l_checkdels[fl][q](rw,cl) = abs(l_checkdels[fl][q](rw,cl));
		    }
		}
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
	      l_checkgrads[fl][q] = 0.9*(l_checkgrads[fl][q]%l_checkgrads[fl][q]) + 0.1*(l_tgrads[fl][q]%l_tgrads[fl][q]);
	      l_checkdels[fl][q] = 0.9*(l_checkdels[fl][q]%l_checkdels[fl][q]) + 0.1*(l_tdels[fl][q+1]%l_tdels[fl][q+1]);
	      int rows = l_checkgrads[fl][q].n_rows;
	      int cols = l_checkgrads[fl][q].n_cols;
	      for(int rw = 0; rw < rows; rw++)
		{
		  for(int cl = 0; cl < cols; cl++)
		    {
		      l_checkgrads[fl][q](rw,cl) = sqrt(l_checkgrads[fl][q](rw,cl));
		    }
		}
	      int brows = l_checkdels[fl][q].n_rows;
	      int bcols = l_checkdels[fl][q].n_cols;
	      for(int rw = 0; rw < brows; rw++)
		{
		  for(int cl = 0; cl < bcols; cl++)
		    {
		      l_checkdels[fl][q](rw,cl) = sqrt(l_checkdels[fl][q](rw,cl));
		    }
		}
	    }
	}
    }
  return;
}




//RPROP for latent parameter learning
void NNet::l_trainrprop(int numlatent, double tmax, int mode, double tol)
{
  if (l_yvals.empty())
    {
      cout<<"Please load the files first!"<<endl;
      return;
    }
  for (int fl = 0; fl < numfiles; fl++)
    {
      int l_numhid = l_numhids[fl];
      for(int q = 0; q < l_numhid + 1; q++)
	{
	  if (!l_checkgrads[fl].empty())
	    {
	      l_checkgrads[fl].clear();
	      l_checkdels[fl].clear();
	    }
	}
    }
  int trainmode = mode;
  int rprop = 0;
  vector<thread> l_bpthreads;
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
      lrates.push_back(rate);
    }
  int tolcheck;
  if (epoch < 0)
    {
      if (tol < 0)
	{
	  cout<<"Please enter a tolerence value!"<<endl;
	  for (int i = 0; i < numfiles; i++)
	    {
	      l_params[i].clear();
	      l_tgrads[i].clear();
	      l_bias[i].clear();
	      l_tdels[i].clear();
	    }
	  return;
	}
      tolcheck = 1;
      epoch = 10;
    }
  else
    {
      tolcheck = 0;
    }
  if (gradd == 0)
    {
      int c_epoch = 0;
      for (int k = 0; k < epoch; k++)
	{
	  if (k == 0 && (trainmode == 1))
	    {
	      cout<<"Initial error"<<endl;
	      l_testall();
	      cout<<endl;
	    }
	  if (tolcheck == 1)
	    {
	      k = -2;
	    }
	  for (int i = 0; i < l_train; i++)
	    {
	      int threadcount = 0;
	      if (!l_bpthreads.empty())
		{
		  l_bpthreads.clear();
		}
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
	      mat lat_grads = zeros<mat>(numlatent,1);
	      for (int q = 0; q < numfiles; q++)
		{
		  if(qmat == 1)
		    {
		      if (Q_mat[i][q] == 1)
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
		  for(int j = 0; j < numlatent; j++)
		    {
		      if (lat_checkgrads(j,0)*lat_grads(j,0) > 0)
			{
			  if (lat_rprop == 1)
			    {
			      double sign = copysign(1,lat_grads(j,0));
			      lat_grads(j,0) = sign*1.2*0.1;
			      lat_checkgrads(j,0) = lat_grads(j,0);
			    }
			  else
			    {
			      double sign = copysign(1,lat_grads(j,0));
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
			      double sign = copysign(1,lat_grads(j,0));
			      lat_grads(j,0) = sign*0.5*0.1;
			      lat_checkgrads(j,0) = lat_grads(j,0);
			    }
			  else
			    {
			      double sign = copysign(1,lat_grads(j,0));
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
			      double sign = copysign(1,lat_grads(j,0));
			      lat_grads(j,0) =  sign*0.1;
			    }
			  else
			    {
			      double sign = copysign(1,lat_grads(j,0));
			      lat_grads(j,0) =  sign*abs(lat_checkgrads(j,0));
			    }
			}
		    }
		}
	      if (lat_rprop == 0)
		{
		  for(int j = 0; j < numlatent; j++)
		    {
		      l_xvals[i](j,0) = l_xvals[i](j,0) - (0.00001)*lat_grads(j,0);
		      //l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
		    }
		}
	      else
		{
		  for(int j = 0; j < numlatent; j++)
		    {
		      l_xvals[i](j,0) = l_xvals[i](j,0) - lat_grads(j,0);
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
	  l_update(rprop,rmax);
	  for(int t = 0; t < numfiles; t++)
	    {
	      int lnumhid = l_numhids[t];
	      for (int l = 0; l < lnumhid + 1; l++)
		{
		  if (rprop == 0)
		    {
		      l_params[t][l] = l_params[t][l] - 0.00000001*l_tgrads[t][l] - 0.00001*l_params[t][l];
		      l_bias[t][l] = l_bias[t][l] - 0.00000001*l_tdels[t][l+1];
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
	  if (tolcheck == 0)
	    {
	      if (trainmode == 0)
		{
		  double pc = ((double)k/(double)epoch)*100;
		  cout<<setprecision(5);
		  cout<<"\r"<<pc<<"%"<<flush;
		}
	      else if (trainmode == 1)
		{
		  cout<<setprecision(5);
		  cout<<((double)k/(double)epoch)*100<<"%"<<endl;
		  l_testall();
		  cout<<endl;
		}
	    }
	  else
	    {
	      if (trainmode == 0)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  cout<<endl;
		  l_testall(1);
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		}
	      else if (trainmode == 1)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  cout<<endl;
		  l_testall();
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		  cout<<endl;
		}
	    }
	}
    }
  else if (gradd == 1)
    {
      int c_epoch = 0;
      cout<<"Initializing Stochastic Gradient Descent"<<endl;
      double pc = 0;
      vector<int> idxs;
      for (int i = 0; i < l_train; i++)
	{
	  idxs.push_back(i);
	}
      random_shuffle(idxs.begin(),idxs.end());
      for (int i = 0; i < epoch; i++)
	{
	  if (i == 0 && (trainmode == 1))
	    {
	      cout<<"Initial error"<<endl;
	      l_testall();
	      cout<<endl;
	    }
	  if (tolcheck == 1)
	    {
	      i = -2;
	    }
	  int step = 0;
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
		      for(int j = 0; j < numlatent; j++)
			{
			  if (lat_checkgrads(j,0)*lat_grads(j,0) > 0)
			    {
			      if (lat_rprop == 1)
				{
				  double sign = copysign(1,lat_grads(j,0));
				  lat_grads(j,0) = sign*1.2*0.1;
				  lat_checkgrads(j,0) = lat_grads(j,0);
				}
			      else
				{
				  double sign = copysign(1,lat_grads(j,0));
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
				  double sign = copysign(1,lat_grads(j,0));
				  lat_grads(j,0) = sign*0.5*0.1;
				  lat_checkgrads(j,0) = lat_grads(j,0);
				}
			      else
				{
				  double sign = copysign(1,lat_grads(j,0));
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
				  double sign = copysign(1,lat_grads(j,0));
				  lat_grads(j,0) =  sign*0.1;
				}
			      else
				{
				  double sign = copysign(1,lat_grads(j,0));
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
	      l_rmsprop(rprop);
	      for(int q = 0; q < numfiles; q++)
		{
		  int lnumhid = l_numhids[q];
		  for (int j = 0; j < lnumhid + 1; j++)
		    {
		      if (rprop == 0)
			{
			  l_params[q][j] = l_params[q][j] - (0.0000001)*l_tgrads[q][j] - 0.0001*l_params[q][j];
			  l_bias[q][j] = l_bias[q][j] - (0.0000001)*l_tdels[q][j+1];
			  l_tgrads[q][j].fill(0.0);
			  l_tdels[q][j+1].fill(0.0);
			}
		      else
			{
			  l_params[q][j] = l_params[q][j] - 0.001*(l_tgrads[q][j]/l_checkgrads[q][j]);
			  l_bias[q][j] = l_bias[q][j] - 0.001*(l_tdels[q][j+1]/l_checkdels[q][j]);
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
	    }
	  if (tolcheck == 0)
	    {
	      if (trainmode == 0)
		{
		  pc = ((double)i/(double)epoch)*100;
		  cout<<setprecision(5);
		  cout<<"\r"<<pc<<"%"<<flush;
		}
	      else if (trainmode == 1)
		{
		  cout<<setprecision(5);
		  cout<<((double)i/(double)epoch)*100<<"%"<<endl;
		  l_testall();
		  cout<<endl;
		}
	    }
	  else
	    {
	      if (trainmode == 0)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  cout<<endl;
		  l_testall(1);
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		}
	      else if (trainmode == 1)
		{
		  cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		  cout<<endl;
		  l_testall();
		  if (l_error <= tol)
		    {
		      cout<<"Convergence reached!"<<endl;
		      return;
		    }
		  c_epoch++;
		  cout<<endl;
		}
	    }
	}
    }
  l_trained++;
  cout<<endl;
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
		  l_funclayer.clear();
		  return;
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
      return;
    }
  return;
}


//
void NNet::l_testall(int mode)
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
		      err = err + pow(l_activ[j][l_numhid + 1](t,0) - l_yvals[j][i](t,0),2);
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
		      err = err + pow(l_activ[j][l_numhid + 1](t,0) - l_yvals[j][i](t,0),2);
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
		   err = err + pow(l_activ[j][l_numhid + 1](t,0) - l_yvals[j][i](t,0),2);
		 }
	       TRRMSE[j] = TRRMSE[j] + err;
	       trcounts[j]++;
	    }
	}
    }
  l_error = 0;
  for(int i = 0; i < numfiles; i++)
    {
      double frmse = sqrt(TRRMSE[i]/(double)trcounts[i]);
      double averr = (sqrt(TRRMSE[i])/(double)trcounts[i]);
      if (mode == 0)
	{
	  cout<<setprecision(5);
	  cout<<"Training error"<<endl;
	  cout<<"Error of file "<<filenames[i]<<" is: "<<averr<<endl;
	  cout<<"RMSE of file "<<filenames[i]<<" is: "<<frmse<<endl;
	}
      if (qmat == 1)
	{
	  double tsrmse = sqrt(TSRMSE[i]/(double)tscounts[i]);
	  double tserror = sqrt(TSRMSE[i])/(double)tscounts[i];
	  l_error = l_error + tserror;
	  if (mode == 0)
	    {
	      cout<<setprecision(5);
	      cout<<"Test Error"<<endl;
	      cout<<"Error of file "<<filenames[i]<<" is: "<<tserror<<endl;
	      cout<<"RMSE of file "<<filenames[i]<<" is: "<<tsrmse<<endl;
	    }
	}
    }
}



//Calculates the second order gradients
void NNet::ld_backprop(mat x, mat y, int gpos)
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
  int l_numhid = l_numhids.at(idx);
  l_feedforward(x,idx);
  vector<mat> le_dels;
  if (costfunc == 0)
    {
      le_dels.push_back((l_activ.at(idx).at(l_numhid+1) - y));
    }
  else
    {
      //TODO: have to complete results for other cost functions
      return;
    }
  for (int i = 0; i < l_numhid; i++)
    {
      mat temp;
      mat tle_del = le_dels[i];
      mat derv = l_sums[idx][l_numhid + 1 - i];
      if (l_funclayer[idx][l_numhid - i] == 3)
	{
	  int n = l_numlayers[idx][l_numhid + 1 - i];
	  for (int j = 0; j < n; j++)
	    {
	      if (l_funclayer[idx][l_numhid - i] == 0)
		{
		  tle_del(j,0) = tle_del(j,0)*sigmoid_d(derv(j,0));
		}
	      else if(l_funclayer[idx][l_numhid - i] == 1)
		{
		  tle_del(j,0) = tle_del(j,0)*tanh_dr(derv(j,0));
		}
	      else if (l_funclayer[idx][l_numhid - i] == 3)
		{
		  tle_del(j,0) = tle_del(j,0)*tanh_d(derv(j,0));
		}
	    }
	}
      temp = (l_params[idx][l_numhid - i].t())*(tle_del);
      le_dels.push_back(temp);
    }
  if(!ld_grads[idx].empty())
    { 
      ld_grads[idx].clear();
    }
  if(!ld_dels[idx].empty())
    {
      ld_dels[idx].clear();
    }
  if (costfunc == 0)
    {
      if (l_funclayer[idx][l_numhid] == 0)
	{
	  for (int i = 0; i < l_numlayers[idx][l_numhid+1]; i++)
	    {
	      le_dels[0](i,0) = le_dels[0](i,0)*sigmoid_dd(l_sums[idx][l_numhid + 1](i,0));
	    }
	  ld_dels[idx].push_back(((l_activ[idx][l_numhid+1] - (l_activ.at(idx).at(l_numhid+1)%l_activ.at(idx).at(l_numhid+1)))%(l_activ[idx][l_numhid+1] - (l_activ.at(idx).at(l_numhid+1)%l_activ.at(idx).at(l_numhid+1)))) + le_dels[0]);
	}
      else if (l_funclayer[idx][l_numhid] == 1)
	{
	  for (int i = 0; i < l_numlayers[idx][l_numhid+1]; i++)
	    {
	      le_dels[0](i,0) = le_dels[0](i,0)*tanh_dd(l_sums[idx][l_numhid + 1](i,0));
	    }
	  ld_dels[idx].push_back(((ones<mat>(numlayers[l_numhid+1],1) - l_activ[idx][l_numhid+1]%l_activ[idx][l_numhid+1])%(ones<mat>(l_numlayers[idx][l_numhid+1],1) - l_activ[idx][l_numhid+1]%l_activ[idx][l_numhid+1])) + le_dels[0]);
	}
      else if (l_funclayer[idx][l_numhid] == 2)
	{
	  for (int i = 0; i < l_numlayers[idx][l_numhid+1]; i++)
	    {
	      le_dels[0](i,0) = 0.0*le_dels[0](i,0);
	      l_sums[idx][l_numhid + 1](i,0) = rec_D(l_sums[idx][l_numhid + 1](i,0));
	    }
	  ld_dels[idx].push_back((l_sums[idx][l_numhid+1]%l_sums[idx][l_numhid+1]) + le_dels[0]);
	}
      else if (l_funclayer[idx][l_numhid] == 3)
	{
	  for (int i = 0; i < l_numlayers[idx][l_numhid+1]; i++)
	    {
	      le_dels[0](i,0) = le_dels[0](i,0)*tanh_dd(l_sums[idx][l_numhid + 1](i,0));
	      l_sums[idx][l_numhid + 1](i,0) = tanh_d(l_sums[idx][l_numhid + 1](i,0));
	    }
	  ld_dels[idx].push_back(l_sums[idx][l_numhid+1]%l_sums[idx][l_numhid+1] + le_dels[0]);
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
      temp = ((l_params[idx][l_numhid - i]%l_params[idx][l_numhid - i]).t())*(ld_dels[idx][i]);
      mat derv = l_sums[idx][l_numhid - i];
      if (i < l_numhid)
	{
	  if (l_funclayer[idx][l_numhid - count] == 0)
	    { 
	      int rows = le_dels[i+1].n_rows;
	      for(int rw = 0; rw < rows; rw++)
		{
		  if (l_funclayer[idx][l_numhid-count] == 0)
		    {
		      le_dels[i+1](rw,0) = le_dels[i+1](rw,0)*sigmoid_dd(derv(rw,0));
		    }
		}
	      derv = l_activ[idx][l_numhid- i] - (l_activ[idx][l_numhid - i]%l_activ[idx][l_numhid - i]);
	    }
	  else if (l_funclayer[idx][l_numhid-count] == 1)
	    {
	      int n = l_numlayers[idx][l_numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  le_dels[i+1](j,0) = le_dels[i+1](j,0)*tanh_dd(derv(j,0));
		  derv(j,0) = tanh_dr(derv(j,0));
		}
	    }
	  else if (l_funclayer[idx][l_numhid - count] == 2)
	    {
	      int n = l_numlayers[idx][l_numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  le_dels[i+1](j,0) = le_dels[i+1](j,0)*0.0;
		  derv(j,0) = rec_D(derv(j,0));
		}
	    }
	  else if (l_funclayer[idx][l_numhid-count] == 3)
	    {
	      int n = l_numlayers[idx][l_numhid - i];
	      for (int j = 0; j < n; j++)
		{
		  le_dels[i+1](j,0) = le_dels[i+1](j,0)*tanh_dd(derv(j,0));
		  derv(j,0) = tanh_d(derv(j,0));
		}
	    }
	   temp = temp%(derv%derv);
	   ld_dels[idx].push_back(temp+ le_dels[i+1]);
	}
      else
	{
	  derv = ones<mat>(l_numlayers[idx][0],1);
	  temp = temp%(derv%derv);
	  ld_dels[idx].push_back(temp);
	}
      count++;
    }
  for (int i = 0; i < l_numhid + 1; i++)
    {
      ld_grads[idx].push_back(ld_dels[idx][i]*((l_activ[idx][l_numhid - i]%l_activ[idx][l_numhid - i]).t()));
    }
  return;
}



void NNet::OBD_init(void)
{
  if (!ld_tdels.empty())
    {
      ld_tdels.clear();
    }
  for(int i = 0; i < l_train; i++)
    {
      ld_tdels.push_back(zeros<mat>(l_numx,1));
    }
}



void NNet::l_optimalBD(int pos)
{
  int l_numhid = l_numhids[pos];
  if (!ld_tgrads[pos].empty())
    {
      ld_tgrads[pos].clear();
    }
  for(int j = 0; j < l_numhid + 1; j++)
    {
      int rows = l_numlayers[pos][j+1];
      int cols = l_numlayers[pos][j];
      ld_tgrads[pos].push_back(zeros<mat>(rows,cols));
    }
  int l_count = 0;
  for (int i = 0; i < l_train; i++)
    {
      if (qmat == 1)
	{
	  if (Q_mat[i][pos] == 1)
	    {
	      ld_backprop(l_xvals[i],l_yvals[pos][i],pos);
	      l_count++;
	      reverse(ld_grads[pos].begin(),ld_grads[pos].end());
	      for(int j = 0; j < l_numhid + 1; j++)
		{
		  ld_tgrads[pos][j] = ld_tgrads[pos][j] + ld_grads[pos][j];
		}
	      for (int j = 0; j < l_numx; j++)
		{
		  ld_tdels[i](j,0) = ld_tdels[i](j,0) + ld_dels[pos][l_numhid + 1](j,0);
		}
	    }
	}
      else
	{
	  ld_backprop(l_xvals[i],l_yvals[pos][i],pos);
	  l_count++;
	  reverse(ld_grads[pos].begin(),ld_grads[pos].end());
	  for(int j = 0; j < l_numhid + 1; j++)
	    {
	      ld_tgrads[pos][j] = ld_tgrads[pos][j] + ld_grads[pos][j];
	    }
	  for (int j = 0; j < l_numx; j++)
	    {
	      ld_tdels[i](j,0) = ld_tdels[i](j,0) + ld_dels[pos][l_numhid + 1](j,0);
	    }
	}
    }
  if (!l_saliencies[pos].empty())
    {
      l_checksals[pos] = l_saliencies[pos];
      l_saliencies[pos].clear();
    }
  for(int i = 0; i < l_numhid + 1; i++)
    {
      l_saliencies[pos].push_back(0.5*((l_params[pos][i]%l_params[pos][i])%ld_tgrads[pos][i]));
    }
  double min_s = 1000.0 + l_saliencies[0][0](0,0);
  for(int i = 0; i < l_numhid + 1; i++)
    {
      int rows = l_saliencies[pos][i].n_rows;
      int cols = l_saliencies[pos][i].n_cols;
      for(int rw = 0; rw < rows; rw++)
	{
	  for(int cl = 0; cl < cols; cl++)
	    {
	      if (l_checksals[pos].empty())
		{
		  if (l_saliencies[pos][i](rw,cl) < min_s)
		    {
		      min_s = l_saliencies[pos][i](rw,cl);
		    }
		  else
		    {
		      continue;
		    }
		}
	      else
		{
		  if ((l_saliencies[pos][i](rw,cl) < min_s) && (l_checksals[pos][i](rw,cl) != 0))
		    {
		      min_s = l_saliencies[pos][i](rw,cl);
		    }
		  else
		    {
		      continue;
		    }
		}
	    }
	}
    }
  for(int i = 0; i < l_numhid + 1; i++)
    {
      int rows = l_saliencies[pos][i].n_rows;
      int cols = l_saliencies[pos][i].n_cols;
      double s_tol = min_s/2.0;
      for(int rw = 0; rw < rows; rw++)
	{
	  for(int cl = 0; cl < cols; cl++)
	    {
	      if (abs(l_saliencies[pos][i](rw,cl) - min_s) < s_tol)
		{
		  l_saliencies[pos][i](rw,cl) = 0.0;
		}
	      else
		{
		  l_saliencies[pos][i](rw,cl) = 1.0*l_checksals[pos][i](rw,cl);
		}
	    }
	}
    }
}


void NNet::ls_optimalBD(void)
{
  if (!ls_saliencies.empty())
    {
      ls_saliencies.clear();
    }
  vector<double> l_mins;
  for (int i = 0; i < l_train; i++)
    {
      ls_saliencies.push_back((0.5/(double)numfiles)*ld_tdels[i]%(l_xvals[i]%l_xvals[i]));
      l_mins.push_back(1000.0);
    }
  for (int i = 0; i < l_train; i++)
    {
      for(int j = 0; j < l_numlatent; j++)
	{
	  if (ls_saliencies[i](j,0) < l_mins[i])
	    {
	      l_mins[i] = ls_saliencies[i](j,0);
	    }
	  else
	    {
	      continue;
	    }
	}
    }
  vector<double> s_tol;
  for(int i = 0; i < l_train; i++)
    {
      s_tol.push_back(l_mins[i]/10.0);
      for(int j = 0; j < l_numx; j++)
	{
	  if (j < l_numlatent)
	    {
	      if (abs(ls_saliencies[i](j,0) - l_mins[i]) < s_tol[i])
		{
		  ls_saliencies[i](j,0) = 0.0;
		}
	      else
		{
		  ls_saliencies[i](j,0) = 1.0;
		}
	    }
	  else
	    {
	      ls_saliencies[i](j,0) = 1.0;
	    }
	}
    }
}


//With optimal brain damage
void NNet::ld_trainrprop(int numlatent, double tmax, int mode, double tol)
{
  if (l_yvals.empty())
    {
      cout<<"Please load the files first!"<<endl;
      return;
    }
  for (int fl = 0; fl < numfiles; fl++)
    {
      int l_numhid = l_numhids[fl];
      for(int q = 0; q < l_numhid + 1; q++)
	{
	  if (!l_checkgrads[fl].empty())
	    {
	      l_checkgrads[fl].clear();
	      l_checkdels[fl].clear();
	    }
	}
    }
  vector<mat> tr;
  ld_dels.push_back(tr);
  for (int i = 0; i < numfiles; i++)
    {
      ld_grads.push_back(tr);
      ld_dels.push_back(tr);
      ld_tgrads.push_back(tr);
      l_saliencies.push_back(tr);
      l_checksals.push_back(tr);
    }
  int trainmode = mode;
  int rprop = 0;
  vector<thread> l_bpthreads;
  double rmax = tmax;
  int lat_rprop = 0;
  if ((trainmode != 0) && (trainmode != 1))
    {
      cout<<"Training mode can only be 0 or 1"<<endl;
      return;
    }
  if ((numlatent >= 0) && (l_trained == 0))
    {
      l_numlatent = numlatent;
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
      lrates.push_back(rate);
    }
  int tolcheck;
  if (epoch < 0)
    {
      if (tol < 0)
	{
	  cout<<"Please enter a tolerence value!"<<endl;
	  for (int i = 0; i < numfiles; i++)
	    {
	      l_params[i].clear();
	      l_tgrads[i].clear();
	      l_bias[i].clear();
	      l_tdels[i].clear();
	    }
	  return;
	}
      tolcheck = 1;
      epoch = 10;
    }
  else
    {
      tolcheck = 0;
    }
  if (gradd == 0)
    {
      for(int obd = 0; obd < 3; obd++)
	{
	  if (obd == 0)
	    {
	      for(int fl = 0; fl < numfiles; fl++)
		{
		  int l_numhid = l_numhids[fl];
		  for(int j = 0; j < l_numhid + 1; j++)
		    {
		      int rows = l_numlayers[fl][j+1];
		      int cols = l_numlayers[fl][j];
		      l_saliencies[fl].push_back(ones<mat>(rows,cols));
		    }
		}
	    }
	  int c_epoch = 0;
	  for (int k = 0; k < epoch; k++)
	    {
	      if (k == 0 && (trainmode == 1))
		{
		  cout<<"Initial error"<<endl;
		  l_testall();
		  cout<<endl;
		}
	      if (tolcheck == 1)
		{
		  k = -2;
		}
	      for (int i = 0; i < l_train; i++)
		{
		  int threadcount = 0;
		  if (!l_bpthreads.empty())
		    {
		      l_bpthreads.clear();
		    }
		  if(obd == 0)
		    {
		      ls_saliencies.push_back(ones<mat>(l_numx,1));
		    }
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
		  mat lat_grads = zeros<mat>(numlatent,1);
		  for (int q = 0; q < numfiles; q++)
		    {
		      if(qmat == 1)
			{
			  if (Q_mat[i][q] == 1)
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
		      for(int j = 0; j < numlatent; j++)
			{
			  if (lat_checkgrads(j,0)*lat_grads(j,0) > 0)
			    {
			      if (lat_rprop == 1)
				{
				  double sign = copysign(1,lat_grads(j,0));
				  lat_grads(j,0) = sign*1.2*0.1;
				  lat_checkgrads(j,0) = lat_grads(j,0);
				}
			      else
				{
				  double sign = copysign(1,lat_grads(j,0));
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
				  double sign = copysign(1,lat_grads(j,0));
				  lat_grads(j,0) = sign*0.5*0.1;
				  lat_checkgrads(j,0) = lat_grads(j,0);
				}
			      else
				{
				  double sign = copysign(1,lat_grads(j,0));
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
				  double sign = copysign(1,lat_grads(j,0));
				  lat_grads(j,0) =  sign*0.1;
				}
			      else
				{
				  double sign = copysign(1,lat_grads(j,0));
				  lat_grads(j,0) =  sign*abs(lat_checkgrads(j,0));
				}
			    }
			}
		    }
		  if (lat_rprop == 0)
		    {
		      for(int j = 0; j < numlatent; j++)
			{
			  l_xvals[i](j,0) = l_xvals[i](j,0) - (0.00001)*lat_grads(j,0);
			  //l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
			}
		      l_xvals[i] = l_xvals[i]%ls_saliencies[i];
		    }
		  else
		    {
		      for(int j = 0; j < numlatent; j++)
			{
			  l_xvals[i](j,0) = l_xvals[i](j,0) - lat_grads(j,0);
			  //l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
			}
		      l_xvals[i] = l_xvals[i]%ls_saliencies[i];
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
	      l_update(rprop,rmax);
	      for(int t = 0; t < numfiles; t++)
		{
		  int lnumhid = l_numhids[t];
		  for (int l = 0; l < lnumhid + 1; l++)
		    {
		      if (rprop == 0)
			{
			  l_params[t][l] = l_params[t][l] - 0.00000001*l_tgrads[t][l] - 0.00001*l_params[t][l];
			  l_params[t][l] = l_params[t][l]%l_saliencies[t][l];
			  l_bias[t][l] = l_bias[t][l] - 0.00000001*l_tdels[t][l+1];
			  l_tgrads[t][l].fill(0);
			  l_tdels[t][l+1].fill(0);
			}
		      else
			{
			  l_params[t][l] = l_params[t][l] - l_tgrads[t][l];
			  l_params[t][l] = l_params[t][l]%l_saliencies[t][l];
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
	      if (tolcheck == 0)
		{
		  if (trainmode == 0)
		    {
		      double pc = ((double)k/(double)epoch)*100;
		      cout<<setprecision(5);
		      cout<<"\r"<<pc<<"%"<<flush;
		    }
		  else if (trainmode == 1)
		    {
		      cout<<setprecision(5);
		      cout<<((double)k/(double)epoch)*100<<"%"<<endl;
		      l_testall();
		      cout<<endl;
		    }
		}
	      else
		{
		  if (trainmode == 0)
		    {
		      cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		      l_testall(1);
		      if (l_error <= tol)
			{
			  cout<<"Convergence reached!"<<endl;
			  return;
			}
		      c_epoch++;
		    }
		  else if (trainmode == 1)
		    {
		      cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		      l_testall();
		      if (l_error <= tol)
			{
			  cout<<"Convergence reached!"<<endl;
			  return;
			}
		      c_epoch++;
		      cout<<endl;
		    }
		}
	    }
	  OBD_init();
	  for(int fl = 0; fl < numfiles; fl++)
	    {
	      l_optimalBD(fl);
	    }
	  ls_optimalBD();
	}
    }
  else if (gradd == 1)
    {
      int c_epoch = 0;
      cout<<"Initializing Stochastic Gradient Descent"<<endl;
      double pc = 0;
      vector<int> idxs;
      for (int i = 0; i < l_train; i++)
	{
	  idxs.push_back(i);
	  ls_saliencies.push_back(ones<mat>(l_numx,1));
	}
      random_shuffle(idxs.begin(),idxs.end());
      for(int obd = 0; obd < 3; obd++)
	{
	  if (obd == 0)
	    {
	      for(int fl = 0; fl < numfiles; fl++)
		{
		  int l_numhid = l_numhids[fl];
		  for(int j = 0; j < l_numhid + 1; j++)
		    {
		      int rows = l_numlayers[fl][j+1];
		      int cols = l_numlayers[fl][j];
		      l_saliencies[fl].push_back(ones<mat>(rows,cols));
		    }
		}
	    }
	  for (int i = 0; i < epoch; i++)
	    {
	      if (i == 0 && (trainmode == 1))
		{
		  cout<<"Initial error"<<endl;
		  l_testall();
		  cout<<endl;
		}
	      if (tolcheck == 1)
		{
		  i = -2;
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
			  for(int j = 0; j < numlatent; j++)
			    {
			      if (lat_checkgrads(j,0)*lat_grads(j,0) > 0)
				{
				  if (lat_rprop == 1)
				    {
				      double sign = copysign(1,lat_grads(j,0));
				      lat_grads(j,0) = sign*1.2*0.1;
				      lat_checkgrads(j,0) = lat_grads(j,0);
				    }
				  else
				    {
				      double sign = copysign(1,lat_grads(j,0));
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
				      double sign = copysign(1,lat_grads(j,0));
				      lat_grads(j,0) = sign*0.5*0.1;
				      lat_checkgrads(j,0) = lat_grads(j,0);
				    }
				  else
				    {
				      double sign = copysign(1,lat_grads(j,0));
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
				      double sign = copysign(1,lat_grads(j,0));
				      lat_grads(j,0) =  sign*0.1;
				    }
				  else
				    {
				      double sign = copysign(1,lat_grads(j,0));
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
			  l_xvals[idxs.at(k)] = l_xvals[idxs.at(k)]%ls_saliencies[idxs.at(k)];
			}
		      else
			{
			  for(int j = 0; j < numlatent; j++)
			    {
			      l_xvals[idxs.at(k)](j,0) = l_xvals[idxs.at(k)](j,0) - lat_grads(j,0);
			      //l_tdels[q][0].fill(0); //Doing this is textbook but commenting this out just works much much better
			    }
			  l_xvals[idxs.at(k)] = l_xvals[idxs.at(k)]%ls_saliencies[idxs.at(k)];
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
		  l_rmsprop(rprop);
		  for(int q = 0; q < numfiles; q++)
		    {
		      int lnumhid = l_numhids[q];
		      for (int j = 0; j < lnumhid + 1; j++)
			{
			  //the below only applies if regression is going.....
			  if (rprop == 0)
			    {
			      l_params[q][j] = l_params[q][j] - (0.00001/(double)l_train)*l_tgrads[q][j] - 0.0001*l_params[q][j];
			      l_params[q][j] = l_params[q][j]%l_saliencies[q][j];
			      l_bias[q][j] = l_bias[q][j] - (0.00001/(double)l_train)*l_tdels[q][j+1];
			      l_tgrads[q][j].fill(0.0);
			      l_tdels[q][j+1].fill(0.0);
			    }
			  else
			    {
			      l_params[q][j] = l_params[q][j] - 0.001*(l_tgrads[q][j]/l_checkgrads[q][j]);
			      l_params[q][j] = l_params[q][j]%l_saliencies[q][j];
			      l_bias[q][j] = l_bias[q][j] - 0.001*(l_tdels[q][j+1]/l_checkdels[q][j]);
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
		}
	      if (tolcheck == 0)
		{
		  if (trainmode == 0)
		    {
		      pc = ((double)i/(double)epoch)*100;
		      cout<<setprecision(5);
		      cout<<"\r"<<pc<<"%"<<flush;
		    }
		  else if (trainmode == 1)
		    {
		      cout<<setprecision(5);
		      cout<<((double)i/(double)epoch)*100<<"%"<<endl;
		      l_testall();
		      cout<<endl;
		    }
		}
	      else
		{
		  if (trainmode == 0)
		    {
		      cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		      l_testall(1);
		      if (l_error <= tol)
			{
			  cout<<"Convergence reached!"<<endl;
		      return;
			}
		      c_epoch++;
		    }
		  else if (trainmode == 1)
		    {
		      cout<<"\r"<<"Epoch No. "<<c_epoch<<flush;
		      l_testall();
		      if (l_error <= tol)
			{
			  cout<<"Convergence reached!"<<endl;
			  return;
			}
		      c_epoch++;
		      cout<<endl;
		    }
		}
	    }
	   OBD_init();
	  for(int fl = 0; fl < numfiles; fl++)
	    {
	      l_optimalBD(fl);
	    }
	  ls_optimalBD();
	}
    }
  l_trained++;
  cout<<endl;
  return;
}
