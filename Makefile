CC = g++
CFLAGS = -Wall -Werror -ggdb -std=c++11
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

BOOST_INC = /usr/include
BOOST_LIB = /usr/lib/i386-linux-gnu

TARGET = NNet

$(TARGET).so: $(TARGET).o
	$(CC) -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python-py27 -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so -larmadillo 

$(TARGET).o: $(TARGET).cpp
	$(CC) $(CFLAGS) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cpp -larmadillo


clean:
	rm -r *.py
	rm -r *.o
	rm -r *.so
	rm -r *.pyc
	rm -r *.cxx
	rm -r main
