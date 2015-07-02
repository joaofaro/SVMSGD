CC = g++

# Optimisation level
OPT     = -O3

## final compilation flags
CFLAGS	= $(OPT) -Wall

# OpenCV libraries
PKG_CONFIG_CFLAGS=`pkg-config --cflags opencv`

# OpenCV libraries
PKG_CONFIG_LIB=`pkg-config --libs opencv`

# EXECcutable
EXEC = ./bin/SVMSGD

# Main
MAIN = main

# Objects to be linked with main
OBJS = SvmSgd.o

$(EXEC) : $(MAIN).cpp ${OBJS}
	$(CC) $(CFLAGS) $(PKG_CONFIG_CFLAGS) $(MAIN).cpp ${OBJS} -o $(EXEC) $(PKG_CONFIG_LIB) $(LIB)

SvmSgd.o: SvmSgd.cpp SvmSgd.hpp	
	$(CC) $(CFLAGS) -c SvmSgd.cpp $(PKG_CONFIG_CFLAGS)

clean: 
	rm -f ${OBJS} ${EXEC}
