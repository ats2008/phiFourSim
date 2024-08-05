INCPATH 	= -Iinclude -I/usr/local/cuda/include
OBJ		= obj
SRC		= src

CXX 		= g++
NVCC 		= nvcc

LIBS		= -lm -lcudart
CXXFLAGS 	= -std=c++14 $(INCPATH) -g
NVCCFLAGS	= -std=c++14 $(INCPATH) -g -G
NVCCLIBS	= -lcudart

TARGET		= main.exe
DEPS		= main lattice
CUDEPS		= lattice

OBJ_  := $(DEPS:%=$(OBJ)/%.o)
OBJCU_  := $(CUDEPS:%=$(OBJ)/%cu.o)

all:	$(TARGET) 

$(TARGET) : $(OBJ_) $(OBJCU_)
	$(NVCC)  -o $(TARGET) $(OBJ_) $(OBJCU_)

$(OBJ)/%.o : $(SRC)/%.cpp 
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ -c $^

$(OBJ)/%cu.o : $(SRC)/%.cu 
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBS) -c $^  -o $@


clean :
	@rm obj/*.o
	@rm *.exe
