INCPATH 	= include
OBJ		= obj
SRC		= src

CXX 		= g++
NVCC 		= nvcc

LIBS		= -lm
CXXFLAGS 	= -std=c++14 -I$(INCPATH) 
NVCCFLAGS	= -std=c++14 -I$(INCPATH)
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
