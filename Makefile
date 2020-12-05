INCPATH 	= include
OBJ		= obj
SRC		= src

CXX 		= g++

LIBS		= -lm
CXXFLAGS 	= -Wall -std=c++11 -I$(INCPATH) 

TARGET		= main.exe
DEPS		= main lattice



OBJ_  := $(DEPS:%=$(OBJ)/%.o)

all:	$(TARGET) 

$(TARGET) : $(OBJ_)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ_)

$(OBJ)/%.o : $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ -c $^


clean :
	@rm obj/*.o
	@rm *.exe
	@rm *~
