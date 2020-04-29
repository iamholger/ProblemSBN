# ProblemSBN

# Dependencies
AppEval, SignalGenerator, cppoptlib

# Compilation of library

g++ -O3 -fopenmp -shared -fPIC  -march=native -mtune=native  -DNDEBUG  -std=c++17  -g -IEigen3.3.7 -Icppoptlib/include -IAppEval -ISignalGen -IProblemSBN -Llib -lAppEval -lSignalGenerator -o lib/libProblemSBN.so src/ProblemSBN.cxx
