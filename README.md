# ProblemSBN

# Dependencies
AppEval, SignalGenerator, cppoptlib, HighFive, libhdf5, libmpi, DIY, optim

# Compilation of library

g++ -O3 -fopenmp -shared -fPIC  -march=native -mtune=native  -DNDEBUG  -std=c++17  -g -IEigen3.3.7 -Icppoptlib/include -IAppEval -ISignalGen -IProblemSBN -Llib -lAppEval -lSignalGenerator -o lib/libProblemSBN.so src/ProblemSBN.cxx

# Compilation of example
g++ -O3 -flto -fopenmp  -march=native -mtune=native  -DNDEBUG  -std=c++17  -IEigen3.3.7 -I/usr/include/mpich-x86_64 -Ioptim/include -IAppEval -ISignalGen -IProblemSBN -IHighFive/include -Idiy/include  -Llib -lAppEval -lSignalGenerator -lProblemSBN -L/usr/lib64/mpich/lib -lhdf5 -lmpi -Loptim/local/lib -loptim -lopenblas -o multimiseOPTIM example/multimiseOPTIM.cxx
