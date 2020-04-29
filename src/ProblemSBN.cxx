#include "ProblemSBN.h"
#include <random>

double ProblemSBN::ProblemSBN::value(const Eigen::VectorXd & x) 
{
   _ncalls++;
   Vector3d xeval; xeval << x[0], 1, x[1];
   auto D     = _sig.predict(xeval, 1);
   auto CINV  = updateInvCov(_covmat, D, _config);
   return  calcChi(_data - collapseVector(D, _config), CINV);
}

double ProblemSBN::ProblemSBNIT::value(const Eigen::VectorXd & x)
{
   _ncalls++;
   Vector3d xeval; xeval << x[0], 1, x[1];
   auto D     = _sig.predict(xeval, 1);
   return  calcChi(_data - collapseVector(D, _config), _covmat);
}
