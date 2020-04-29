#pragma once
#include <random>
#include <Eigen/Eigen>
#include "SignalGenerator.h"

#include "cppoptlib/meta.h"
#include "cppoptlib/boundedproblem.h"

using namespace Eigen;
using namespace cppoptlib;
using namespace SignalGenerator;

namespace ProblemSBN 
{

  inline MatrixXd calcMatrix(MatrixXd const & M, VectorXd const & spec)
  {
     MatrixXd ret(M.cols(), M.cols());
     ret.array()    = M.array()*(spec*spec.transpose()).array();
     return ret;
  }

  inline MatrixXd cholD(MatrixXd const & M, VectorXd const & spec, double tol=1e-7)
  {
     auto in = calcMatrix(M, spec);
     SelfAdjointEigenSolver<MatrixXd> eigensolver(in);
     auto const & EV = eigensolver.eigenvalues();

     for (int i=0;i<EV.rows();++i) 
     {
         if (EV[i]<=0) 
         {
           if (fabs(EV[i]) < tol) for (int a=0; a<in.cols(); ++a) in(a,a) += EV[i];
         }
         if (fabs(EV[i])< tol) for (int a =0; a<in.cols(); a++) in(a,a) += tol;
     }
     LLT<MatrixXd> llt(in);
     return llt.matrixL();
  }


  inline MatrixXd collapseSubchannels(MatrixXd const & EE, MyConfig const & conf)
  {
      MatrixXd  retMat = MatrixXd::Zero(conf.num_bins_detector_block_compressed, conf.num_bins_detector_block_compressed);

      int mrow(0), mcol(0), mrow_out(0), mcol_out(0);
#pragma code_align 32
      for(int ic = 0; ic < conf.num_channels; ic++)
      {
          for(int jc =0; jc < conf.num_channels; jc++)
          {
              for(int m=0; m < conf.num_subchannels[ic]; m++)
              {
                  for(int n=0; n< conf.num_subchannels[jc]; n++)
                  {
                     int a, c;
                     a=mrow + n*conf.num_bins[jc];
                     c=mcol + m*conf.num_bins[ic];
                     retMat.block(mrow_out, mcol_out, conf.num_bins[jc], conf.num_bins[ic]).noalias() += EE.block(a, c, conf.num_bins[jc], conf.num_bins[ic]);
                  }
              }
              mrow     += conf.num_subchannels[jc]*conf.num_bins[jc];
              mrow_out += conf.num_bins[jc];
          } // end of column loop
          mrow      = 0; // as we end this row, reSet row count, but jump down 1 column
          mrow_out  = 0;
          mcol     += conf.num_subchannels[ic]*conf.num_bins[ic];
          mcol_out += conf.num_bins[ic];
      } // end of row loop
      return retMat;
  }

  inline MatrixXd collapseDetectors(MatrixXd const & M, MyConfig const & conf)
  {
      MatrixXd  retMat = MatrixXd::Zero(conf.num_bins_mode_block_compressed, conf.num_bins_mode_block_compressed);
      auto const & nrow = conf.num_bins_detector_block;
      auto const & crow = conf.num_bins_detector_block_compressed;
#pragma omp parallel for
#pragma code_align 32
      for (int m=0; m<conf.num_detectors; m++)
      {
          for (int n=0; n<conf.num_detectors; n++) 
          {
              retMat.block(n*crow, m*crow, crow, crow).noalias() = collapseSubchannels(M.block(n*nrow, m*nrow, nrow, nrow), conf);
          }
      }
      return retMat;
  }

  inline MatrixXd invertMatrix(MatrixXd const & M)
  {
      return M.llt().solve(MatrixXd::Identity(M.rows(), M.rows()));
  }

  inline MatrixXd calcCovarianceMatrix(MatrixXd const & M, VectorXd const & spec)
  {
     MatrixXd ret(M.cols(), M.cols());
     ret.array()    = M.array()*(spec*spec.transpose()).array();
     ret.diagonal() += spec;
     return ret;
  }

  inline MatrixXd updateInvCov(MatrixXd const & covmat, VectorXd const & spec_full, MyConfig const & conf)
  {
      auto const & cov = calcCovarianceMatrix(covmat, spec_full);
      auto const & out = collapseDetectors(cov, conf);
      return invertMatrix(out);
  }

  inline VectorXd collapseVector(VectorXd  const & vin, MyConfig const & conf)
  {
     // All we want is a representation with the subchannels added together
     VectorXd cvec(conf.num_bins_total_compressed);
     cvec.setZero();
     for (int d=0; d<conf.num_detectors;++d)
     {
        size_t offset_in(0), offset_out(0);
        for (int i=0; i<conf.num_channels; i++)
        {
            size_t nbins_chan = conf.num_bins[i];
            for (int j=0; j<conf.num_subchannels[i]; j++)
            {
               size_t first_in   = d*conf.num_bins_detector_block            + offset_in;
               size_t first_out  = d*conf.num_bins_detector_block_compressed + offset_out;
               cvec.segment(first_out, nbins_chan).noalias() += vin.segment(first_in, nbins_chan);
               offset_in +=nbins_chan;
            }
            offset_out += nbins_chan;
        }
     }
     return cvec;
  }

  inline double calcChi(VectorXd const & diff, MatrixXd const & C_inv )
  {
     return diff.transpose() * C_inv * diff;
  }

  inline double calcChi(VectorXd const & data, VectorXd const & prediction, MatrixXd const & C_inv )
  {
     auto const & diff = data-prediction;
     return diff.transpose() * C_inv * diff;
  }

  inline VectorXd poisson_fluctuate(VectorXd const & spec, std::mt19937 & rng)
  {
     VectorXd RDM(spec.rows());
     for (int i=0;i<spec.rows();++i) {
        std::poisson_distribution<int> dist_pois(spec[i]);
        RDM[i] = double(dist_pois(rng));
     }
     return RDM;
  }

  inline VectorXd sample(VectorXd const & spec, MatrixXd const & LMAT, std::mt19937 & rng)
  {

    std::normal_distribution<double> dist_normal(0,1);
    VectorXd RDM(spec.rows());
    for (int i=0;i<spec.rows();++i) RDM[i] = dist_normal(rng);
   
    return LMAT*RDM + spec;
  }

  class ProblemSBN : public BoundedProblem<double>
  {

    public:
      ProblemSBN(
              int dim,
              SignalGenerator::SignalGenerator  sig,
              MatrixXd const & covmat,
              MyConfig const & config
              ) :
          BoundedProblem<double>(dim), _sig(sig), _covmat(covmat), _config(config), _ncalls(0) {}

      double value(const Eigen::VectorXd & x); 
      void setData(VectorXd const & data) {_data=data;}
      int ncalls()  const {return _ncalls;}

    private:
      VectorXd _data;
      SignalGenerator::SignalGenerator _sig;
      MatrixXd const _covmat;
      MyConfig const _config;
      int _ncalls;
  };
  
  class ProblemSBNIT : public BoundedProblem<double>
  {

    public:
      ProblemSBNIT(
              int dim,
              SignalGenerator::SignalGenerator  sig,
              MyConfig const & config
              ) :
          BoundedProblem<double>(dim), _sig(sig), _config(config), _ncalls(0) {}

      double value(const Eigen::VectorXd & x); 
      void setData(VectorXd const & data) {_data=data;}
      void setCOV(MatrixXd const & cov) {_covmat=cov;}
      int ncalls()  const {return _ncalls;}

    private:
      SignalGenerator::SignalGenerator _sig;
      MatrixXd  _covmat;
      VectorXd _data;
      MyConfig const _config;
      int _ncalls;
  };
};
