#include <random>
#include <fstream>

#include <diy/master.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/merge.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/mpi.hpp>
#include <diy/serialization.hpp>
#include <diy/partners/broadcast.hpp>
#include <diy/reduce-operations.hpp>

#include <Eigen/Eigen>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>


#include "AppEval.h"
#include "SignalGenerator.h"
#include "ProblemSBN.h"
#include "optim.hpp"

#include "opts.h"

using namespace Eigen;
using namespace Apprentice;
using namespace SignalGenerator;
using namespace ProblemSBN;

using namespace std;

std::tuple<std::vector<std::vector<size_t> >, size_t, size_t>  mkRankWork(size_t nPoints, size_t nUniverses, int thisRank, int worldSize) {
    size_t _S(nPoints*nUniverses);
    std::vector<size_t> _L;
    size_t maxwork = size_t(ceil(_S/worldSize));
    for (size_t r=0; r <             _S%worldSize; ++r) _L.push_back(maxwork + 1);
    for (size_t r=0; r < worldSize - _S%worldSize; ++r) _L.push_back(maxwork    );

    std::vector<size_t> _bp, _bu;
    _bp.push_back(0);
    _bu.push_back(0);

    size_t _n(0), _temp(0);
    for (size_t i=0; i<nPoints;++i) {
       for (size_t j=0; j<nUniverses;++j) {
          if (_temp == _L[_n]) {
             _bp.push_back(i);
             _bu.push_back(j);
             _temp = 0;
             _n+=1;
          }
          _temp+=1;
       }
    }
    _bp.push_back(nPoints-1);
    _bu.push_back(nUniverses);
    
    std::vector<size_t> rankworkbl;
    //std::cerr << thisRank << " : " << _bu.size() -1 << " sets of work " << _bp.size() -1 << "\n";
    rankworkbl.push_back(_bp[thisRank]);
    rankworkbl.push_back(_bu[thisRank]);
    rankworkbl.push_back(_bp[thisRank+1]);
    rankworkbl.push_back(_bu[thisRank+1]);

    size_t pStart = rankworkbl[0];
    size_t uStart = rankworkbl[1];
    size_t pLast  = rankworkbl[2];
    size_t uLast  = rankworkbl[3];

    size_t i_begin = pStart * nUniverses + uStart;
    size_t i_end   = pLast  * nUniverses + uLast;

    //std::cerr << "[" << i << "] " << pStart << " " <<  uStart << " " << pLast << " " << uLast << " starts: " << i_begin << " " << i_end << " " <<i_end-i_begin << "\n";
    size_t lenDS = i_end - i_begin;
    size_t d_bgn = i_begin;


    std::vector<std::vector<size_t> > RW;
    if (pStart == pLast) RW.push_back({pStart, uStart, uLast});
    else {
      RW.push_back({pStart, uStart, nUniverses});
       for (size_t _p = pStart+1; _p<pLast;++_p) {
          RW.push_back({_p, 0, nUniverses});
       }
       if (uLast>0) RW.push_back({pLast, 0, uLast});

    }

    return std::make_tuple(RW, i_begin, lenDS);
}


void loadData(const char* fname, std::string what, std::vector<double> & v_buffer, int & n_rows, int & n_cols) {
    H5Easy::File file(fname, H5Easy::File::ReadOnly);
    MatrixXd _mat      = H5Easy::load<MatrixXd>(file, what);
    n_rows = _mat.rows();
    n_cols = _mat.cols();
    v_buffer = std::vector<double>(_mat.data(), _mat.data() + _mat.rows() * _mat.cols());
}

void loadData(const char* fname, std::string what, std::vector<int> & v_buffer, int & n_rows, int & n_cols) {
    H5Easy::File file(fname, H5Easy::File::ReadOnly);
    MatrixXi _mat      = H5Easy::load<MatrixXi>(file, what);
    n_rows = _mat.rows();
    n_cols = _mat.cols();
    v_buffer = std::vector<int>(_mat.data(), _mat.data() + _mat.rows() * _mat.cols());
}

MatrixXd bcMatrixXd(diy::mpi::communicator world, std::vector<double>  v_buffer, int  n_rows, int  n_cols) {
    diy::mpi::broadcast(world, v_buffer, 0);
    diy::mpi::broadcast(world, n_rows,   0);
    diy::mpi::broadcast(world, n_cols,   0);

    Map<MatrixXd> mat(v_buffer.data(), n_rows, n_cols);
    return mat;
}

MatrixXi bcMatrixXi(diy::mpi::communicator world, std::vector<int>  v_buffer, int  n_rows, int  n_cols) {
    diy::mpi::broadcast(world, v_buffer, 0);
    diy::mpi::broadcast(world, n_rows,   0);
    diy::mpi::broadcast(world, n_cols,   0);

    Map<MatrixXi> mat(v_buffer.data(), n_rows, n_cols);
    return mat;
}

AppEval mkAppEval(diy::mpi::communicator world, const char* fname) {
    std::vector<double> v_buff;
    std::vector<int> v_buff_i;
    int ncols, nrows;
    if (world.rank()==0) loadData(fname, "pcoeff",      v_buff,   nrows, ncols);
    MatrixXd PC = bcMatrixXd(world, v_buff, nrows, ncols);          
    if (world.rank()==0) loadData(fname, "qcoeff",      v_buff,   nrows, ncols);
    MatrixXd QC = bcMatrixXd(world, v_buff, nrows, ncols);          
    if (world.rank()==0) loadData(fname, "a",           v_buff,   nrows, ncols);
    VectorXd a = bcMatrixXd(world, v_buff, nrows, ncols);           
    if (world.rank()==0) loadData(fname, "xmin",        v_buff,   nrows, ncols);
    VectorXd xmin = bcMatrixXd(world, v_buff, nrows, ncols);        
    if (world.rank()==0) loadData(fname, "scaleterm",   v_buff,   nrows, ncols);
    VectorXd scaleterm = bcMatrixXd(world, v_buff, nrows, ncols);
    if (world.rank()==0) loadData(fname, "structure",   v_buff_i, nrows, ncols);
    MatrixXi structure = bcMatrixXi(world, v_buff_i, nrows, ncols);

    return AppEval(PC, QC, structure, a, xmin, scaleterm);
}

MyConfig mkConfig() {
    std::vector<int> num_bins = {19};
    std::vector<int> num_subchannels = {2};
    std::vector<std::vector<int> > subchannel_osc_patterns = { {22, 0} };
    return MyConfig{3, 38, 1, 19, 57, 114, num_bins, num_subchannels, subchannel_osc_patterns};
}

typedef diy::DiscreteBounds Bounds;



struct FitBlock {
    static void*    create()            { return new FitBlock; }
    static void     destroy(void* b)    { delete static_cast<FitBlock*>(b); }

    std::vector<double> last_chi_min, delta_chi;
    std::vector<int> n_iter, i_grid, i_univ;
};

struct MyGrid {
  ArrayXXd points;
  Array2d MIN;
  Array2d MAX;
};

MyGrid mkGrid(int nx, double xmin, double xmax, int ny, double ymin, double ymax) {

  int N=nx*ny;
  ArrayXXd ret(N,2);
  auto M = ArrayXd::LinSpaced(nx, xmin, xmax);
  auto P = ArrayXd::LinSpaced(ny, ymin, ymax);
  for (auto m=0; m<nx;++m) {
    for (auto p=0;p<ny;++p) {
       int idx = m*nx +p;
       ret(idx,0) = M[m];
       ret(idx,1) = P[p];
    }
  }
  Array2d minbound;minbound<<xmin,ymin;
  Array2d maxbound;maxbound<<xmax,ymax;
  return {ret, minbound, maxbound};
}

void writeGrid(HighFive::File* file, ArrayXXd  const & coords) {
    size_t npoints(coords.rows()), dim(coords.cols());
    file->createDataSet<double>("grid", HighFive::DataSpace( {dim, npoints} ));
    diy::mpi::communicator world;
    if (world.rank()==0) {
       HighFive::DataSet d_grid = file->getDataSet("grid");
       d_grid.select({0, 0}, {dim,npoints}).write(coords.data());
    }
}

void createDataSets(HighFive::File* file, size_t nPoints, size_t nUniverses) {
   file->createDataSet<size_t>("point",    HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<size_t>("universe", HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<double>("xmin",     HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<double>("ymin",     HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<double>("thischi2",     HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<double>("minchi2",     HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<double>("minchi2pf",     HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<double>("xtrue",    HighFive::DataSpace( {1, nPoints*nUniverses} ));
   file->createDataSet<double>("ytrue",    HighFive::DataSpace( {1, nPoints*nUniverses} ));
}

struct MyFitData
{
    VectorXd data;
    MatrixXd covmat;
    MatrixXd invcovbg;
    SignalGenerator::SignalGenerator sig;
    MyConfig config;
};

double prefit_fn(const arma::vec& vals_inp, arma::vec* grad_out, void * opt_data) {
  MyFitData* mfd  = reinterpret_cast<MyFitData*>(opt_data);
  Vector3d xeval; xeval << vals_inp[0], 1, vals_inp[1];
  auto D     = mfd->sig.predict(xeval, 1);
  double obj =  calcChi(mfd->data - collapseVector(D, mfd->config), mfd->invcovbg);

  if (grad_out) {
    const double h=1e-5;
    double g[2];

#pragma omp parallel for
    for (unsigned int i=0;i<2;++i) {
      Vector3d xgrad0; xgrad0 << vals_inp[0]+h, 1, vals_inp[1];
      auto D0     = mfd->sig.predict(xgrad0, 1);
      double obj0 =  calcChi(mfd->data - collapseVector(D0, mfd->config), mfd->invcovbg);
      g[i] = (obj0-obj)/h;
    }
    *grad_out = {g[0], g[1]};
  }
  return obj;
}

double fit_fn(const arma::vec& vals_inp, arma::vec* grad_out, void * opt_data) {
  MyFitData* mfd  = reinterpret_cast<MyFitData*>(opt_data);
  Vector3d xeval; xeval << vals_inp[0], 1, vals_inp[1];
  auto D     = mfd->sig.predict(xeval, 1);
  auto CINV  = updateInvCov(mfd->covmat, D, mfd->config);
  double obj =  calcChi(mfd->data - collapseVector(D, mfd->config), CINV);

  if (grad_out) {
    const double h=1e-5;
    double g[2];

#pragma omp parallel for
    for (unsigned int i=0;i<2;++i) {
      Vector3d xgrad0; xgrad0 << vals_inp[0]+h, 1, vals_inp[1];
      auto D0     = mfd->sig.predict(xgrad0, 1);
      auto CINV0  = updateInvCov(mfd->covmat, D0, mfd->config);
      double obj0 =  calcChi(mfd->data - collapseVector(D0, mfd->config), CINV0);
      g[i] = (obj0-obj)/h;
    }
    *grad_out = {g[0], g[1]};
  }
  return obj;
}

inline Array2d unscale(Array2d const & X, Array2d const & xmin, Array2d const & scale) {
  return xmin + (X + Array2d::Ones()) / scale;
}

ArrayXXd mkPoints(Array2d const & p0, Array2d const & pmin, Array2d const & pmax, int nPoints, double dist) {
    Array2d scale = 2./(pmax-pmin);
    int good(0);
    ArrayXXd ret(nPoints, 2);
    double d2 = dist*dist;
    while(good<nPoints) {
        Array2d tt = unscale(Array2d::Random(), pmin, scale);

        double l2 = (p0(0) - tt(0)) * (p0(0) - tt(0)) + (p0(1) - tt(1)) * (p0(1) - tt(1));
        if (l2 < d2) {
            ret.row(good) = tt;
            good++;
        }
    }
    return ret;
}

std::tuple<double, int> universeChi2(MyFitData & mfd, MyGrid const & MG) {
   double chimin=std::numeric_limits<double>::infinity();
  
   
   int bestP(0);
   for (size_t i=0; i<MG.points.rows(); ++i) {

      double chi = prefit_fn({MG.points(i,0), MG.points(i,1)}, nullptr, &mfd);
       if (chi<chimin) {
          chimin = chi;
          bestP=i;
       }
   }
   return {chimin, bestP};
}

void runMin(FitBlock* b, diy::Master::ProxyWithLink const& cp, SignalGenerator::SignalGenerator  sig, const MatrixXd & covmat, const MatrixXd & invcovbg, MyConfig const & config, MyGrid const & MG, size_t nUniverses, HighFive::File* file) {

  
  optim::algo_settings_t settings;
  settings.err_tol = 1e-6;
  settings.iter_max=100;
  settings.vals_bound=true;
  settings.lower_bounds = {MG.MIN(0), MG.MIN(1)};
  settings.upper_bounds = {MG.MAX(0), MG.MAX(1)};


  Array2d pmin; pmin <<settings.lower_bounds[0], settings.lower_bounds[1];
  Array2d pmax; pmax <<settings.upper_bounds[0], settings.upper_bounds[1];

  std::mt19937 rng(cp.gid());
  diy::mpi::communicator world;
  const auto  [rankwork, ds_bgn, _ds_len] = mkRankWork(MG.points.rows(), nUniverses, world.rank(),  world.size());
  const size_t ds_len(_ds_len);
  ArrayXXi res_meta(ds_len, 2);
  ArrayXXd res_val( ds_len, 7);

  bool success;
  arma::vec  xwork = arma::vec(2);
  arma::vec _xwork = arma::vec(2);
  size_t idx(0);

  double minValue(1e40);

  for (auto r : rankwork) {
     size_t i_grid = r[0];

     Array3d xeval; xeval << MG.points(i_grid, 0), 1, MG.points(i_grid, 1);
     auto const D  = sig.predict(xeval, 1);
     Eigen::MatrixXd const & LMAT = cholD(covmat, D);
     
     auto const Dc = collapseVector(D, config);
     MyFitData mfd = {Dc, covmat, invcovbg, sig, config};


     for (size_t uu=r[1]; uu<r[2];++uu) {
        VectorXd const fake_data  = ProblemSBN::poisson_fluctuate(sample(D, LMAT, rng), rng);
        VectorXd const fake_dataC = collapseVector(fake_data, config); 
        mfd.data = std::move(fake_dataC);

        xwork = { MG.points(i_grid,0),  MG.points(i_grid,1) };
        
        Vector2d ewk; ewk<< xwork[0], xwork[1];
        auto VIC = mkPoints(ewk, pmin, pmax, 10, 0.1); 

        optim::lbfgs(xwork, prefit_fn, &mfd, settings);
        double minValuePrefit = prefit_fn(xwork, nullptr, &mfd);
        //std::cerr << "[" << world.rank() <<"] "  << "WINNERpre: " << minValuePrefit << " at " << xwork[0] << " " << xwork[1] << "(" << G(i_grid,0) << ", "<<  G(i_grid,1) << ")" << "\n";
        //success = optim::cg(xwork, fit_fn, &mfd, settings);

        for (int ii=0;ii<VIC.rows();++ii) {
           _xwork[0] = MG.points(ii, 0);
           _xwork[1] = MG.points(ii, 1);
           optim::lbfgs(_xwork, prefit_fn, &mfd, settings);
           double _minValue = prefit_fn(_xwork, nullptr, &mfd);
        
           if (_minValue < minValuePrefit) {
              minValuePrefit = _minValue;
              xwork = _xwork;
           }
        }

        //std::cerr << "[" << world.rank() <<"] "  << "WINNERpre MS: " << minValuePrefit << " at " << xwork[0] << " " << xwork[1] << "(" << G(i_grid,0) << ", "<<  G(i_grid,1) << ")" << "\n";
        Vector2d ewk2; ewk2<< xwork[0], xwork[1];
        auto VIC2 = mkPoints(ewk, pmin, pmax, 5, 0.1); 


        optim::lbfgs(xwork, fit_fn, &mfd, settings);
        double minValue = fit_fn(xwork, nullptr, &mfd);
        //std::cerr << "[" << world.rank() <<"] "  << "WINNER:   " << minValue << " at " << xwork[0] << " " << xwork[1] << "(" << G(i_grid,0) << ", "<<  G(i_grid,1) << ")" << "\n";
        //  Multistart
        for (int ii=0;ii<VIC2.rows();++ii) {
           _xwork[0] = MG.points(ii, 0);
           _xwork[1] = MG.points(ii, 1);
           success = optim::lbfgs(_xwork, fit_fn, &mfd, settings);
           double _minValue = fit_fn(_xwork, nullptr, &mfd);

           if (_minValue < minValue) {
              minValue = _minValue;
              xwork = _xwork;
           }
        }

        xwork = { MG.points(i_grid,0),  MG.points(i_grid,1) };
        //std::cerr << "WINNERMS: " << minValue << " at " << xwork[0] << " " << xwork[1] << "\n\n";


        Vector3d xmineval; xmineval << xwork[0], 1, xwork[1];
        VectorXd sigmin = sig.predict(xmineval, 1);
        MatrixXd invcovatmin  = updateInvCov(covmat, sigmin, config);
        double this_chi = calcChi(fake_dataC, Dc, invcovatmin);

        _xwork = { MG.points(i_grid,0),  MG.points(i_grid,1) };
        double thus_chi = fit_fn(_xwork, nullptr, &mfd);

        //float this_chi = this->CalcChi(fake_data, true_spec->collapsed_vector,inverse_current_collapsed_covariance_matrix);


        res_meta(idx,0) = r[0];
        res_meta(idx,1) = uu;
        res_val(idx,0) = xwork[0];
        res_val(idx,1) = xwork[1];
        res_val(idx,2) = this_chi;
        res_val(idx,3) = minValue;
        res_val(idx,4) = thus_chi;
        //res_val(idx,3) = this_chi - globmin;
        res_val(idx,5) = MG.points(i_grid, 0);
        res_val(idx,6) = MG.points(i_grid, 1);
        idx++;
       }
       if (world.rank()==0 && idx%100==0) fmt::print(stderr, "[{}] progress: {}/{}\n",cp.gid(), idx, ds_len);
    }

    HighFive::DataSet d_point    = file->getDataSet("point");
    HighFive::DataSet d_universe = file->getDataSet("universe");
    d_point   .select({0, ds_bgn}, {1,idx}).write(res_meta.col(0).data());
    d_universe.select({0, ds_bgn}, {1,idx}).write(res_meta.col(1).data());

    HighFive::DataSet d_xmin     = file->getDataSet("xmin");
    HighFive::DataSet d_ymin     = file->getDataSet("ymin");
    HighFive::DataSet d_chi2     = file->getDataSet("thischi2");
    HighFive::DataSet d_minchi2  = file->getDataSet("minchi2");
    HighFive::DataSet d_minchi2pf= file->getDataSet("minchi2pf");
    HighFive::DataSet d_xtrue    = file->getDataSet("xtrue");
    HighFive::DataSet d_ytrue    = file->getDataSet("ytrue");

    
    d_xmin     .select({0, ds_bgn}, {1,idx}).write(res_val.col(0).data());
    d_ymin     .select({0, ds_bgn}, {1,idx}).write(res_val.col(1).data());
    d_chi2     .select({0, ds_bgn}, {1,idx}).write(res_val.col(2).data());
    d_minchi2  .select({0, ds_bgn}, {1,idx}).write(res_val.col(3).data());
    d_minchi2pf.select({0, ds_bgn}, {1,idx}).write(res_val.col(4).data());

    d_xtrue    .select({0, ds_bgn}, {1,idx}).write(res_val.col(5).data());
    d_ytrue    .select({0, ds_bgn}, {1,idx}).write(res_val.col(6).data());
}

void runGrid(FitBlock* b, diy::Master::ProxyWithLink const& cp, SignalGenerator::SignalGenerator  sig, const MatrixXd & covmat, const MatrixXd & invcovbg, MyConfig const & config, MyGrid const & MG, size_t nUniverses, HighFive::File* file) {

  std::mt19937 rng(cp.gid());
  diy::mpi::communicator world;
  const auto  [rankwork, ds_bgn, _ds_len] = mkRankWork(MG.points.rows(), nUniverses, world.rank(),  world.size());
  const size_t ds_len(_ds_len);
  ArrayXXi res_meta(ds_len, 2);
  ArrayXXd res_val( ds_len, 7);

  size_t idx(0);

  for (auto r : rankwork) {
     size_t i_grid = r[0];

     Array3d xeval; xeval << MG.points(i_grid, 0), 1, MG.points(i_grid, 1);
     auto const D  = sig.predict(xeval, 1);
     auto const Dc = collapseVector(D, config);
     
     Eigen::MatrixXd const & LMAT = cholD(covmat, D);


     for (size_t uu=r[1]; uu<r[2];++uu) {
        VectorXd const fake_data  = ProblemSBN::poisson_fluctuate(sample(D, LMAT, rng), rng);
        VectorXd const fake_dataC = collapseVector(fake_data, config); 
        MyFitData mfd = {fake_dataC, covmat, invcovbg, sig, config};

        size_t n_iter(0);
        double last_chi_min = 1e40;
        int    pbest = -99;
        int _pbest(-99);

        for(n_iter = 0; n_iter < 5; n_iter++){
            if(n_iter!=0){
                Vector3d xeval; xeval << MG.points(pbest,0), 1, MG.points(pbest,1);
                auto D     = mfd.sig.predict(xeval, 1);
                auto CINV  = updateInvCov(mfd.covmat, D, mfd.config);
                mfd.invcovbg = std::move(CINV);
            }
            auto [chi_min, _pbest]  = universeChi2(mfd, MG);//, goodpoints);
            if(n_iter!=0){
                if(fabs(chi_min-last_chi_min) < .001){
                    last_chi_min = chi_min;
                    pbest = _pbest;
                    break;
                }
            }
            last_chi_min = chi_min;
                    pbest = _pbest;
        }



        Vector3d xmineval; xmineval << MG.points(pbest,0), 1, MG.points(pbest,1);
        double this_chi = calcChi(fake_dataC, Dc, mfd.invcovbg);


        res_meta(idx,0) = r[0];
        res_meta(idx,1) = uu;
        res_val(idx,0) = MG.points(pbest,0);
        res_val(idx,1) = MG.points(pbest,1);
        res_val(idx,2) = this_chi;
        res_val(idx,3) = last_chi_min;
        res_val(idx,4) = last_chi_min;
        res_val(idx,5) = MG.points(i_grid, 0);
        res_val(idx,6) = MG.points(i_grid, 1);
        idx++;
       }
       if (world.rank()==0 && idx%100==0) fmt::print(stderr, "[{}] progress: {}/{}\n",cp.gid(), idx, ds_len);
    }

    HighFive::DataSet d_point    = file->getDataSet("point");
    HighFive::DataSet d_universe = file->getDataSet("universe");
    d_point   .select({0, ds_bgn}, {1,idx}).write(res_meta.col(0).data());
    d_universe.select({0, ds_bgn}, {1,idx}).write(res_meta.col(1).data());

    HighFive::DataSet d_xmin     = file->getDataSet("xmin");
    HighFive::DataSet d_ymin     = file->getDataSet("ymin");
    HighFive::DataSet d_chi2     = file->getDataSet("thischi2");
    HighFive::DataSet d_minchi2  = file->getDataSet("minchi2");
    HighFive::DataSet d_minchi2pf= file->getDataSet("minchi2pf");
    HighFive::DataSet d_xtrue    = file->getDataSet("xtrue");
    HighFive::DataSet d_ytrue    = file->getDataSet("ytrue");

    
    d_xmin     .select({0, ds_bgn}, {1,idx}).write(res_val.col(0).data());
    d_ymin     .select({0, ds_bgn}, {1,idx}).write(res_val.col(1).data());
    d_chi2     .select({0, ds_bgn}, {1,idx}).write(res_val.col(2).data());
    d_minchi2  .select({0, ds_bgn}, {1,idx}).write(res_val.col(3).data());
    d_minchi2pf.select({0, ds_bgn}, {1,idx}).write(res_val.col(4).data());

    d_xtrue    .select({0, ds_bgn}, {1,idx}).write(res_val.col(5).data());
    d_ytrue    .select({0, ds_bgn}, {1,idx}).write(res_val.col(6).data());
}


void readConfig(const char* fname, int & nbx, double & xmin, double & xmax, int & nby, double & ymin, double & ymax) {
    std::ifstream f(fname);
    f >> nbx;
    f >> xmin;
    f >> xmax;
    f >> nby;
    f >> ymin;
    f >> ymax;
}

int main(int argc, char* argv[]) {
    diy::mpi::environment env(argc, argv);
    diy::mpi::communicator world;

    std::vector<double> v_buff;
    int ncols, nrows;
    if (world.rank()==0) loadData(argv[1], "covmat",      v_buff,   nrows, ncols);
    MatrixXd covmat = bcMatrixXd(world, v_buff, nrows, ncols);      
    if (world.rank()==0) loadData(argv[1], "core",        v_buff,   nrows, ncols);
    VectorXd core = bcMatrixXd(world, v_buff, nrows, ncols);
    if (world.rank()==0) loadData(argv[1], "invcovbg",      v_buff,   nrows, ncols);
    MatrixXd invcovbg = bcMatrixXd(world, v_buff, nrows, ncols);      
   
    auto config = mkConfig(); 
    SignalGenerator::SignalGenerator sig(mkAppEval(world, argv[2]), mkAppEval(world, argv[3]), core, config);

    double xmin, xmax, ymin, ymax;
    int nbx, nby;
    readConfig(argv[4], nbx, xmin, xmax, nby, ymin, ymax);
    MyGrid MG = mkGrid(nbx, xmin, xmax, nby, ymin, ymax);

    
    int nuniv = atoi(argv[5]);
    std::string out_file=argv[6];
    
    HighFive::File* f_out  = new HighFive::File(out_file,
                        HighFive::File::ReadWrite|HighFive::File::Create|HighFive::File::Truncate,
                        HighFive::MPIOFileDriver(MPI_COMM_WORLD,MPI_INFO_NULL));
    
    writeGrid(f_out, MG.points);
    createDataSets(f_out, MG.points.rows(), nuniv);

    size_t blocks = world.size();//nPoints;//*nUniverses;
    if (world.rank()==0) fmt::print(stderr, "FC will be done on {} blocks, distributed over {} ranks\n", blocks, world.size());
    Bounds fc_domain(1);
    fc_domain.min[0] = 0;
    fc_domain.max[0] = blocks-1;
    diy::RoundRobinAssigner        fc_assigner(world.size(), blocks);
    diy::Master                    fc_master(world, 1, -1, &FitBlock::create, &FitBlock::destroy);
    diy::decompose(1, world.rank(), fc_domain, fc_assigner, fc_master);
   
    fc_master.foreach([world, sig, covmat, invcovbg, config, MG, nuniv, f_out](FitBlock* b, const diy::Master::ProxyWithLink& cp) {runMin(b, cp, sig, covmat, invcovbg, config, MG, nuniv, f_out); });
    //fc_master.foreach([world, sig, covmat, invcovbg, config, MG, nuniv, f_out](FitBlock* b, const diy::Master::ProxyWithLink& cp) {runGrid(b, cp, sig, covmat, invcovbg, config, MG, nuniv, f_out); });
    
    delete f_out;
    return 0;
}
