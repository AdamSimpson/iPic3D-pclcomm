/***************************************************************************
  iPIC3D.cpp  -  Main file for 3D simulation
  -------------------
 ************************************************************************** */

#ifndef _IPIC3D_H_
#define _IPIC3D_H_

#include "MPIdata.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Particles3D.h"
#include "Restart3D.h"
#include "Timing.h"
#include "WriteOutputParallel.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
using std::cerr;
using std::endl;
using std::ofstream;

using namespace PSK;

namespace iPic3D {

  class c_Solver {

  public:
    ~c_Solver();
    c_Solver():
      col(0),
      vct(0),
      grid(0),
      EMf(0),
      part(0),
      Ke(0),
      momentum(0),
      Qremoved(0),
      my_clock(0)
    {}
    int Init(int argc, char **argv);
    void CalculateMoments();
    void CalculateField(); //! calculate Efield
    bool ParticlesMover();
    void CalculateB();
    //
    // output methods
    //
    void WriteRestart(int cycle);
    void WriteConserved(int cycle);
    void WriteVelocityDistribution(int cycle);
    void WriteVirtualSatelliteTraces();
    void WriteFields(int cycle);
    void WriteParticles(int cycle);
    void WriteOutput(int cycle);
    void Finalize();

    inline int FirstCycle();
    inline int LastCycle();
    inline int get_myrank();

  private:
    void pad_particle_capacities();
    void convertParticlesToSoA();
    void convertParticlesToAoS();
    void convertParticlesToSynched();
    void sortParticles();

  private:
    //static MPIdata * mpi;
    Collective    *col;
    VCtopology3D  *vct;
    Grid3DCU      *grid;
    EMfields3D    *EMf;
    Particles3D   *part;
    double        *Ke;
    double        *momentum;
    double        *Qremoved;
    Timing        *my_clock;

    PSK::OutputManager < PSK::OutputAdaptor > output_mgr; // Create an Output Manager
    myOutputAgent < PSK::HDF5OutputAdaptor > hdf5_agent;  // Create an Output Agent for HDF5 output

    bool verbose;
    string SaveDirName;
    string RestartDirName;
    string cqsat;
    string cq;
    string ds;
    stringstream num_proc;
    int restart_cycle;
    int restart;
    int first_cycle;
    int ns;
    int nprocs;
    int myrank;
    int nsat;
    int nDistributionBins;
    double Eenergy;
    double Benergy;
    double TOTenergy;
    double TOTmomentum;
  };

  inline int c_Solver::FirstCycle() {
    return (first_cycle);
  }
  inline int c_Solver::LastCycle() {
    return (col->getNcycles() + first_cycle);
  }
  inline int c_Solver::get_myrank() {
    return (myrank);
  }
}

#endif
