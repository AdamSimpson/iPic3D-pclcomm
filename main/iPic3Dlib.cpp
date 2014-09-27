
#include "iPic3D.h"
#include "TimeTasks.h"
#include "ipicdefs.h"
#include "debug.h"
#include "Parameters.h"
#include "ompdefs.h"

#include "Moments.h" // for debugging

using namespace iPic3D;
//MPIdata* iPic3D::c_Solver::mpi=0;

c_Solver::~c_Solver()
{
  delete col; // configuration parameters ("collectiveIO")
  delete vct; // process topology
  delete grid; // grid
  delete EMf; // field

  // delete particles
  //
  if(part)
  {
    for (int i = 0; i < ns; i++)
    {
      // placement delete
      part[i].~Particles3D();
    }
    free(part);
  }

  delete [] Ke;
  delete [] momentum;
  delete [] Qremoved;
  delete my_clock;
}

int c_Solver::Init(int argc, char **argv) {
  #if defined(__MIC__)
  assert_eq(DVECWIDTH,8);
  #endif
  // get MPI data
  //
  // c_Solver is not a singleton, so the following line was pulled out.
  //MPIdata::init(&argc, &argv);
  //
  // initialized MPI environment
  // nprocs = number of processors
  // myrank = rank of tha process*/
  Parameters::init_parameters();
  //mpi = &MPIdata::instance();
  nprocs = MPIdata::get_nprocs();
  myrank = MPIdata::get_rank();

  col = new Collective(argc, argv); // Every proc loads the parameters of simulation from class Collective
  verbose = col->getVerbose();
  restart_cycle = col->getRestartOutputCycle();
  SaveDirName = col->getSaveDirName();
  RestartDirName = col->getRestartDirName();
  restart = col->getRestart_status();
  ns = col->getNs();            // get the number of particle species involved in simulation
  first_cycle = col->getLast_cycle() + 1; // get the last cycle from the restart
  // initialize the virtual cartesian topology 
  vct = new VCtopology3D(*col);
  // Check if we can map the processes into a matrix ordering defined in Collective.cpp
  if (nprocs != vct->getNprocs()) {
    if (myrank == 0) {
      cerr << "Error: " << nprocs << " processes cant be mapped into " << vct->getXLEN() << "x" << vct->getYLEN() << "x" << vct->getZLEN() << " matrix: Change XLEN,YLEN, ZLEN in method VCtopology3D.init()" << endl;
      MPIdata::instance().finalize_mpi();
      return (1);
    }
  }
  // We create a new communicator with a 3D virtual Cartesian topology
  vct->setup_vctopology(MPI_COMM_WORLD);
  // initialize the central cell index

#ifdef BATSRUS
  // set index offset for each processor
  col->setGlobalStartIndex(vct);
#endif

  // Print the initial settings to stdout and a file
  if (myrank == 0) {
    MPIdata::instance().Print();
    vct->Print();
    col->Print();
    col->save();
  }
  // Create the local grid
  former_MPI_Barrier(MPI_COMM_WORLD);
  grid = new Grid3DCU(col, vct);  // Create the local grid
  EMf = new EMfields3D(col, grid);  // Create Electromagnetic Fields Object

  if      (col->getCase()=="GEMnoPert") EMf->initGEMnoPert(vct,grid,col);
  else if (col->getCase()=="ForceFree") EMf->initForceFree(vct,grid,col);
  else if (col->getCase()=="GEM")       EMf->initGEM(vct, grid,col);
#ifdef BATSRUS
  else if (col->getCase()=="BATSRUS")   EMf->initBATSRUS(vct,grid,col);
#endif
  else if (col->getCase()=="Dipole")    EMf->initDipole(vct,grid,col);
  else if (col->getCase()=="DipoleNoCurv")    EMf->initDipoleNoCurv(vct,grid,col);
  else if (col->getCase()=="RandomCase") {
    EMf->initRandomField(vct,grid,col);
    if (myrank==0) {
      cout << "Case is " << col->getCase() <<"\n";
      cout <<"total # of particle per cell is " << col->getNpcel(0) << "\n";
    }
  }
  else {
    if (myrank==0) {
      cout << " =========================================================== " << endl;
      cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
      cout << "          Runing simulation with the default initialization. " << endl;
      cout << " =========================================================== " << endl;
    }
    EMf->init(vct,grid,col);
  }

  // OpenBC
  EMf->updateInfoFields(grid,vct,col);

  // Allocation of particles
  // part = new Particles3D[ns];
  part = (Particles3D*) malloc(sizeof(Particles3D)*ns);
  for (int i = 0; i < ns; i++)
  {
    new(&part[i]) Particles3D(i,col,vct,grid);
    //part[i] = new Particles3D(i, col, vct, grid);
    //part[i].allocate(i, col, vct, grid);
  }

  // Initial Condition for PARTICLES if you are not starting from RESTART
  if (restart == 0) {
    // wave = new Planewave(col, EMf, grid, vct);
    // wave->Wave_Rotated(part); // Single Plane Wave
    for (int i = 0; i < ns; i++)
    {
      if      (col->getCase()=="ForceFree") part[i].force_free(EMf);
#ifdef BATSRUS
      else if (col->getCase()=="BATSRUS")   part[i].MaxwellianFromFluid(EMf,col,i);
#endif
      else if (col->getCase()=="WhistlerMode")   part[i].maxwellianWithPerturbation(EMf);
      else                                  part[i].maxwellian(EMf);
      part[i].reserve_remaining_particle_IDs();
    }
  }

  // Initialize the output (simulation results and restart file)
  // PSK::OutputManager < PSK::OutputAdaptor > output_mgr; // Create an Output Manager
  // myOutputAgent < PSK::HDF5OutputAdaptor > hdf5_agent; // Create an Output Agent for HDF5 output
  hdf5_agent.set_simulation_pointers(EMf, grid, vct, col);
  for (int i = 0; i < ns; ++i)
    hdf5_agent.set_simulation_pointers_part(&part[i]);
  output_mgr.push_back(&hdf5_agent);  // Add the HDF5 output agent to the Output Manager's list
  if (myrank == 0 & restart < 2) {
    hdf5_agent.open(SaveDirName + "/settings.hdf");
    output_mgr.output("collective + total_topology + proc_topology", 0);
    hdf5_agent.close();
    hdf5_agent.open(RestartDirName + "/settings.hdf");
    output_mgr.output("collective + total_topology + proc_topology", 0);
    hdf5_agent.close();
  }
  // Restart
  num_proc << myrank;
  if (restart == 0) {           // new simulation from input file
    hdf5_agent.open(SaveDirName + "/proc" + num_proc.str() + ".hdf");
    output_mgr.output("proc_topology ", 0);
    hdf5_agent.close();
  }
  else {                        // restart append the results to the previous simulation 
    hdf5_agent.open_append(SaveDirName + "/proc" + num_proc.str() + ".hdf");
    output_mgr.output("proc_topology ", 0);
    hdf5_agent.close();
  }

  former_MPI_Barrier(MPI_COMM_WORLD);
  Eenergy, Benergy, TOTenergy = 0.0, TOTmomentum = 0.0;
  Ke = new double[ns];
  momentum = new double[ns];
  cq = SaveDirName + "/ConservedQuantities.txt";
  if (myrank == 0) {
    ofstream my_file(cq.c_str());
    my_file.close();
  }
  // Distribution functions
//  nDistributionBins = 1000;
//  ds = SaveDirName + "/DistributionFunctions.txt";
//  if (myrank == 0) {
//    ofstream my_file(ds.c_str());
//    my_file.close();
//  }
//  cqsat = SaveDirName + "/VirtualSatelliteTraces" + num_proc.str() + ".txt";
//  // if(myrank==0)
//  ofstream my_file(cqsat.c_str(), fstream::binary);
//  nsat = 3;
//  const int nx0 = grid->get_nxc_r();
//  const int ny0 = grid->get_nyc_r();
//  const int nz0 = grid->get_nzc_r();
//  for (int isat = 0; isat < nsat; isat++) {
//    for (int jsat = 0; jsat < nsat; jsat++) {
//      for (int ksat = 0; ksat < nsat; ksat++) {
//        int index1 = 1 + isat * nx0 / nsat + nx0 / nsat / 2;
//        int index2 = 1 + jsat * ny0 / nsat + ny0 / nsat / 2;
//        int index3 = 1 + ksat * nz0 / nsat + nz0 / nsat / 2;
//        my_file << grid->getXC(index1, index2, index3) << "\t" << grid->getYC(index1, index2, index3) << "\t" << grid->getZC(index1, index2, index3) << endl;
//      }}}
//  my_file.close();

  Qremoved = new double[ns];

  my_clock = new Timing(myrank);

  return 0;
}

void c_Solver::sortParticles() {
  timeTasks_begin_task(TimeTasks::MOMENT_PCL_SORTING);
  for(int species_idx=0; species_idx<ns; species_idx++)
    part[species_idx].sort_particles_serial();
  timeTasks_end_task(TimeTasks::MOMENT_PCL_SORTING);
}

void c_Solver::CalculateMoments() {

  timeTasks_set_main_task(TimeTasks::MOMENTS);

  pad_particle_capacities();

  // vectorized assumes that particles are sorted by mesh cell
  if(Parameters::get_VECTORIZE_MOMENTS())
  {
    switch(Parameters::get_MOMENTS_TYPE())
    {
      case Parameters::SoA:
        // since particles are sorted,
        // we can vectorize interpolation of particles to grid
        convertParticlesToSoA();
        sortParticles();
        EMf->sumMoments_vectorized(part, grid, vct);
        break;
      case Parameters::AoS:
        convertParticlesToAoS();
        sortParticles();
        EMf->sumMoments_vectorized_AoS(part, grid, vct);
        break;
      default:
        unsupported_value_error(Parameters::get_MOMENTS_TYPE());
    }
  }
  else
  {
    if(Parameters::get_SORTING_PARTICLES())
      sortParticles();
    switch(Parameters::get_MOMENTS_TYPE())
    {
      case Parameters::SoA:
        EMf->setZeroPrimaryMoments();
        convertParticlesToSoA();
        EMf->sumMoments(part, grid, vct);
        break;
      case Parameters::AoS:
        EMf->setZeroPrimaryMoments();
        convertParticlesToAoS();
        EMf->sumMoments_AoS(part, grid, vct);
        break;
      case Parameters::AoSintr:
        EMf->setZeroPrimaryMoments();
        convertParticlesToAoS();
        EMf->sumMoments_AoS_intr(part, grid, vct);
        break;
      default:
        unsupported_value_error(Parameters::get_MOMENTS_TYPE());
    }
  }
  //for (int i = 0; i < ns; i++)
  //{
  //  EMf->sumMomentsOld(part[i], grid, vct);
  //}
  EMf->setZeroDerivedMoments();
  EMf->sumOverSpecies(vct);                 // sum all over the species

  // Fill with constant charge the planet
  if (col->getCase()=="Dipole") {
    EMf->ConstantChargePlanet(grid, vct, col->getL_square(),col->getx_center(),col->gety_center(),col->getz_center());
  }

  EMf->ConstantChargeOpenBC(grid, vct);     // Set a constant charge in the OpenBC boundaries

  former_MPI_Barrier(MPI_COMM_WORLD);

  EMf->interpDensitiesN2C(vct, grid);       // calculate densities on centers from nodes
  EMf->calculateHatFunctions(grid, vct);    // calculate the hat quantities for the implicit method
  former_MPI_Barrier(MPI_COMM_WORLD);

  // why is this being done here?
  EMf->updateInfoFields(grid,vct,col);
}

//! MAXWELL SOLVER for Efield
void c_Solver::CalculateField() {
  timeTasks_set_main_task(TimeTasks::FIELDS);
  EMf->calculateE(grid, vct, col);               // calculate the E field
}

//! MAXWELL SOLVER for Bfield (assuming Efield has already been calculated)
void c_Solver::CalculateB() {
  timeTasks_set_main_task(TimeTasks::FIELDS);
  timeTasks_set_task(TimeTasks::BFIELD); // subtask
  EMf->calculateB(grid, vct, col);   // calculate the B field
}

/*  -------------- */
/*!  Particle mover */
/*  -------------- */
bool c_Solver::ParticlesMover()
{
  // move all species of particles
  {
    timeTasks_set_main_task(TimeTasks::PARTICLES);
    // Should change this to add background field
    EMf->set_fieldForPcls();

    pad_particle_capacities();

    #pragma omp parallel
    {
    for (int i = 0; i < ns; i++)  // move each species
    {
      // #pragma omp task inout(part[i]) in(grid) target_device(booster)
      //
      // should merely pass EMf->get_fieldForPcls() rather than EMf.
      // use the Predictor Corrector scheme to move particles
      switch(Parameters::get_MOVER_TYPE())
      {
        case Parameters::SoA:
          part[i].mover_PC(EMf);
          break;
        //case Parameters::SoA_vec_resort:
        //  part[i].mover_PC_vectorized(EMf);
        //  break;
        case Parameters::AoS:
          part[i].mover_PC_AoS(EMf);
          break;
        case Parameters::AoSintr:
          part[i].mover_PC_AoS_vec_intr(EMf);
          break;
        case Parameters::AoSvec:
          part[i].mover_PC_AoS_vec(EMf);
          break;
        //case Parameters::AoS_vec_onesort:
        //  part[i].mover_PC_AoS_vec_onesort(EMf);
        //  break;
        default:
          unsupported_value_error(Parameters::get_MOVER_TYPE());
      }
      // overlap initial communication of electrons with moving of ions
      #pragma omp master
      part[i].separate_and_send_particles();
    }
    }
    for (int i = 0; i < ns; i++)  // communicate each species
    {
      //part[i].communicate_particles();
      part[i].recommunicate_particles_until_done(1);
    }
  }

  /* -------------------------------------- */
  /* Repopulate the buffer zone at the edge */
  /* -------------------------------------- */

  for (int i=0; i < ns; i++) {
    if (col->getRHOinject(i)>0.0)
      part[i].repopulate_particles();
  }

  /* --------------------------------------- */
  /* Remove particles from depopulation area */
  /* --------------------------------------- */

  if (col->getCase()=="Dipole") {
    for (int i=0; i < ns; i++)
      Qremoved[i] = part[i].deleteParticlesInsideSphere(col->getL_square(),col->getx_center(),col->gety_center(),col->getz_center());
  }
  return (false);
}

void c_Solver::WriteRestart(int cycle)
{
  bool do_WriteRestart = (cycle % restart_cycle == 0 && cycle != first_cycle);
  if(!do_WriteRestart)
    return;

  convertParticlesToSynched(); // hack
  // write the RESTART file
  writeRESTART(RestartDirName, myrank, cycle, ns, vct, col, grid, EMf, part, 0); // without ,0 add to restart file
}

// write the conserved quantities
void c_Solver::WriteConserved(int cycle) {
  if(cycle % col->getDiagnosticsOutputCycle() == 0)
  {
    Eenergy = EMf->getEenergy();
    Benergy = EMf->getBenergy();
    TOTenergy = 0.0;
    TOTmomentum = 0.0;
    for (int is = 0; is < ns; is++) {
      Ke[is] = part[is].getKe();
      TOTenergy += Ke[is];
      momentum[is] = part[is].getP();
      TOTmomentum += momentum[is];
    }
    if (myrank == 0) {
      ofstream my_file(cq.c_str(), fstream::app);
      my_file << cycle << "\t" << "\t" << (Eenergy + Benergy + TOTenergy) << "\t" << TOTmomentum << "\t" << Eenergy << "\t" << Benergy << "\t" << TOTenergy << endl;
      my_file.close();
    }
  }
}

void c_Solver::WriteVelocityDistribution(int cycle)
{
  // Velocity distribution
  //if(cycle % col->getVelocityDistributionOutputCycle() == 0)
  {
    for (int is = 0; is < ns; is++) {
      double maxVel = part[is].getMaxVelocity();
      long long *VelocityDist = part[is].getVelocityDistribution(nDistributionBins, maxVel);
      if (myrank == 0) {
        ofstream my_file(ds.c_str(), fstream::app);
        my_file << cycle << "\t" << is << "\t" << maxVel;
        for (int i = 0; i < nDistributionBins; i++)
          my_file << "\t" << VelocityDist[i];
        my_file << endl;
        my_file.close();
      }
      delete [] VelocityDist;
    }
  }
}

// This seems to record values at a grid of sample points
//
void c_Solver::WriteVirtualSatelliteTraces()
{
  if(ns <= 2) return;
  assert_eq(ns,4);

  ofstream my_file(cqsat.c_str(), fstream::app);
  const int nx0 = grid->get_nxc_r();
  const int ny0 = grid->get_nyc_r();
  const int nz0 = grid->get_nzc_r();
  for (int isat = 0; isat < nsat; isat++) {
    for (int jsat = 0; jsat < nsat; jsat++) {
      for (int ksat = 0; ksat < nsat; ksat++) {
        int index1 = 1 + isat * nx0 / nsat + nx0 / nsat / 2;
        int index2 = 1 + jsat * ny0 / nsat + ny0 / nsat / 2;
        int index3 = 1 + ksat * nz0 / nsat + nz0 / nsat / 2;
        my_file << EMf->getBx(index1, index2, index3) << "\t" << EMf->getBy(index1, index2, index3) << "\t" << EMf->getBz(index1, index2, index3) << "\t";
        my_file << EMf->getEx(index1, index2, index3) << "\t" << EMf->getEy(index1, index2, index3) << "\t" << EMf->getEz(index1, index2, index3) << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 0) + EMf->getJxs(index1, index2, index3, 2) << "\t" << EMf->getJys(index1, index2, index3, 0) + EMf->getJys(index1, index2, index3, 2) << "\t" << EMf->getJzs(index1, index2, index3, 0) + EMf->getJzs(index1, index2, index3, 2) << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 1) + EMf->getJxs(index1, index2, index3, 3) << "\t" << EMf->getJys(index1, index2, index3, 1) + EMf->getJys(index1, index2, index3, 3) << "\t" << EMf->getJzs(index1, index2, index3, 1) + EMf->getJzs(index1, index2, index3, 3) << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 0) + EMf->getRHOns(index1, index2, index3, 2) << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 1) + EMf->getRHOns(index1, index2, index3, 3) << "\t";
      }}}
  my_file << endl;
  my_file.close();
}

void c_Solver::WriteFields(int cycle) {
  if(cycle % (col->getFieldOutputCycle()) == 0 || cycle == first_cycle)
  {
    if (col->getWriteMethod() == "Parallel") {
        WriteOutputParallel(grid, EMf, col, vct, cycle);
    }
    else // OUTPUT to large file, called proc**
    {
        hdf5_agent.open_append(SaveDirName + "/proc" + num_proc.str() + ".hdf");
        output_mgr.output("Eall + Ball + rhos + Jsall + pressure", cycle);
        // Pressure tensor is available
        hdf5_agent.close();
    }
  }
}

void c_Solver::WriteParticles(int cycle)
{
  const bool do_WriteParticles
    = (cycle % (col->getParticlesOutputCycle()) == 0
       && col->getParticlesOutputCycle() != 1);
  if(!do_WriteParticles)
    return;

  // this is a hack
  convertParticlesToSynched();

  if (col->getWriteMethod() == "Parallel")
  {
    dprintf("pretending to write particles (not yet implemented)");
  }
  else
  {
    hdf5_agent.open_append(SaveDirName + "/proc" + num_proc.str() + ".hdf");
    output_mgr.output("position + velocity + q ", cycle, 1);
    hdf5_agent.close();
  }
}

// This needs to be separated into methods that save particles
// and methods that save field data
//
void c_Solver::WriteOutput(int cycle) {
  // write fields-related data
  WriteFields(cycle);
  if (col->getWriteMethod() != "Parallel")
  {
    // This should be invoked by user if desired
    // by means of a callback mechanism.
    //WriteVirtualSatelliteTraces();
  }

  // write particles-related data
  //
  // this also writes field data...
  WriteRestart(cycle);
  WriteParticles(cycle);
  //
  // This should be invoked by user if desired
  // by means of a callback mechanism.
  //WriteVelocityDistribution(cycle);
  WriteConserved(cycle);
}

void c_Solver::Finalize() {
  if (col->getCallFinalize())
  {
    convertParticlesToSynched();
    writeRESTART(RestartDirName, myrank, (col->getNcycles() + first_cycle) - 1, ns, vct, col, grid, EMf, part, 0);
  }

  // stop profiling
  my_clock->stopTiming();
}

//void c_Solver::copyParticlesToSoA()
//{
//  for (int i = 0; i < ns; i++)
//    part[i].copyParticlesToSoA();
//}

void c_Solver::pad_particle_capacities()
{
  for (int i = 0; i < ns; i++)
    part[i].pad_capacities();
}

// convert particle to struct of arrays (assumed by I/O)
void c_Solver::convertParticlesToSoA()
{
  for (int i = 0; i < ns; i++)
    part[i].convertParticlesToSoA();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertParticlesToAoS()
{
  for (int i = 0; i < ns; i++)
    part[i].convertParticlesToAoS();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertParticlesToSynched()
{
  for (int i = 0; i < ns; i++)
    part[i].convertParticlesToSynched();
}
