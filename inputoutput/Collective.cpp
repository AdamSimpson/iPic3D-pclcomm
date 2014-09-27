
#include <mpi.h>
#include <math.h>
//#include <iostream>
//#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "input_array.h"
#include "hdf5.h"
#include "Collective.h"
#include "ConfigFile.h"
#include "limits.h" // for INT_MAX
#include "MPIdata.h"
#include "errors.h"
#include "asserts.h" // for assert_ge
#include "string.h"

// order must agree with Enum in Collective.h
static const char *enumNames[] =
{
    "default",
    "initial",
    "final",
    // used by ImplSusceptMode
    "explPredict",
    "implPredict",
    // marker for last enumerated symbol of this class
    "NUMBER_OF_ENUMS",
    "INVALID_ENUM"
};

int Collective::read_enum_parameter(const char* option_name, const char* default_value,
                                    const ConfigFile& config)
{
    string enum_name = config.read < string >(option_name,default_value);
    // search the list (could use std::map)
    //
    for(int i=0;i<NUMBER_OF_ENUMS;i++)
    {
        if(!strcmp(enum_name.c_str(),enumNames[i]))
            return i;
    }
    // could not find enum, so issue error and quit.
    if(!MPIdata::get_rank())
    {
        eprintf("in input file %s there is an invalid option %s\n",
                inputfile.c_str(), enum_name.c_str());
    }
    MPIdata::exit(1);
    // this is a better way
    return INVALID_ENUM;
}

const char* Collective::get_name_of_enum(int in)
{
    assert_ge(in, 0);
    assert_lt(in, NUMBER_OF_ENUMS);
    return enumNames[in];
}

/*! Read the input file from text file and put the data in a collective wrapper: if it's a restart read from input file basic sim data and load particles and EM field from restart file */
void Collective::ReadInput(string inputfile) {
    using namespace std;
    int test_verbose;
    // Loading the input file
    ConfigFile config(inputfile);
    // the following variables are ALWAYS taken from inputfile, even if restarting
    {
        
#ifdef BATSRUS
        if(RESTART1)
        {
            cout<<" The fluid interface can not handle RESTART yet, aborting!\n"<<flush;
            abort();
        }
#endif
        
        dt = config.read < double >("dt",0.1);
        ncycles = config.read < int >("ncycles",1000);
        th = config.read < double >("th",1.0);
        config.readInto(Smooth, "Smooth",1.0);
        SaveDirName = config.read < string > ("SaveDirName","data");
        RestartDirName = config.read < string > ("RestartDirName","data");
        ns = config.read < int >("ns",2);
        NpMaxNpRatio = config.read < double >("NpMaxNpRatio",2);
        assert_ge(NpMaxNpRatio, 1.);
        // mode parameters for second order in time
        PushWithBatTime = config.read < double >("PushWithBatTime",0);
        PushWithEatTime = config.read < double >("PushWithEatTime",1);
        ImplSusceptTime = config.read < double >("ImplSusceptTime",0);
        ImplSusceptMode = read_enum_parameter("ImplSusceptMode", "initial",config);
        switch(ImplSusceptMode)
        {
                // values not yet supported:
            case explPredict:
            case implPredict:
            default:
                unsupported_value_error(ImplSusceptMode);
                // supported values:
            case initial:
                ;
        }
        // GEM Challenge
        B0x = config.read <double>("B0x",0.0);
        B0y = config.read <double>("B0y",0.0);
        B0z = config.read <double>("B0z",0.0);
        
        // Earth parameters
        B1x = 0.0;
        B1y = 0.0;
        B1z = 0.0;
        B1x = config.read <double>("B1x",0.0);
        B1y = config.read <double>("B1y",0.0);
        B1z = config.read <double>("B1z",0.0);
        
        delta = config.read < double >("delta",0.5);
        
        Case              = config.read<string>("Case","GEM");
        wmethod           = config.read<string>("WriteMethod","default");
        SimName           = config.read<string>("SimulationName","Simulation");
        PoissonCorrection = config.read<string>("PoissonCorrection","yes");
        
        rhoINIT = new double[ns];
        array_double rhoINIT0 = config.read < array_double > ("rhoINIT");
        rhoINIT[0] = rhoINIT0.a;
        if (ns > 1)
            rhoINIT[1] = rhoINIT0.b;
        if (ns > 2)
            rhoINIT[2] = rhoINIT0.c;
        if (ns > 3)
            rhoINIT[3] = rhoINIT0.d;
        if (ns > 4)
            rhoINIT[4] = rhoINIT0.e;
        if (ns > 5)
            rhoINIT[5] = rhoINIT0.f;
        
        rhoINJECT = new double[ns];
        
        // DEFAULT FOR INJECTION
        array_double rhoINJECT_DEFAULT;
        rhoINJECT_DEFAULT.a= 1.0;
        if (ns > 1)
            rhoINJECT_DEFAULT.b= 1.0;
        if (ns > 2)
            rhoINJECT_DEFAULT.c= 1.0;
        if (ns > 3)
            rhoINJECT_DEFAULT.d= 1.0;;
        if (ns > 4)
            rhoINJECT_DEFAULT.e= 1.0;;
        if (ns > 5)
            rhoINJECT_DEFAULT.f= 1.0;
        
        array_double rhoINJECT0 = config.read<array_double>("rhoINJECT",rhoINJECT_DEFAULT);
        
        rhoINJECT[0]=rhoINJECT0.a;
        if (ns > 1)
            rhoINJECT[1]=rhoINJECT0.b;
        if (ns > 2)
            rhoINJECT[2]=rhoINJECT0.c;
        if (ns > 3)
            rhoINJECT[3]=rhoINJECT0.d;
        if (ns > 4)
            rhoINJECT[4]=rhoINJECT0.e;
        if (ns > 5)
            rhoINJECT[5]=rhoINJECT0.f;
        
        // take the tolerance of the solvers
        CGtol = config.read < double >("CGtol",1E-3);
        GMREStol = config.read < double >("GMREStol",1E-3);
        NiterMover = config.read < int >("NiterMover",3);
        // take the injection of the particless
        Vinj = config.read < double >("Vinj",0.0);
        
        // take the output cycles
        FieldOutputCycle = config.read < int >("FieldOutputCycle",100);
        ParticlesOutputCycle = config.read < int >("ParticlesOutputCycle",100000);
        RestartOutputCycle = config.read < int >("RestartOutputCycle",200000);
        DiagnosticsOutputCycle = config.read < int >("DiagnosticsOutputCycle", FieldOutputCycle);
        CallFinalize = config.read < bool >("CallFinalize", true);
    }
    
    if (RESTART1) {               // you are restarting
        RestartDirName = config.read < string > ("RestartDirName");
        ReadRestart(RestartDirName);
    }
    else {
        restart_status = 0;
        last_cycle = -1;
        c = config.read < double >("c",1.0);
        
#ifdef BATSRUS
        // set grid size and resolution based on the initial file from fluid code
        Lx =  getFluidLx();
        Ly =  getFluidLy();
        Lz =  getFluidLz();
        nxc = getFluidNxc();
        nyc = getFluidNyc();
        nzc = getFluidNzc();
#else
        Lx = config.read < double >("Lx");
        Ly = config.read < double >("Ly");
        Lz = config.read < double >("Lz");
        nxc = config.read < int >("nxc");
        nyc = config.read < int >("nyc");
        nzc = config.read < int >("nzc");
#endif
        XLEN = config.read < int >("XLEN",1);
        YLEN = config.read < int >("YLEN",1);
        ZLEN = config.read < int >("ZLEN",1);
        PERIODICX = config.read < bool >("PERIODICX",true);
        PERIODICY = config.read < bool >("PERIODICY",true);
        PERIODICZ = config.read < bool >("PERIODICZ",true);
        
        x_center = config.read < double >("x_center",1.0);
        y_center = config.read < double >("y_center",1.0);
        z_center = config.read < double >("z_center",1.0);
        L_square = config.read < double >("L_square",0.25);
        
        npcelx = new int[ns];
        npcely = new int[ns];
        npcelz = new int[ns];
        qom = new double[ns];
        uth = new double[ns];
        vth = new double[ns];
        wth = new double[ns];
        u0 = new double[ns];
        v0 = new double[ns];
        w0 = new double[ns];
        
        //nop_rs = new int[ns];
        
        array_int npcelx0 = config.read < array_int > ("npcelx");
        array_int npcely0 = config.read < array_int > ("npcely");
        array_int npcelz0 = config.read < array_int > ("npcelz");
        array_double qom0 = config.read < array_double > ("qom");
        array_double uth0 = config.read < array_double > ("uth");
        array_double vth0 = config.read < array_double > ("vth");
        array_double wth0 = config.read < array_double > ("wth");
        array_double u00 = config.read < array_double > ("u0");
        array_double v00 = config.read < array_double > ("v0");
        array_double w00 = config.read < array_double > ("w0");
        
        npcelx[0] = npcelx0.a;
        npcely[0] = npcely0.a;
        npcelz[0] = npcelz0.a;
        qom[0] = qom0.a;
        uth[0] = uth0.a;
        vth[0] = vth0.a;
        wth[0] = wth0.a;
        u0[0] = u00.a;
        v0[0] = v00.a;
        w0[0] = w00.a;
        
        if (ns > 1) {
            npcelx[1] = npcelx0.b;
            npcely[1] = npcely0.b;
            npcelz[1] = npcelz0.b;
            qom[1] = qom0.b;
            uth[1] = uth0.b;
            vth[1] = vth0.b;
            wth[1] = wth0.b;
            u0[1] = u00.b;
            v0[1] = v00.b;
            w0[1] = w00.b;
        }
        if (ns > 2) {
            npcelx[2] = npcelx0.c;
            npcely[2] = npcely0.c;
            npcelz[2] = npcelz0.c;
            qom[2] = qom0.c;
            uth[2] = uth0.c;
            vth[2] = vth0.c;
            wth[2] = wth0.c;
            u0[2] = u00.c;
            v0[2] = v00.c;
            w0[2] = w00.c;
        }
        if (ns > 3) {
            npcelx[3] = npcelx0.d;
            npcely[3] = npcely0.d;
            npcelz[3] = npcelz0.d;
            qom[3] = qom0.d;
            uth[3] = uth0.d;
            vth[3] = vth0.d;
            wth[3] = wth0.d;
            u0[3] = u00.d;
            v0[3] = v00.d;
            w0[3] = w00.d;
        }
        if (ns > 4) {
            npcelx[4] = npcelx0.e;
            npcely[4] = npcely0.e;
            npcelz[4] = npcelz0.e;
            qom[4] = qom0.e;
            uth[4] = uth0.e;
            vth[4] = vth0.e;
            wth[4] = wth0.e;
            u0[4] = u00.e;
            v0[4] = v00.e;
            w0[4] = w00.e;
        }
        if (ns > 5) {
            npcelx[5] = npcelx0.f;
            npcely[5] = npcely0.f;
            npcelz[5] = npcelz0.f;
            qom[5] = qom0.f;
            uth[5] = uth0.f;
            vth[5] = vth0.f;
            wth[5] = wth0.f;
            u0[5] = u00.f;
            v0[5] = v00.f;
            w0[1] = w00.f;
        }
        
        
        verbose = config.read < bool > ("verbose",true);
        
        // PHI Electrostatic Potential
        bcPHIfaceXright = config.read < int >("bcPHIfaceXright",1);
        bcPHIfaceXleft  = config.read < int >("bcPHIfaceXleft",1);
        bcPHIfaceYright = config.read < int >("bcPHIfaceYright",1);
        bcPHIfaceYleft  = config.read < int >("bcPHIfaceYleft",1);
        bcPHIfaceZright = config.read < int >("bcPHIfaceZright",1);
        bcPHIfaceZleft  = config.read < int >("bcPHIfaceZleft",1);
        
        // EM field boundary condition
        bcEMfaceXright = config.read < int >("bcEMfaceXright",0);
        bcEMfaceXleft  = config.read < int >("bcEMfaceXleft",0);
        bcEMfaceYright = config.read < int >("bcEMfaceYright",0);
        bcEMfaceYleft  = config.read < int >("bcEMfaceYleft",0);
        bcEMfaceZright = config.read < int >("bcEMfaceZright",0);
        bcEMfaceZleft  = config.read < int >("bcEMfaceZleft",0);
        
        /*  ---------------------------------------------------------- */
        /*  Electric and Magnetic field boundary conditions for BCface */
        /*  ---------------------------------------------------------- */
        // if bcEM* is 0: perfect conductor, if bcEM* is not 0: perfect mirror
        // perfect conductor: normal = free, perpendicular = 0
        // perfect mirror   : normal = 0,    perpendicular = free
        /*  ---------------------------------------------------------- */
        
        /* X component in faces Xright, Xleft, Yright, Yleft, Zright and Zleft (0, 1, 2, 3, 4, 5) */
        bcEx[0] = bcEMfaceXright == 0 ? 2 : 1;   bcBx[0] = bcEMfaceXright == 0 ? 1 : 2;
        bcEx[1] = bcEMfaceXleft  == 0 ? 2 : 1;   bcBx[1] = bcEMfaceXleft  == 0 ? 1 : 2;
        bcEx[2] = bcEMfaceYright == 0 ? 1 : 2;   bcBx[2] = bcEMfaceYright == 0 ? 2 : 1;
        bcEx[3] = bcEMfaceYleft  == 0 ? 1 : 2;   bcBx[3] = bcEMfaceYleft  == 0 ? 2 : 1;
        bcEx[4] = bcEMfaceZright == 0 ? 1 : 2;   bcBx[4] = bcEMfaceZright == 0 ? 2 : 1;
        bcEx[5] = bcEMfaceZleft  == 0 ? 1 : 2;   bcBx[5] = bcEMfaceZleft  == 0 ? 2 : 1;
        /* Y component */
        bcEy[0] = bcEMfaceXright == 0 ? 1 : 2;   bcBy[0] = bcEMfaceXright == 0 ? 2 : 1;
        bcEy[1] = bcEMfaceXleft  == 0 ? 1 : 2;   bcBy[1] = bcEMfaceXleft  == 0 ? 2 : 1;
        bcEy[2] = bcEMfaceYright == 0 ? 2 : 1;   bcBy[2] = bcEMfaceYright == 0 ? 1 : 2;
        bcEy[3] = bcEMfaceYleft  == 0 ? 2 : 1;   bcBy[3] = bcEMfaceYleft  == 0 ? 1 : 2;
        bcEy[4] = bcEMfaceZright == 0 ? 1 : 2;   bcBy[4] = bcEMfaceZright == 0 ? 2 : 1;
        bcEy[5] = bcEMfaceZleft  == 0 ? 1 : 2;   bcBy[5] = bcEMfaceZleft  == 0 ? 2 : 1;
        /* Z component */
        bcEz[0] = bcEMfaceXright == 0 ? 1 : 2;   bcBz[0] = bcEMfaceXright == 0 ? 2 : 1;
        bcEz[1] = bcEMfaceXleft  == 0 ? 1 : 2;   bcBz[1] = bcEMfaceXleft  == 0 ? 2 : 1;
        bcEz[2] = bcEMfaceYright == 0 ? 1 : 1;   bcBz[2] = bcEMfaceYright == 0 ? 2 : 1;
        bcEz[3] = bcEMfaceYleft  == 0 ? 1 : 1;   bcBz[3] = bcEMfaceYleft  == 0 ? 2 : 1;
        bcEz[4] = bcEMfaceZright == 0 ? 2 : 1;   bcBz[4] = bcEMfaceZright == 0 ? 1 : 2;
        bcEz[5] = bcEMfaceZleft  == 0 ? 2 : 1;   bcBz[5] = bcEMfaceZleft  == 0 ? 1 : 2;
        
        // Particles Boundary condition
        bcPfaceXright = config.read < int >("bcPfaceXright",1);
        bcPfaceXleft  = config.read < int >("bcPfaceXleft",1);
        bcPfaceYright = config.read < int >("bcPfaceYright",1);
        bcPfaceYleft  = config.read < int >("bcPfaceYleft",1);
        bcPfaceZright = config.read < int >("bcPfaceZright",1);
        bcPfaceZleft  = config.read < int >("bcPfaceZleft",1);
        
        
        
        
    }
    TrackParticleID = new bool[ns];
    array_bool TrackParticleID0 = config.read < array_bool > ("TrackParticleID");
    TrackParticleID[0] = TrackParticleID0.a;
    if (ns > 1)
        TrackParticleID[1] = TrackParticleID0.b;
    if (ns > 2)
        TrackParticleID[2] = TrackParticleID0.c;
    if (ns > 3)
        TrackParticleID[3] = TrackParticleID0.d;
    if (ns > 4)
        TrackParticleID[4] = TrackParticleID0.e;
    if (ns > 5)
        TrackParticleID[5] = TrackParticleID0.f;
    
    
    
    
}
/*! Read the collective information from the RESTART file in HDF5 format There are three restart status: restart_status = 0 ---> new inputfile restart_status = 1 ---> RESTART and restart and result directories does not coincide restart_status = 2 ---> RESTART and restart and result directories coincide */
int Collective::ReadRestart(string inputfile) {
    restart_status = 1;
    // hdf stuff
    hid_t file_id;
    hid_t dataset_id;
    herr_t status;
    // Open the setting file for the restart.
    file_id = H5Fopen((inputfile + "/settings.hdf").c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        cout << "couldn't open file: " << inputfile << endl;
        return -1;
    }
    
    // read c
    dataset_id = H5Dopen2(file_id, "/collective/c", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &c);
    status = H5Dclose(dataset_id);
    
    // read Lx
    dataset_id = H5Dopen2(file_id, "/collective/Lx", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Lx);
    status = H5Dclose(dataset_id);
    // read Ly
    dataset_id = H5Dopen2(file_id, "/collective/Ly", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Ly);
    status = H5Dclose(dataset_id);
    // read Lz
    dataset_id = H5Dopen2(file_id, "/collective/Lz", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Lz);
    status = H5Dclose(dataset_id);
    // read x_center
    dataset_id = H5Dopen2(file_id, "/collective/x_center", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &x_center);
    status = H5Dclose(dataset_id);
    // read y_center
    dataset_id = H5Dopen2(file_id, "/collective/y_center", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &y_center);
    status = H5Dclose(dataset_id);
    // read z_center
    dataset_id = H5Dopen2(file_id, "/collective/z_center", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &z_center);
    status = H5Dclose(dataset_id);
    // read L_square
    dataset_id = H5Dopen2(file_id, "/collective/L_square", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &L_square);
    status = H5Dclose(dataset_id);
    // read nxc
    dataset_id = H5Dopen2(file_id, "/collective/Nxc", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nxc);
    status = H5Dclose(dataset_id);
    // read nyc
    dataset_id = H5Dopen2(file_id, "/collective/Nyc", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nyc);
    status = H5Dclose(dataset_id);
    // read nyc
    dataset_id = H5Dopen2(file_id, "/collective/Nzc", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nzc);
    status = H5Dclose(dataset_id);
    // read ns
    dataset_id = H5Dopen2(file_id, "/collective/Ns", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ns);
    status = H5Dclose(dataset_id);
    
    
    /*! Boundary condition information */
    // read EMfaceXleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/EMfaceXleft", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcEMfaceXleft);
    status = H5Dclose(dataset_id);
    // read EMfaceXright
    dataset_id = H5Dopen2(file_id, "/collective/bc/EMfaceXright", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcEMfaceXright);
    status = H5Dclose(dataset_id);
    // read EMfaceYleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/EMfaceYleft", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcEMfaceYleft);
    status = H5Dclose(dataset_id);
    // read EMfaceYright
    dataset_id = H5Dopen2(file_id, "/collective/bc/EMfaceYright", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcEMfaceYright);
    status = H5Dclose(dataset_id);
    // read EMfaceZleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/EMfaceZleft", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcEMfaceZleft);
    status = H5Dclose(dataset_id);
    // read EMfaceZright
    dataset_id = H5Dopen2(file_id, "/collective/bc/EMfaceZright", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcEMfaceZright);
    status = H5Dclose(dataset_id);
    
    // read PHIfaceXleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/PHIfaceXleft", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPHIfaceXleft);
    status = H5Dclose(dataset_id);
    // read PHIfaceXright
    dataset_id = H5Dopen2(file_id, "/collective/bc/PHIfaceXright", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPHIfaceXright);
    status = H5Dclose(dataset_id);
    // read PHIfaceYleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/PHIfaceYleft", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPHIfaceYleft);
    status = H5Dclose(dataset_id);
    // read PHIfaceYright
    dataset_id = H5Dopen2(file_id, "/collective/bc/PHIfaceYright", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPHIfaceYright);
    status = H5Dclose(dataset_id);
    // read PHIfaceZleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/PHIfaceZleft", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPHIfaceZleft);
    status = H5Dclose(dataset_id);
    // read PHIfaceZright
    dataset_id = H5Dopen2(file_id, "/collective/bc/PHIfaceZright", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPHIfaceZright);
    status = H5Dclose(dataset_id);
    
    // read PfaceXleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/PfaceXleft", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPfaceXleft);
    status = H5Dclose(dataset_id);
    // read PfaceXright
    dataset_id = H5Dopen2(file_id, "/collective/bc/PfaceXright", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPfaceXright);
    status = H5Dclose(dataset_id);
    // read PfaceYleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/PfaceYleft", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPfaceYleft);
    status = H5Dclose(dataset_id);
    // read PfaceYright
    dataset_id = H5Dopen2(file_id, "/collective/bc/PfaceYright", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPfaceYright);
    status = H5Dclose(dataset_id);
    // read PfaceZleft
    dataset_id = H5Dopen2(file_id, "/collective/bc/PfaceZleft", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPfaceZleft);
    status = H5Dclose(dataset_id);
    // read PfaceZright
    dataset_id = H5Dopen2(file_id, "/collective/bc/PfaceZright", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bcPfaceZright);
    status = H5Dclose(dataset_id);
    // allocate fields depending on species
    npcelx = new int[ns];
    npcely = new int[ns];
    npcelz = new int[ns];
    qom = new double[ns];
    uth = new double[ns];
    vth = new double[ns];
    wth = new double[ns];
    u0 = new double[ns];
    v0 = new double[ns];
    w0 = new double[ns];
    // read data from species0, species 1, species2,...
    string *name_species = new string[ns];
    stringstream *ss = new stringstream[ns];
    
    for (int i = 0; i < ns; i++) {
        ss[i] << i;
        name_species[i] = "/collective/species_" + ss[i].str() + "/";
    }
    // npcelx for different species
    for (int i = 0; i < ns; i++) {
        dataset_id = H5Dopen2(file_id, (name_species[i] + "Npcelx").c_str(), H5P_DEFAULT); // HDF 1.8.8
        status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &npcelx[i]);
        status = H5Dclose(dataset_id);
    }
    // npcely for different species
    for (int i = 0; i < ns; i++) {
        dataset_id = H5Dopen2(file_id, (name_species[i] + "Npcely").c_str(), H5P_DEFAULT); // HDF 1.8.8
        status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &npcely[i]);
        status = H5Dclose(dataset_id);
    }
    // npcelz for different species
    for (int i = 0; i < ns; i++) {
        dataset_id = H5Dopen2(file_id, (name_species[i] + "Npcelz").c_str(), H5P_DEFAULT); // HDF 1.8.8
        status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &npcelz[i]);
        status = H5Dclose(dataset_id);
    }
    // qom for different species
    for (int i = 0; i < ns; i++) {
        dataset_id = H5Dopen2(file_id, (name_species[i] + "qom").c_str(), H5P_DEFAULT);  // HDF 1.8.8
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &qom[i]);
        status = H5Dclose(dataset_id);
    }
    /*! not needed for restart * */
    for (int i = 0; i < ns; i++)
        uth[i] = 0.0;
    for (int i = 0; i < ns; i++)
        vth[i] = 0.0;
    for (int i = 0; i < ns; i++)
        wth[i] = 0.0;
    for (int i = 0; i < ns; i++)
        u0[i] = 0.0;
    for (int i = 0; i < ns; i++)
        v0[i] = 0.0;
    for (int i = 0; i < ns; i++)
        w0[i] = 0.0;
    // verbose on
    verbose = 1;
    
    
    // if RestartDirName == SaveDirName overwrite dt,Th,Smooth (append to old hdf files)
    if (RestartDirName == SaveDirName) {
        restart_status = 2;
        // read dt
        dataset_id = H5Dopen2(file_id, "/collective/Dt", H5P_DEFAULT); // HDF 1.8.8
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dt);
        status = H5Dclose(dataset_id);
        // read th
        dataset_id = H5Dopen2(file_id, "/collective/Th", H5P_DEFAULT); // HDF 1.8.8
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &th);
        status = H5Dclose(dataset_id);
        // read Smooth
        dataset_id = H5Dopen2(file_id, "/collective/Smooth", H5P_DEFAULT); // HDF 1.8.8
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Smooth);
        status = H5Dclose(dataset_id);
    }
    
    status = H5Fclose(file_id);
    
    
    // read last cycle (not from settings, but from restart0.hdf)
    
    file_id = H5Fopen((inputfile + "/restart0.hdf").c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        cout << "couldn't open file: " << inputfile << endl;
        return -1;
    }
    
    dataset_id = H5Dopen2(file_id, "/last_cycle", H5P_DEFAULT);  // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &last_cycle);
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);
    
    // deallocate
    delete[]name_species;
    delete[]ss;
    
    return (0);
}
/*! constructor */
Collective::Collective(int argc, char **argv) {
    if (argc < 2) {
        inputfile = "inputfile";
        RESTART1 = false;
    }
    else if (argc < 3) {
        inputfile = argv[1];
        RESTART1 = false;
    }
    else {
        if (strcmp(argv[1], "restart") == 0) {
            inputfile = argv[2];
            RESTART1 = true;
        }
        else if (strcmp(argv[2], "restart") == 0) {
            inputfile = argv[1];
            RESTART1 = true;
        }
        else {
            cout << "Error: syntax error in mpirun arguments. Did you mean to 'restart' ?" << endl;
            return;
        }
    }
    ReadInput(inputfile);
    init_derived_parameters();
}

void Collective::init_derived_parameters()
{
    /*! fourpi = 4 greek pi */
    fourpi = 16.0 * atan(1.0);
    /*! dx = space step - X direction */
    dx = Lx / (double) nxc;
    /*! dy = space step - Y direction */
    dy = Ly / (double) nyc;
    /*! dz = space step - Z direction */
    dz = Lz / (double) nzc;
    /*! npcel = number of particles per cell */
    npcel = new int[ns];
    /*! np = number of particles of different species */
    //np = new int[ns];
    /*! npMax = maximum number of particles of different species */
    //npMax = new int[ns];
    
    /* quantities per process */
    
    // check that procs divides grid
    // (this restriction should be removed).
    //
    if(0==MPIdata::get_rank())
    {
        fflush(stdout);
        bool xerror = false;
        bool yerror = false;
        bool zerror = false;
        if(nxc % XLEN) xerror=true;
        if(nyc % YLEN) yerror=true;
        if(nzc % ZLEN) zerror=true;
        if(xerror) printf("!!!ERROR: XLEN=%d does not divide nxc=%d\n", XLEN,nxc);
        if(yerror) printf("!!!ERROR: YLEN=%d does not divide nyc=%d\n", YLEN,nyc);
        if(zerror) printf("!!!ERROR: ZLEN=%d does not divide nzc=%d\n", ZLEN,nzc);
        fflush(stdout);
        bool error = xerror||yerror||zerror;
        if(error) exit(1);
    }
    
    int num_cells_r = nxc*nyc*nzc;
    //num_procs = XLEN*YLEN*ZLEN;
    //ncells_rs = nxc_rs*nyc_rs*nzc_rs;
    
    for (int i = 0; i < ns; i++)
    {
        npcel[i] = npcelx[i] * npcely[i] * npcelz[i];
        //np[i] = npcel[i] * num_cells;
        //nop_rs[i] = npcel[i] * ncells_rs;
        //maxnop_rs[i] = NpMaxNpRatio * nop_rs[i];
        // INT_MAX is about 2 billion, surely enough
        // to index the particles in a single MPI process:
        //assert_le(NpMaxNpRatio * npcel[i] * ncells_proper_per_proc , double(INT_MAX));
        //double npMaxi = (NpMaxNpRatio * np[i]);
        //npMax[i] = (int) npMaxi;
    }
}

/*! destructor */
Collective::~Collective() {
    //delete[]np;
    delete[]npcel;
    delete[]npcelx;
    delete[]npcely;
    delete[]npcelz;
    //delete[]npMax;
    delete[]qom;
    
    delete[]uth;
    delete[]vth;
    delete[]wth;
    
    delete[]u0;
    delete[]v0;
    delete[]w0;
    
    delete[]TrackParticleID;
    
    delete[]rhoINIT;
    delete[]rhoINJECT;
    
}
/*! Print Simulation Parameters */
void Collective::Print() {
    cout << endl;
    cout << "Simulation Parameters" << endl;
    cout << "---------------------" << endl;
    cout << "Number of species    = " << ns << endl;
    for (int i = 0; i < ns; i++)
        cout << "qom[" << i << "] = " << qom[i] << endl;
    cout << "x-Length                 = " << Lx << endl;
    cout << "y-Length                 = " << Ly << endl;
    cout << "z-Length                 = " << Lz << endl;
    cout << "Number of cells (x)      = " << nxc << endl;
    cout << "Number of cells (y)      = " << nyc << endl;
    cout << "Number of cells (z)      = " << nzc << endl;
    cout << "Time step                = " << dt << endl;
    cout << "Number of cycles         = " << ncycles << endl;
    cout << "Results saved in  : " << SaveDirName << endl;
    cout << "Case type         : " << Case << endl;
    cout << "Simulation name   : " << SimName << endl;
    cout << "Poisson correction: " << PoissonCorrection << endl;
    cout << "Accuracy Constraint:  " << endl;
    for (int i = 0; i < ns; i++) {
        cout << "u_th dx/dt species " << i << "....." << (uth[i] * (dx / dt)) << endl;
    }
    
  
    
    
}
/*! Print Simulation Parameters */
void Collective::save() {
    string temp;
    temp = SaveDirName + "/SimulationData.txt";
    ofstream my_file(temp.c_str());
    my_file << "---------------------------" << endl;
    my_file << "-  Simulation Parameters  -" << endl;
    my_file << "---------------------------" << endl;
    
    my_file << "Number of species    = " << ns << endl;
    for (int i = 0; i < ns; i++)
        my_file << "qom[" << i << "] = " << qom[i] << endl;
    my_file << "---------------------------" << endl;
    my_file << "x-Length                 = " << Lx << endl;
    my_file << "y-Length                 = " << Ly << endl;
    my_file << "z-Length                 = " << Lz << endl;
    my_file << "Number of cells (x)      = " << nxc << endl;
    my_file << "Number of cells (y)      = " << nyc << endl;
    my_file << "Number of cells (z)      = " << nzc << endl;
    my_file << "---------------------------" << endl;
    my_file << "Time step                = " << dt << endl;
    my_file << "Number of cycles         = " << ncycles << endl;
    my_file << "---------------------------" << endl;
    for (int is = 0; is < ns; is++){
        my_file << "rho init species   " << is << " = " << rhoINIT[is] << endl;
        my_file << "rho inject species " << is << " = " << rhoINJECT[is]  << endl;
    }
    my_file << "current sheet thickness  = " << delta << endl;
    my_file << "B0x                      = " << B0x << endl;
    my_file << "BOy                      = " << B0y << endl;
    my_file << "B0z                      = " << B0z << endl;
    my_file << "---------------------------" << endl;
    my_file << "Smooth                   = " << Smooth << endl;
    my_file << "GMRES error tolerance    = " << GMREStol << endl;
    my_file << "CG error tolerance       = " << CGtol << endl;
    my_file << "Mover error tolerance    = " << NiterMover << endl;
    my_file << "---------------------------" << endl;
    my_file << "Results saved in: " << SaveDirName << endl;
    my_file << "Restart saved in: " << RestartDirName << endl;
    my_file << "---------------------" << endl;
    my_file.close();
    
}
