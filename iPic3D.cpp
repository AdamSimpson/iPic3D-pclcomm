
#include <mpi.h>
#include <iomanip>
#include "iPic3D.h"
#include "debug.h"
#include "TimeTasks.h"

using namespace iPic3D;

int main(int argc, char **argv) {

 MPIdata::init(&argc, &argv);
 {
  iPic3D::c_Solver KCode;
  bool b_err = false;

  KCode.Init(argc, argv);

  timeTasks.resetCycle();
  KCode.CalculateMoments();
  for (int i = KCode.FirstCycle(); i < KCode.LastCycle(); i++) {

    if (KCode.get_myrank() == 0) cout << " ======= Cycle " << i << " ======= " << endl;

    if (!b_err) {
      timeTasks.resetCycle();
      KCode.CalculateField();
      b_err = KCode.ParticlesMover();
      KCode.CalculateB();
      KCode.CalculateMoments();

      // print out total time for all tasks
      timeTasks.print_cycle_times(i);
    }

    if (b_err) {
      i = KCode.LastCycle() + 1;
    }

    KCode.WriteOutput(i);
  }

  KCode.Finalize();
 }
 // close MPI
 MPIdata::instance().finalize_mpi();

 return 0;
}
