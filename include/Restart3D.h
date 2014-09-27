// Developed by Stefano Markidis, and Giovanni Lapenta
#ifndef _RESTART3D_H_
#define _RESTART3D_H_

#include "ipicfwd.h"
#include "Particles3Dcomm.h"
#include "PSKhdf5adaptor.h"

#include <iosfwd>

using std::string;
using std::stringstream;


/** write the restart file at any RESTART_CYCLE, useful for reading intermediate results */
void writeRESTART(const string& SaveDirName, int myrank, int cycle, int ns, VCtopology3D * vct, Collective * col, Grid * grid, Field * field, Particles3Dcomm * part);

/** this restart function writes the last restart with the last cycle */
void writeRESTART(const string& SaveDirName, int myrank, int cycle, int ns, VCtopology3D * vct, Collective * col, Grid * grid, Field * field, Particles3Dcomm * part, bool fool);

/** write the restart file at any RESTART_CYCLE, useful for reading intermediate results */
void writeRESTART_ES(const string& SaveDirName, int myrank, int cycle, int ns, VCtopology3D * vct, Collective * col, Grid * grid, Field * field, Particles * part);

/** this restart function writes the last restart with the last cycle */
void writeRESTART_ES(const string& SaveDirName, int myrank, int cycle, int ns, VCtopology3D * vct, Collective * col, Grid * grid, Field * field, Particles * part, bool fool);

#endif // _RESTART_H_
