#include "Parameters.h"

using namespace Parameters;

//********** edit these parameters *********
//
bool Parameters::get_VECTORIZE_MOMENTS() { return false; }
// supported options: SoA AoS
Parameters::Enum Parameters::get_MOMENTS_TYPE() { return AoS; }
// supported options: SoA AoS AoSvec AoSintr AoS_vec_onesort SoA_vec_resort
Parameters::Enum Parameters::get_MOVER_TYPE() { return AoS; }
//********** derived parameters *********

static bool SORTING_PARTICLES;
static bool RESORTING_PARTICLES;
static bool USING_AOS;
static bool SORTING_SOA;

void Parameters::init_parameters()
{
  RESORTING_PARTICLES = 
       get_MOVER_TYPE()==SoA_vec_resort
    || get_MOVER_TYPE()==AoS_vec_resort;
  SORTING_PARTICLES = get_VECTORIZE_MOMENTS()
    || get_MOVER_TYPE()==SoA_vec_onesort
    || get_MOVER_TYPE()==AoS_vec_onesort
    || get_MOVER_TYPE()==SoA_vec_resort
    || get_MOVER_TYPE()==AoS_vec_resort;
  SORTING_SOA = get_VECTORIZE_MOMENTS()
    || get_MOVER_TYPE()==SoA_vec_onesort
    || get_MOVER_TYPE()==SoA_vec_resort;
  USING_AOS =
       get_MOMENTS_TYPE()==AoS
    || get_MOVER_TYPE()==AoS
    || get_MOVER_TYPE()==AoSintr
    || get_MOVER_TYPE()==AoS_vec_onesort
    || get_MOVER_TYPE()==AoS_vec_resort;
}

bool Parameters::get_RESORTING_PARTICLES() { return RESORTING_PARTICLES; }
bool Parameters::get_SORTING_PARTICLES() { return SORTING_PARTICLES; }
bool Parameters::get_SORTING_SOA() { return SORTING_SOA; }
bool Parameters::get_USING_AOS() { return USING_AOS; }
// int Parameters::get_blockSize() { return 64; }
int Parameters::get_blockSize() { return 2048; }
int Parameters::get_numBlocks() { return 4; }

//bool Parameters::get_RESORTING_PARTICLES() { return true; }
//bool Parameters::get_SORTING_PARTICLES() { return true; }
//bool Parameters::get_SORTING_SOA() { return true; }
//bool Parameters::get_USING_AOS() { return true; }
//
