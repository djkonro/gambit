//
// FILE: enum.cc -- Nash Enum module
//
// $Id$
//

#include "enum.imp"

//---------------------------------------------------------------------------
//                        EnumParams: member functions
//---------------------------------------------------------------------------

EnumParams::EnumParams(gStatus &status_)
  : trace(0), stopAfter(0), precision(precDOUBLE),
    tracefile(&gnull), status(status_)
{ }

int Enum(const NFSupport &support, const EnumParams &params,
	 gList<MixedSolution> &solutions, long &npivots, double &time)
{
  if (params.precision == precDOUBLE)  {
    EnumModule<double> module(support.Game(), params, support);
    module.Enum();
    npivots = module.NumPivots();
    time = module.Time();
    solutions = module.GetSolutions();
  }
  else if (params.precision == precRATIONAL)  {
    EnumModule<gRational> module(support.Game(), params, support);
    module.Enum();
    npivots = module.NumPivots();
    time = module.Time();
    solutions = module.GetSolutions();
  }

  return 1;
}

#include "rational.h"

template class EnumModule<double>;
template class EnumModule<gRational>;











