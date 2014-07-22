//
// This file is part of Gambit
// Copyright (c) 1994-2014, The Gambit Project (http://www.gambit-project.org)
//
// FILE: src/tools/liap/nfgliap.cc
// Compute Nash equilibria by minimizing Liapunov function
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//

#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "libgambit/libgambit.h"
#include "funcmin.h"

using namespace Gambit;

extern int m_stopAfter;
extern int m_numTries;
extern int m_maxits1;
extern int m_maxitsN;
extern double m_tol1;
extern double m_tolN;
extern std::string startFile;
extern bool useRandom;
extern int g_numDecimals;
extern bool verbose;

//---------------------------------------------------------------------
//                        class NFLiapFunc
//---------------------------------------------------------------------

class StrategicLyapunovFunction : public Function  {
private:
  Game m_game;
  mutable MixedStrategyProfile<double> m_profile;

  double Value(const Vector<double> &) const;
  bool Gradient(const Vector<double> &, Vector<double> &) const;

  double LiapDerivValue(int, int, const MixedStrategyProfile<double> &) const;
    
public:
  StrategicLyapunovFunction(const MixedStrategyProfile<double> &);
  virtual ~StrategicLyapunovFunction() { }
};

StrategicLyapunovFunction::StrategicLyapunovFunction(const MixedStrategyProfile<double> &p_start)
  : m_game(p_start.GetGame()), m_profile(p_start)
{ }

double StrategicLyapunovFunction::LiapDerivValue(int i1, int j1,
						 const MixedStrategyProfile<double> &p) const
{
  double x = 0.0;
  for (int i = 1; i <= m_game->NumPlayers(); i++)  {
    double psum = 0.0;
    for (int j = 1; j <= p.GetSupport().NumStrategies(i); j++)  {
      psum += p[p.GetSupport().GetStrategy(i,j)];
      double x1 = p.GetPayoff(p.GetSupport().GetStrategy(i, j)) - p.GetPayoff(i);
      if (i1 == i) {
	if (x1 > 0.0)
	  x -= x1 * p.GetPayoffDeriv(i, p.GetSupport().GetStrategy(i1, j1));
      }
      else {
	if (x1> 0.0)
	  x += x1 * (p.GetPayoffDeriv(i, 
				      p.GetSupport().GetStrategy(i, j),
				      p.GetSupport().GetStrategy(i1, j1)) - 
		     p.GetPayoffDeriv(i,
				      p.GetSupport().GetStrategy(i1, j1)));
      }
    }
    if (i == i1)  x += 100.0 * (psum - 1.0);
  }
  if (p[p.GetSupport().GetStrategy(i1, j1)] < 0.0) {
    x += p[p.GetSupport().GetStrategy(i1, j1)];
  }
  return 2.0 * x;
}

//
// This function projects a gradient into the plane of the simplex.
// (Actually, it works by computing the projection of 'x' onto the
// vector perpendicular to the plane, then subtracting to compute the
// component parallel to the plane.)
//
static void Project(Vector<double> &x, const Array<int> &lengths)
{
  int index = 1;
  for (int part = 1; part <= lengths.Length(); part++)  {
    double avg = 0.0;
    int j;
    for (j = 1; j <= lengths[part]; j++, index++)  {
      avg += x[index];
    }
    avg /= (double) lengths[part];
    index -= lengths[part];
    for (j = 1; j <= lengths[part]; j++, index++)  {
      x[index] -= avg;
    }
  }
}

bool StrategicLyapunovFunction::Gradient(const Vector<double> &v, Vector<double> &d) const
{
  static_cast<Vector<double> &>(m_profile).operator=(v);
  for (int pl = 1, ii = 1; pl <= m_game->NumPlayers(); pl++) {
    for (int st = 1; st <= m_game->Players()[pl]->Strategies().size(); st++) {
      d[ii++] = LiapDerivValue(pl, st, m_profile);
    }
  }
  Project(d, m_game->NumStrategies());
  return true;
}
  
double StrategicLyapunovFunction::Value(const Vector<double> &v) const
{
  static_cast<Vector<double> &>(m_profile).operator=(v);
  return m_profile.GetLiapValue();
}

void PrintProfile(std::ostream &p_stream,
		  const std::string &p_label,
		  const MixedStrategyProfile<double> &p_profile)
{
  p_stream << p_label;
  for (int i = 1; i <= p_profile.MixedProfileLength(); i++) {
    p_stream.setf(std::ios::fixed);
    p_stream << ", " << std::setprecision(g_numDecimals) << p_profile[i];
  }

  p_stream << std::endl;
}

bool ReadProfile(std::istream &p_stream,
		 MixedStrategyProfile<double> &p_profile)
{
  for (int i = 1; i <= p_profile.MixedProfileLength(); i++) {
    if (p_stream.eof() || p_stream.bad()) {
      return false;
    }

    p_stream >> p_profile[i];
    if (i < p_profile.MixedProfileLength()) {
      char comma;
      p_stream >> comma;
    }
  }

  // Read in the rest of the line and discard
  std::string foo;
  std::getline(p_stream, foo);
  return true;
}

extern std::string startFile;

void SolveStrategic(const Game &p_game)
{
  List<MixedStrategyProfile<double> > starts;

  if (startFile != "") {
    std::ifstream startPoints(startFile.c_str());

    while (!startPoints.eof() && !startPoints.bad()) {
      MixedStrategyProfile<double> start(p_game->NewMixedStrategyProfile(0.0));
      if (ReadProfile(startPoints, start)) {
	starts.Append(start);
      }
    }
  }
  else {
    // Generate the desired number of points randomly
    for (int i = 1; i <= m_numTries; i++) {
      MixedStrategyProfile<double> start(p_game->NewMixedStrategyProfile(0.0));
      start.Randomize();
      starts.Append(start);
    }
  }

  static const double ALPHA = .00000001;

  for (int i = 1; i <= starts.Length(); i++) {
    MixedStrategyProfile<double> p(starts[i]);

    if (verbose) {
      PrintProfile(std::cout, "start", p);
    }

    StrategicLyapunovFunction F(p);

    // if starting vector not interior, perturb it towards centroid
    int kk;
    for (kk = 1; kk <= p.MixedProfileLength() && p[kk] > ALPHA; kk++);
    if (kk <= p.MixedProfileLength()) {
      MixedStrategyProfile<double> centroid(p.GetSupport().NewMixedStrategyProfile<double>());
      for (int k = 1; k <= p.MixedProfileLength(); k++) {
	p[k] = centroid[k] * ALPHA + p[k] * (1.0-ALPHA);
      }
    }

    ConjugatePRMinimizer minimizer(p.MixedProfileLength());
    Vector<double> gradient(p.MixedProfileLength()), dx(p.MixedProfileLength());
    double fval;
    minimizer.Set(F, (const Vector<double> &) p,
		  fval, gradient, .01, .0001);

    for (int iter = 1; iter <= m_maxitsN; iter++) {
      if (!minimizer.Iterate(F, (Vector<double> &) p, 
			     fval, gradient, dx)) {
	break;
      }

      if (sqrt(gradient.NormSquared()) < .001) {
	PrintProfile(std::cout, "NE", p);
	break;
      }
    }

    if (verbose && sqrt(gradient.NormSquared()) >= .001) {
      PrintProfile(std::cout, "end", p);
    }
  }
}




