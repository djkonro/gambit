//
// This file is part of Gambit
// Copyright (c) 1994-2014, The Gambit Project (http://www.gambit-project.org)
//
// FILE: src/tools/liap/efgliap.cc
// Compute Nash equilibria via Lyapunov function minimization
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

class AgentLyapunovFunction : public Function  {
private:
  Game m_game;
  mutable MixedBehavProfile<double> m_profile;

  double Value(const Vector<double> &x) const;
  bool Gradient(const Vector<double> &, Vector<double> &) const;

public:
  AgentLyapunovFunction(const MixedBehavProfile<double> &);
  virtual ~AgentLyapunovFunction() { }
};


AgentLyapunovFunction::AgentLyapunovFunction(const MixedBehavProfile<double> &p_start)
  : m_game(p_start.GetGame()), m_profile(p_start)
{ }

double AgentLyapunovFunction::Value(const Vector<double> &v) const
{
  static_cast<Vector<double> &>(m_profile).operator=(v);
  return m_profile.GetLiapValue();
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

bool AgentLyapunovFunction::Gradient(const Vector<double> &x,
				     Vector<double> &grad) const
{
  const double DELTA = .00001;
  static_cast<Vector<double> &>(m_profile).operator=(x);
  for (int i = 1; i <= x.Length(); i++) {
    m_profile[i] += DELTA;
    double value = m_profile.GetLiapValue();
    m_profile[i] -= 2.0 * DELTA;
    value -= m_profile.GetLiapValue();
    m_profile[i] += DELTA;
    grad[i] = value / (2.0 * DELTA);
  }
  Project(grad, m_game->NumInfosets());
  return true;
}

void PrintProfile(std::ostream &p_stream,
		  const std::string &p_label,
		  const MixedBehavProfile<double> &p_profile)
{
  p_stream << p_label;
  for (int i = 1; i <= p_profile.Length(); i++) {
    p_stream.setf(std::ios::fixed);
    p_stream << ", " << std::setprecision(g_numDecimals) << p_profile[i];
  }

  p_stream << std::endl;
}

bool ReadProfile(std::istream &p_stream,
		 MixedBehavProfile<double> &p_profile)
{
  for (int i = 1; i <= p_profile.Length(); i++) {
    if (p_stream.eof() || p_stream.bad()) {
      return false;
    }

    p_stream >> p_profile[i];
    if (i < p_profile.Length()) {
      char comma;
      p_stream >> comma;
    }
  }

  // Read in the rest of the line and discard
  std::string foo;
  std::getline(p_stream, foo);
  return true;
}

void SolveExtensive(const Game &p_game)
{
  List<MixedBehavProfile<double> > starts;

  if (startFile != "") {
    std::ifstream startPoints(startFile.c_str());

    while (!startPoints.eof() && !startPoints.bad()) {
      MixedBehavProfile<double> start(p_game);
      if (ReadProfile(startPoints, start)) {
	starts.Append(start);
      }
    }
  }
  else {
    // Generate the desired number of points randomly
    for (int i = 1; i <= m_numTries; i++) {
      MixedBehavProfile<double> start(p_game);
      start.Randomize();
      starts.Append(start);
    }
  }

  static const double ALPHA = .00000001;

  for (int i = 1; i <= starts.Length(); i++) {
    MixedBehavProfile<double> p(starts[i]);

    if (verbose) {
      PrintProfile(std::cout, "start", p);
    }

    AgentLyapunovFunction F(p);

    // if starting vector not interior, perturb it towards centroid
    int kk = 1;
    for (; kk <= p.Length() && p[kk] > ALPHA; kk++);
    if (kk <= p.Length()) {
      MixedBehavProfile<double> c(p_game);
      for (int k = 1; k <= p.Length(); k++) {
	p[k] = c[k]*ALPHA + p[k]*(1.0-ALPHA);
      }
    }

    Matrix<double> xi(p.Length(), p.Length());
  
    ConjugatePRMinimizer minimizer(p.Length());
    Vector<double> gradient(p.Length()), dx(p.Length());
    double fval;
    minimizer.Set(F, p, fval, gradient, .01, .0001);

    for (int iter = 1; iter <= m_maxitsN; iter++) {
      if (!minimizer.Iterate(F, p, fval, gradient, dx)) {
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

