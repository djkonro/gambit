//
// FILE: algfunc.cc -- Solution algorithm functions for GCL
//
// $Id$
//

#include "gsm.h"
#include "portion.h"
#include "gsmfunc.h"

#include "rational.h"

#include "gwatch.h"
#include "mixedsol.h"
#include "behavsol.h"
#include "nfg.h"
#include "nfplayer.h"
#include "efg.h"

#include "vertenum.h"


extern Portion *ArrayToList(const gArray<int> &A);
extern Portion *ArrayToList(const gArray<double> &A);
extern Portion *ArrayToList(const gArray<gRational> &A);
extern gVector<double>* ListToVector_Float(ListPortion* list);
extern gVector<gRational>* ListToVector_Rational(ListPortion* list);
extern gMatrix<double>* ListToMatrix_Float(ListPortion* list);
extern gMatrix<gRational>* ListToMatrix_Rational(ListPortion* list);


//
// Useful utilities for creation of lists of profiles
//
class Mixed_ListPortion : public ListValPortion   {
  public:
    Mixed_ListPortion(const gList<MixedSolution> &);
    virtual ~Mixed_ListPortion()   { }
};

Mixed_ListPortion::Mixed_ListPortion(const gList<MixedSolution> &list)
{
  _DataType = porMIXED;
  for (int i = 1; i <= list.Length(); i++)
    Append(new MixedPortion(new MixedSolution(list[i])));
}

class Behav_ListPortion : public ListValPortion   {
  public:
    Behav_ListPortion(const gList<BehavSolution> &);
    virtual ~Behav_ListPortion()   { }
};

Behav_ListPortion::Behav_ListPortion(const gList<BehavSolution> &list)
{
  _DataType = porBEHAV;
  for (int i = 1; i <= list.Length(); i++)
    Append(new BehavPortion(new BehavSolution(list[i])));
}


//-------------
// AgentForm
//-------------


static Portion *GSM_AgentForm(Portion **param)
{
  Efg &E = *((EfgPortion*) param[0])->Value();
  gWatch watch;

  Nfg *N = MakeAfg(E);
  
  ((NumberPortion *) param[1])->Value() = watch.Elapsed();

  if (N)
    return new NfgValPortion(N);
  else
    return new ErrorPortion("Conversion to agent form failed");
}

//------------
// Behav
//------------

static Portion *GSM_Behav(Portion **param)
{
  MixedSolution &mp = *((MixedPortion*) param[0])->Value();

  Nfg &N = mp.Game();
  const Efg &E = *(const Efg *) N.AssociatedEfg();

  BehavProfile<gNumber> *bp = new BehavProfile<gNumber>(E);
  MixedToBehav(N, mp, E, *bp);

  return new BehavPortion(new BehavSolution(*bp));
}


//------------------
// EnumMixedSolve
//------------------

#include "enum.h"

static Portion *GSM_EnumMixed_Nfg(Portion **param)
{
  NFSupport* S = ((NfSupportPortion*) param[0])->Value();

  EnumParams params;
  params.stopAfter = ((IntPortion *) param[1])->Value();
  params.precision = ((PrecisionPortion *) param[2])->Value();
  params.tracefile = &((OutputPortion *) param[5])->Value();
  params.trace = ((IntPortion *) param[6])->Value();
  
  gList<MixedSolution> solutions;
  double time; 
  Enum(*S, params, solutions,
       ((IntPortion *) param[3])->Value(), time);
  ((NumberPortion *) param[4])->Value() = time;
  return new Mixed_ListPortion(solutions);
}

#include "enumsub.h"


static Portion *GSM_EnumMixed_Efg(Portion **param)
{
  EFSupport &support = *((EfSupportPortion *) param[0])->Value();

  if (!((BoolPortion *) param[1])->Value())
    return new ErrorPortion("algorithm not implemented for extensive forms");

  EnumParams params;
  params.stopAfter = ((IntPortion *) param[2])->Value();
  params.precision = ((PrecisionPortion *) param[3])->Value();
  params.tracefile = &((OutputPortion *) param[6])->Value();
  params.trace = ((IntPortion *) param[7])->Value();

  double time;
  gList<BehavSolution> solutions;
  Enum(support, params, solutions, 
       ((IntPortion *) param[4])->Value(), time);
  ((NumberPortion *) param[5])->Value() = time;

  return new Behav_ListPortion(solutions);
}


//-----------------
// EnumPureSolve
//-----------------

#include "nfgpure.h"

static Portion *GSM_EnumPure_Nfg(Portion **param)
{
  NFSupport* S = ((NfSupportPortion*) param[0])->Value();

  gWatch watch;

  gList<MixedSolution> solns;
  FindPureNash(S->Game(), *S, solns);
  ((NumberPortion *) param[2])->Value() = watch.Elapsed();
  return new Mixed_ListPortion(solns);
}

#include "efgpure.h"
#include "psnesub.h"

static Portion *GSM_EnumPure_Efg(Portion **param)
{
  EFSupport &support = *((EfSupportPortion *) param[0])->Value();

  if (((BoolPortion *) param[1])->Value())   {
    gList<BehavSolution> solutions;
    double time;
    EnumPureNfg(support, solutions, time);
    ((NumberPortion *) param[4])->Value() = time;
    return new Behav_ListPortion(solutions);
  }
  else  {
    gList<BehavSolution> solutions;
    double time;
    EnumPure(support, solutions, time);
    ((NumberPortion *) param[4])->Value() = time;
    return new Behav_ListPortion(solutions);
  }
}

//------------------
// GobitGridSolve
//------------------

#include "grid.h"

static Portion *GSM_GobitGrid_Support(Portion **param)
{
  NFSupport& S = * ((NfSupportPortion*) param[0])->Value();

  GridParams GP;
  
  if(((TextPortion*) param[1])->Value() != "")
    GP.pxifile = new gFileOutput(((TextPortion*) param[1])->Value());
  else
    GP.pxifile = &gnull;
  GP.minLam = ((NumberPortion *) param[2])->Value();
  GP.maxLam = ((NumberPortion *) param[3])->Value();
  GP.delLam = ((NumberPortion *) param[4])->Value();
  GP.powLam = ((IntPortion *) param[5])->Value();
  GP.delp1 = ((NumberPortion *) param[6])->Value();
  GP.tol1 = ((NumberPortion *) param[7])->Value();

  GP.delp2 = ((NumberPortion *) param[8])->Value();
  GP.tol2 = ((NumberPortion *) param[9])->Value();

  GP.multi_grid = 0;
  if(GP.delp2 > 0.0 && GP.tol2 > 0.0)GP.multi_grid = 1;
  
  gList<MixedSolution> solutions;
  GridSolve(S, GP, solutions);

  if (GP.pxifile != &gnull)  delete GP.pxifile;

  return new Mixed_ListPortion(solutions);
}

//---------------
// GobitSolve
//---------------

#include "ngobit.h"
#include "egobit.h"

static Portion *GSM_Gobit_Start(Portion **param)
{
  if (param[0]->Spec().Type == porMIXED)  {
    MixedSolution &start = *((MixedPortion *) param[0])->Value();
    Nfg &N = start.Game();

    NFGobitParams NP;
    if (((TextPortion *) param[1])->Value() != "")
      NP.pxifile = new gFileOutput(((TextPortion *) param[1])->Value());
    else
      NP.pxifile = &gnull;
    NP.minLam = ((NumberPortion *) param[2])->Value();
    NP.maxLam = ((NumberPortion *) param[3])->Value();
    NP.delLam = ((NumberPortion *) param[4])->Value();
    NP.powLam = ((IntPortion *) param[5])->Value();
    NP.fullGraph = ((BoolPortion *) param[6])->Value();

    NP.maxitsN = ((IntPortion *) param[7])->Value();
    NP.tolN = ((NumberPortion *) param[8])->Value();
    NP.maxits1 = ((IntPortion *) param[9])->Value();
    NP.tol1 = ((NumberPortion *) param[10])->Value();
    
    NP.tracefile = &((OutputPortion *) param[14])->Value();
    NP.trace = ((IntPortion *) param[15])->Value();

    gWatch watch;
    gList<MixedSolution> solutions;
    Gobit(N, NP, start, solutions, 
	  ((IntPortion *) param[12])->Value(),
	  ((IntPortion *) param[13])->Value());

    ((NumberPortion *) param[11])->Value() = watch.Elapsed();

    Portion *por = new Mixed_ListPortion(solutions);

    if (NP.pxifile != &gnull)  delete NP.pxifile;
    return por;
  }
  else  {     // BEHAV_FLOAT  
    BehavSolution &start = *((BehavPortion *) param[0])->Value();
    Efg &E = start.Game();
  
    EFGobitParams EP;
    if(((TextPortion*) param[1])->Value() != "")
      EP.pxifile = new gFileOutput(((TextPortion*) param[1])->Value());
    else
      EP.pxifile = &gnull;
    EP.minLam = ((NumberPortion *) param[2])->Value();
    EP.maxLam = ((NumberPortion *) param[3])->Value();
    EP.delLam = ((NumberPortion *) param[4])->Value();
    EP.powLam = ((IntPortion *) param[5])->Value();
    EP.fullGraph = ((BoolPortion *) param[6])->Value();
    
    EP.maxitsN = ((IntPortion *) param[7])->Value();
    EP.tolN = ((NumberPortion *) param[8])->Value();
    EP.maxits1 = ((IntPortion *) param[9])->Value();
    EP.tol1 = ((NumberPortion *) param[10])->Value();
  
    EP.tracefile = &((OutputPortion *) param[14])->Value();
    EP.trace = ((IntPortion *) param[15])->Value();
    
    gWatch watch;
    
    gList<BehavSolution> solutions;
    Gobit(E, EP, start, solutions,
	  ((IntPortion *) param[12])->Value(),
	  ((IntPortion *) param[13])->Value());
    
    ((NumberPortion *) param[11])->Value() = watch.Elapsed();
    
    Portion * por = new Behav_ListPortion(solutions);

    if (EP.pxifile != &gnull)   delete EP.pxifile;
    return por;
  }
}


//------------
// LcpSolve
//------------

#include "lemke.h"

static Portion *GSM_Lcp_Nfg(Portion **param)
{
  NFSupport& S = * ((NfSupportPortion*) param[0])->Value();

  LemkeParams params;
  params.stopAfter = ((IntPortion *) param[1])->Value();
  params.precision = ((PrecisionPortion *) param[2])->Value();
  params.tracefile = &((OutputPortion *) param[5])->Value();
  params.trace = ((IntPortion *) param[6])->Value();

  gList<MixedSolution> solutions;
  double time;
  int npivots;
  Lemke(S, params, solutions, npivots, time);
  ((IntPortion *) param[3])->Value() = npivots;
  ((NumberPortion *) param[4])->Value() = time;

  return new Mixed_ListPortion(solutions);
}


#include "lemketab.h"

Portion* GSM_Lcp_ListFloat(Portion** param)
{
  gMatrix<double>* a = ListToMatrix_Float((ListPortion*) param[0]);
  gVector<double>* b = ListToVector_Float((ListPortion*) param[1]);
  
  LTableau<double>* tab = new LTableau<double>(*a, *b);
  tab->LemkePath(0);
  gVector<double> vector;
  tab->BasisVector(vector);
  Portion* result = ArrayToList(vector);
  delete tab;
  delete a;
  delete b;
  
  return result;
}

Portion* GSM_Lcp_ListRational(Portion** param)
{
  gMatrix<gRational>* a = ListToMatrix_Rational((ListPortion*) param[0]);
  gVector<gRational>* b = ListToVector_Rational((ListPortion*) param[1]);
  
  LTableau<gRational>* tab = new LTableau<gRational>(*a, *b);
  tab->LemkePath(0);
  gVector<gRational> vector;
  tab->BasisVector(vector);
  Portion* result = ArrayToList(vector);
  delete tab;
  delete a;
  delete b;
  
  return result;
}



#include "seqform.h"
#include "lemkesub.h"

static Portion *GSM_Lcp_Efg(Portion **param)
{
  EFSupport& S = *((EfSupportPortion*) param[0])->Value();

  if (((BoolPortion *) param[1])->Value())   {
    LemkeParams params;
    
    params.stopAfter = ((IntPortion *) param[2])->Value();
    params.precision = ((PrecisionPortion *) param[3])->Value();
    params.tracefile = &((OutputPortion *) param[6])->Value();
    params.trace = ((IntPortion *) param[7])->Value();

    gList<BehavSolution> solutions;
    double time;
    int npivots;

    Lemke(S, params, solutions, npivots, time);
    ((IntPortion *) param[4])->Value() = npivots;
    ((NumberPortion *) param[5])->Value() = time;
    return new Behav_ListPortion(solutions);
  }
  else  {
    SeqFormParams params;

    params.stopAfter = ((IntPortion *) param[2])->Value();
    params.precision = ((PrecisionPortion *) param[3])->Value();
    params.tracefile = &((OutputPortion *) param[6])->Value();
    params.trace = ((IntPortion *) param[7])->Value();

    gList<BehavSolution> solutions;
    double time;
    int npivots;

    SeqForm(S, params, solutions, npivots, time);
    ((IntPortion *) param[4])->Value() = npivots;
    ((NumberPortion *) param[5])->Value() = time;
    return new Behav_ListPortion(solutions);
  }
}


//-------------
// LiapSolve
//-------------

#include "liapsub.h"
#include "eliap.h"

static Portion *GSM_Liap_Behav(Portion **param)
{
  BehavProfile<gNumber> &start = *((BehavPortion *) param[0])->Value();
  Efg &E = start.Game();
  
  if (((BoolPortion *) param[1])->Value())   {
    NFLiapParams LP;

    LP.stopAfter = ((IntPortion *) param[2])->Value();
    LP.nTries = ((IntPortion *) param[3])->Value();

    LP.maxitsN = ((IntPortion *) param[4])->Value();
    LP.tolN = ((NumberPortion *) param[5])->Value();
    LP.maxits1 = ((IntPortion *) param[6])->Value();
    LP.tol1 = ((NumberPortion *) param[7])->Value();
    
    LP.tracefile = &((OutputPortion *) param[10])->Value();
    LP.trace = ((IntPortion *) param[11])->Value();

    gWatch watch;

    NFLiapBySubgame M(E, LP, start);
    M.Solve();

    ((NumberPortion *) param[8])->Value() = watch.Elapsed();
    ((IntPortion *) param[9])->Value() = M.NumEvals();

    return new Behav_ListPortion(M.GetSolutions());
  }
  else  {
    EFLiapParams LP;

    LP.stopAfter = ((IntPortion *) param[2])->Value();
    LP.nTries = ((IntPortion *) param[3])->Value();

    LP.maxitsN = ((IntPortion *) param[4])->Value();
    LP.tolN = ((NumberPortion *) param[5])->Value();
    LP.maxits1 = ((IntPortion *) param[6])->Value();
    LP.tol1 = ((NumberPortion *) param[7])->Value();
    
    LP.tracefile = &((OutputPortion *) param[10])->Value();
    LP.trace = ((IntPortion *) param[11])->Value();

    gWatch watch;

    EFLiapBySubgame M(E, LP, start);
    M.Solve();

    ((NumberPortion *) param[8])->Value() = watch.Elapsed();
    ((IntPortion *) param[9])->Value() = M.NumEvals();

    Portion *por = new Behav_ListPortion(M.GetSolutions());
    return por;
  }
}

#include "nliap.h"

static Portion *GSM_Liap_Mixed(Portion **param)
{
  MixedProfile<gNumber> &start = *((MixedPortion *) param[0])->Value();
  Nfg &N = start.Game();

  NFLiapParams params;
  
  params.stopAfter = ((IntPortion *) param[1])->Value();
  params.nTries = ((IntPortion *) param[2])->Value();

  params.maxitsN = ((IntPortion *) param[3])->Value();
  params.tolN = ((NumberPortion *) param[4])->Value();
  params.maxits1 = ((IntPortion *) param[5])->Value();
  params.tol1 = ((NumberPortion *) param[6])->Value();
 
  params.tracefile = &((OutputPortion *) param[9])->Value();
  params.trace = ((IntPortion *) param[10])->Value();

  long niters;
  gWatch watch;
  gList<MixedSolution> solutions;
  Liap(N, params, start, solutions,
       ((IntPortion *) param[8])->Value(), niters);

  ((NumberPortion *) param[7])->Value() = watch.Elapsed();

  return new Mixed_ListPortion(solutions);
}

//------------
// LpSolve
//------------

#include "nfgcsum.h"

static Portion *GSM_Lp_Nfg(Portion **param)
{
  NFSupport& S = * ((NfSupportPortion*) param[0])->Value();
  const Nfg *N = &S.Game();

  ZSumParams ZP;
  ZP.tracefile = &((OutputPortion *) param[4])->Value();
  ZP.trace = ((IntPortion *) param[5])->Value();

  if (N->NumPlayers() > 2 || !IsConstSum(*N))
	  return new ErrorPortion("Only valid for two-person zero-sum games");

  if (((PrecisionPortion *) param[1])->Value() == precDOUBLE)  {
    ZSumModule<double> ZM(S.Game(), ZP, S);
    ZM.ZSum();
    ((IntPortion *) param[2])->Value() = ZM.NumPivots();
    ((NumberPortion *) param[3])->Value() = ZM.Time();
    gList<MixedSolution> solns;
    ZM.GetSolutions(solns);
    return new Mixed_ListPortion(solns);
  }
  else  {
    ZSumModule<gRational> ZM(S.Game(), ZP, S);
    ZM.ZSum();
    ((IntPortion *) param[2])->Value() = ZM.NumPivots();
    ((NumberPortion *) param[3])->Value() = ZM.Time();
    gList<MixedSolution> solns;
    ZM.GetSolutions(solns);
    return new Mixed_ListPortion(solns);
  }
}


#include "lpsolve.h"

Portion* GSM_Lp_ListFloat(Portion** param)
{
  gMatrix<double>* a = ListToMatrix_Float((ListPortion*) param[0]);
  gVector<double>* b = ListToVector_Float((ListPortion*) param[1]);
  gVector<double>* c = ListToVector_Float((ListPortion*) param[2]);
  if(!a || !b || !c)
    return 0;
  int nequals = ((IntPortion*) param[3])->Value();
  bool isFeasible;
  bool isBounded;
  
  LPSolve<double>* s = new LPSolve<double>(*a, *b, *c, nequals);
  Portion* result = ArrayToList(s->OptimumVector());
  isFeasible = s->IsFeasible();
  isBounded = s->IsBounded();
  delete s;
  delete a;
  delete b;
  delete c;
  
  ((BoolPortion*) param[4])->Value() = isFeasible;
  ((BoolPortion*) param[5])->Value() = isBounded;
  return result;
}

Portion* GSM_Lp_ListRational(Portion** param)
{
  gMatrix<gRational>* a = ListToMatrix_Rational((ListPortion*) param[0]);
  gVector<gRational>* b = ListToVector_Rational((ListPortion*) param[1]);
  gVector<gRational>* c = ListToVector_Rational((ListPortion*) param[2]);
  if(!a || !b || !c)
    return 0;
  int nequals = ((IntPortion*) param[3])->Value();
  bool isFeasible;
  bool isBounded;
  
  LPSolve<gRational>* s = new LPSolve<gRational>(*a, *b, *c, nequals);
  Portion* result = ArrayToList(s->OptimumVector());
  isFeasible = s->IsFeasible();
  isBounded = s->IsBounded();
  delete s;
  delete a;
  delete b;
  delete c;
  
  ((BoolPortion*) param[4])->Value() = isFeasible;
  ((BoolPortion*) param[5])->Value() = isBounded;
  return result;
}




#include "csumsub.h"
#include "efgcsum.h"

static Portion *GSM_Lp_Efg(Portion **param)
{
  EFSupport &support = *((EfSupportPortion *) param[0])->Value();
  const Efg &E = support.Game();
  
  if (E.NumPlayers() > 2 || !E.IsConstSum())
    return new ErrorPortion("Only valid for two-person zero-sum games");

  if (((BoolPortion *) param[1])->Value())   {
   	ZSumParams ZP;
   	ZP.tracefile = &((OutputPortion *) param[5])->Value();
	  ZP.trace = ((IntPortion *) param[6])->Value();

    if (((PrecisionPortion *) param[2])->Value() == precDOUBLE)  {
      ZSumBySubgame<double> ZM(E, support, ZP);
      ZM.Solve();
      ((IntPortion *) param[3])->Value() = ZM.NumPivots();
      ((NumberPortion *) param[4])->Value() = ZM.Time();
      gList<BehavSolution> solns;
      solns = ZM.GetSolutions();
      return new Behav_ListPortion(solns);
    }
    else  {
      ZSumBySubgame<gRational> ZM(E, support, ZP);
      ZM.Solve();
      ((IntPortion *) param[3])->Value() = ZM.NumPivots();
      ((NumberPortion *) param[4])->Value() = ZM.Time();
      gList<BehavSolution> solns;
      solns = ZM.GetSolutions();
      return new Behav_ListPortion(solns);
    }
  }
  else  {
   	CSSeqFormParams ZP;
	  ZP.tracefile = &((OutputPortion *) param[5])->Value();
	  ZP.trace = ((IntPortion *) param[6])->Value();

    if (((PrecisionPortion *) param[2])->Value() == precDOUBLE)  {
      CSSeqFormBySubgame<double> ZM(E, support, ZP);
	    ZM.Solve();
      ((IntPortion *) param[3])->Value() = ZM.NumPivots();
	    ((NumberPortion *) param[4])->Value() = ZM.Time();
      return new Behav_ListPortion(ZM.GetSolutions());
    }
    else  {
      CSSeqFormBySubgame<gRational> ZM(E, support, ZP);
	    ZM.Solve();
      ((IntPortion *) param[3])->Value() = ZM.NumPivots();
	    ((NumberPortion *) param[4])->Value() = ZM.Time();
      return new Behav_ListPortion(ZM.GetSolutions());
    }
  }
}


//---------
// Nfg
//---------

static Portion *GSM_Nfg(Portion **param)
{
  Efg &E = * ((EfgPortion*) param[0])->Value();
  gWatch watch;

  Nfg *N = MakeReducedNfg(E, EFSupport(E));

  ((NumberPortion *) param[1])->Value() = watch.Elapsed();

  if (N)
    return new NfgValPortion(N);
  else
    return new ErrorPortion("Conversion to reduced nfg failed");
}



//----------
// Payoff
//----------

Portion* GSM_Payoff_Behav(Portion** param)
{
  BehavProfile<gNumber>* bp = ((BehavPortion*) param[0])->Value();
  EFPlayer *player = ((EfPlayerPortion *) param[1])->Value();

  return new NumberPortion(bp->Payoff(player->GetNumber()));
}

Portion* GSM_Payoff_Mixed(Portion** param)
{
  MixedProfile<gNumber>* mp = ((MixedPortion*) param[0])->Value();
  NFPlayer *player = ((NfPlayerPortion *) param[1])->Value();

  return new NumberPortion(mp->Payoff(player->GetNumber()));
}

//----------------
// SimpDivSolve
//----------------

#include "simpdiv.h"

static Portion *GSM_Simpdiv_Nfg(Portion **param)
{
  NFSupport& S = * ((NfSupportPortion*) param[0])->Value();
  SimpdivParams SP;
  SP.stopAfter = ((IntPortion *) param[1])->Value();
  SP.nRestarts = ((IntPortion *) param[2])->Value();
  SP.leashLength = ((IntPortion *) param[3])->Value();

  SP.tracefile = &((OutputPortion *) param[7])->Value();
  SP.trace = ((IntPortion *) param[8])->Value();

  if (((PrecisionPortion *) param[4])->Value() == precDOUBLE)  {
    SimpdivModule<double> SM(S.Game(), SP, S);
    SM.Simpdiv();
    ((IntPortion *) param[5])->Value() = SM.NumEvals();
    ((NumberPortion *) param[6])->Value() = SM.Time();
    return new Mixed_ListPortion(SM.GetSolutions());
  }
  else  {
    SimpdivModule<gRational> SM(S.Game(), SP, S);
    SM.Simpdiv();
    ((IntPortion *) param[5])->Value() = SM.NumEvals();
    ((NumberPortion *) param[6])->Value() = SM.Time();
    return new Mixed_ListPortion(SM.GetSolutions());
  }
}

#include "simpsub.h"

static Portion *GSM_Simpdiv_Efg(Portion **param)
{
  EFSupport &support = *((EfSupportPortion *) param[0])->Value();

  if (!((BoolPortion *) param[1])->Value())
    return new ErrorPortion("algorithm not implemented for extensive forms");

  SimpdivParams SP;
  SP.stopAfter = ((IntPortion *) param[2])->Value();
  SP.nRestarts = ((IntPortion *) param[3])->Value();
  SP.leashLength = ((IntPortion *) param[4])->Value();

  SP.tracefile = &((OutputPortion *) param[8])->Value();
  SP.trace = ((IntPortion *) param[9])->Value();

  if (((PrecisionPortion *) param[5])->Value() == precDOUBLE)  {
    SimpdivBySubgame<double> SM(support.Game(), support, SP);
    SM.Solve();
    ((IntPortion *) param[6])->Value() = SM.NumEvals();
    ((NumberPortion *) param[7])->Value() = SM.Time();
    return new Behav_ListPortion(SM.GetSolutions());
  }
  else  {
    SimpdivBySubgame<gRational> SM(support.Game(), support, SP);
    SM.Solve();
    ((IntPortion *) param[6])->Value() = SM.NumEvals();
    ((NumberPortion *) param[7])->Value() = SM.Time();
    return new Behav_ListPortion(SM.GetSolutions());
  }
}

Portion* GSM_VertEnum_Float( Portion** param )
{
  gMatrix<double>* A = ListToMatrix_Float((ListPortion*) param[0]);
  gVector<double>* b = ListToVector_Float((ListPortion*) param[1]);
  gVector<double>* start = ListToVector_Float((ListPortion*) param[2]);

  gList< gVector< double > > verts;

  if( start->Length() == 0 ) {
    VertEnum< double >* vertenum = NULL;
    vertenum = new VertEnum< double >( *A, *b );
    vertenum->Vertices( verts );
    delete vertenum;
  }
  else {
    NewVertEnum< double >* vertenum = NULL;
    vertenum = new NewVertEnum< double >( *A, *b, *start );
    vertenum->Vertices( verts );
    delete vertenum;
  }

  delete A;
  delete b;
  delete start;

  ListPortion* list = new ListValPortion();
  int i = 0;
  for( i = 1; i <= verts.Length(); ++i )
  {
    list->Append( ArrayToList( verts[i] ) );
  }

  return list;
}


Portion* GSM_VertEnum_Rational( Portion** param )
{
  gMatrix<gRational>* A = ListToMatrix_Rational((ListPortion*) param[0]);
  gVector<gRational>* b = ListToVector_Rational((ListPortion*) param[1]);
  gVector<gRational>* start = ListToVector_Rational((ListPortion*) param[2]);

  gList< gVector< gRational > > verts;

  if( start->Length() == 0 ) {
    VertEnum< gRational >* vertenum = NULL;
    vertenum = new VertEnum< gRational >( *A, *b );
    vertenum->Vertices( verts );
    delete vertenum;
  }
  else {
    NewVertEnum< gRational >* vertenum = NULL;
    vertenum = new NewVertEnum< gRational >( *A, *b, *start );
    vertenum->Vertices( verts );
    delete vertenum;
  }

  delete A;
  delete b;
  delete start;

  ListPortion* list = new ListValPortion();
  int i = 0;
  for( i = 1; i <= verts.Length(); ++i )
  {
    list->Append( ArrayToList( verts[i] ) );
  }

  return list;
}

void Init_algfunc(GSM *gsm)
{
  FuncDescObj *FuncObj;

  FuncObj = new FuncDescObj("AgentForm", 1);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_AgentForm, porNFG, 2));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("efg", porEFG));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0), BYREF));
  gsm->AddFunction(FuncObj);



  FuncObj = new FuncDescObj("Behav", 1);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Behav, porBEHAV, 1));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("mixed", porMIXED));
  gsm->AddFunction(FuncObj);



  FuncObj = new FuncDescObj("EnumMixedSolve", 2);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_EnumMixed_Nfg, 
				       PortionSpec(porMIXED, 1), 7));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("support", porNFSUPPORT));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(0)));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("nPivots", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull), 
					    BYREF));
  FuncObj->SetParamInfo(0, 6, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_EnumMixed_Efg, 
				       PortionSpec(porBEHAV, 1), 8));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("support", porEFSUPPORT));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("asNfg", porBOOL,
					    new BoolPortion(false)));
  FuncObj->SetParamInfo(1, 2, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(0)));
  FuncObj->SetParamInfo(1, 3, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(1, 4, ParamInfoType("nPivots", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(1, 5, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(1, 6, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull), 
					    BYREF));
  FuncObj->SetParamInfo(1, 7, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));
  gsm->AddFunction(FuncObj);


  FuncObj = new FuncDescObj("EnumPureSolve", 2);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_EnumPure_Nfg, 
				       PortionSpec(porMIXED, 1), 6));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("support", porNFSUPPORT));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(0)));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull), 
					    BYREF));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_EnumPure_Efg, 
				       PortionSpec(porBEHAV, 1), 7));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("support", porEFSUPPORT));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("asNfg", porBOOL,
					    new BoolPortion(false)));
  FuncObj->SetParamInfo(1, 2, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(0)));
  FuncObj->SetParamInfo(1, 3, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(1, 4, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(1, 5, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull), 
					    BYREF));
  FuncObj->SetParamInfo(1, 6, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));
  gsm->AddFunction(FuncObj);


  FuncObj = new FuncDescObj("GobitGridSolve", 1);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_GobitGrid_Support, 
				       PortionSpec(porMIXED, 1), 14));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("support", porNFSUPPORT));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("pxifile", porTEXT,
					    new TextPortion("")));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("minLam", porNUMBER,
					    new NumberPortion(0.001)));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("maxLam", porNUMBER,
					    new NumberPortion(500.0)));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("delLam", porNUMBER,
					    new NumberPortion(0.02)));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("powLam", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(0, 6, ParamInfoType("delp1", porNUMBER,
					    new NumberPortion(.1)));
  FuncObj->SetParamInfo(0, 7, ParamInfoType("tol1", porNUMBER,
					    new NumberPortion(.1)));
  FuncObj->SetParamInfo(0, 8, ParamInfoType("delp2", porNUMBER,
					    new NumberPortion(.01)));
  FuncObj->SetParamInfo(0, 9, ParamInfoType("tol2", porNUMBER,
					    new NumberPortion(.01)));
  FuncObj->SetParamInfo(0, 10, ParamInfoType("nEvals", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 11, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(0, 12, ParamInfoType("traceFile", porOUTPUT,
					     new OutputRefPortion(gnull),
					     BYREF));
  FuncObj->SetParamInfo(0, 13, ParamInfoType("traceLevel", porINTEGER,
					     new IntPortion(0)));

  gsm->AddFunction(FuncObj);


  FuncObj = new FuncDescObj("GobitSolve", 1);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Gobit_Start, 
				       PortionSpec(porMIXED | porBEHAV, 1), 16));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("start",
					    porMIXED | porBEHAV));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("pxifile", porTEXT,
					    new TextPortion("")));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("minLam", porNUMBER,
					    new NumberPortion(0.001)));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("maxLam", porNUMBER,
					    new NumberPortion(500.0)));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("delLam", porNUMBER,
					    new NumberPortion(0.02)));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("powLam", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(0, 6, ParamInfoType("fullGraph", porBOOL,
					    new BoolPortion(false)));
  FuncObj->SetParamInfo(0, 7, ParamInfoType("maxitsN", porINTEGER,
					    new IntPortion(20)));
  FuncObj->SetParamInfo(0, 8, ParamInfoType("tolN", porNUMBER,
					    new NumberPortion(1.0e-10)));
  FuncObj->SetParamInfo(0, 9, ParamInfoType("maxits1", porINTEGER,
					    new IntPortion(100)));
  FuncObj->SetParamInfo(0, 10, ParamInfoType("tol1", porNUMBER,
					     new NumberPortion(2.0e-10)));
  FuncObj->SetParamInfo(0, 11, ParamInfoType("time", porNUMBER,
					     new NumberPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 12, ParamInfoType("nEvals", porINTEGER,
					     new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 13, ParamInfoType("nIters", porINTEGER,
					     new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 14, ParamInfoType("traceFile", porOUTPUT,
					     new OutputRefPortion(gnull), 
					     BYREF));
  FuncObj->SetParamInfo(0, 15, ParamInfoType("traceLevel", porINTEGER,
					     new IntPortion(0)));

  gsm->AddFunction(FuncObj);


  FuncObj = new FuncDescObj("LcpSolve", 4);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Lcp_Nfg, 
				       PortionSpec(porMIXED, 1), 7));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("support", porNFSUPPORT));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("stopAfter", porINTEGER, 
					    new IntPortion(0)));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("nPivots", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull),
					    BYREF));
  FuncObj->SetParamInfo(0, 6, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_Lcp_ListFloat, 
				       PortionSpec(porNUMBER, 1), 2));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("a", PortionSpec(porNUMBER,2),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("b", PortionSpec(porNUMBER,1),
					    REQUIRED, BYVAL));

  FuncObj->SetFuncInfo(2, FuncInfoType(GSM_Lcp_ListRational, 
				       PortionSpec(porNUMBER, 1), 2));
  FuncObj->SetParamInfo(2, 0, ParamInfoType("a", PortionSpec(porNUMBER,2),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(2, 1, ParamInfoType("b", PortionSpec(porNUMBER,1),
					    REQUIRED, BYVAL));

  FuncObj->SetFuncInfo(3, FuncInfoType(GSM_Lcp_Efg, 
				       PortionSpec(porBEHAV, 1), 8));
  FuncObj->SetParamInfo(3, 0, ParamInfoType("support", porEFSUPPORT));
  FuncObj->SetParamInfo(3, 1, ParamInfoType("asNfg", porBOOL,
					    new BoolPortion(false)));
  FuncObj->SetParamInfo(3, 2, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(0)));
  FuncObj->SetParamInfo(3, 3, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(3, 4, ParamInfoType("nPivots", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(3, 5, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(3, 6, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull), 
					    BYREF));
  FuncObj->SetParamInfo(3, 7, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));
  
  gsm->AddFunction(FuncObj);


  FuncObj = new FuncDescObj("LiapSolve", 2);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Liap_Behav,
				       PortionSpec(porBEHAV, 1), 12));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("start", porBEHAV));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("asNfg", porBOOL,
					    new BoolPortion(false)));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("nTries", porINTEGER,
					    new IntPortion(10)));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("maxitsN", porINTEGER,
					    new IntPortion(20)));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("tolN", porNUMBER,
					    new NumberPortion(1.0e-10)));
  FuncObj->SetParamInfo(0, 6, ParamInfoType("maxits1", porINTEGER,
					    new IntPortion(100)));
  FuncObj->SetParamInfo(0, 7, ParamInfoType("tol1", porNUMBER,
					    new NumberPortion(2.0e-10)));
  FuncObj->SetParamInfo(0, 8, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 9, ParamInfoType("nEvals", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 10, ParamInfoType("traceFile", porOUTPUT,
					     new OutputRefPortion(gnull), 
					     BYREF));
  FuncObj->SetParamInfo(0, 11, ParamInfoType("traceLevel", porINTEGER,
					     new IntPortion(0)));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_Liap_Mixed,
				       PortionSpec(porMIXED, 1), 11));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("start", porMIXED));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(1, 2, ParamInfoType("nTries", porINTEGER,
					    new IntPortion(10)));
  FuncObj->SetParamInfo(1, 3, ParamInfoType("maxitsN", porINTEGER,
					    new IntPortion(20)));
  FuncObj->SetParamInfo(1, 4, ParamInfoType("tolN", porNUMBER,
					    new NumberPortion(1.0e-10)));
  FuncObj->SetParamInfo(1, 5, ParamInfoType("maxits1", porINTEGER,
					    new IntPortion(100)));
  FuncObj->SetParamInfo(1, 6, ParamInfoType("tol1", porNUMBER,
					    new NumberPortion(2.0e-10)));
  FuncObj->SetParamInfo(1, 7, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0), BYREF));
  FuncObj->SetParamInfo(1, 8, ParamInfoType("nEvals", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(1, 9, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull), 
					    BYREF));
  FuncObj->SetParamInfo(1, 10, ParamInfoType("traceLevel", porINTEGER,
					     new IntPortion(0)));

  gsm->AddFunction(FuncObj);


  FuncObj = new FuncDescObj("LpSolve", 4);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Lp_Nfg, 
				       PortionSpec(porMIXED, 1), 6));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("support", porNFSUPPORT));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("nPivots", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull),
					    BYREF));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_Lp_Efg, 
				       PortionSpec(porBEHAV, 1), 7));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("support", porEFSUPPORT));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("asNfg", porBOOL,
					    new BoolPortion(false)));
  FuncObj->SetParamInfo(1, 2, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(1, 3, ParamInfoType("nPivots", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(1, 4, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(1, 5, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull), 
					    BYREF));
  FuncObj->SetParamInfo(1, 6, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));

  FuncObj->SetFuncInfo(2, FuncInfoType(GSM_Lp_ListFloat, 
				       PortionSpec(porNUMBER, 1), 6));
  FuncObj->SetParamInfo(2, 0, ParamInfoType("a", PortionSpec(porNUMBER,2),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(2, 1, ParamInfoType("b", PortionSpec(porNUMBER,1),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(2, 2, ParamInfoType("c", PortionSpec(porNUMBER,1),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(2, 3, ParamInfoType("nEqualities", porINTEGER));
  FuncObj->SetParamInfo(2, 4, ParamInfoType("isFeasible", porBOOL,
					    new BoolPortion(false), BYREF));
  FuncObj->SetParamInfo(2, 5, ParamInfoType("isBounded", porBOOL,
					    new BoolPortion(false), BYREF));

  gsm->AddFunction(FuncObj);


  FuncObj = new FuncDescObj("Nfg", 1);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Nfg, porNFG, 2));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("efg", porEFG));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0), BYREF));
  gsm->AddFunction(FuncObj);




  FuncObj = new FuncDescObj("Payoff", 2);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Payoff_Behav, porNUMBER, 2,
				       0, funcLISTABLE | funcGAMEMATCH));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("profile", porBEHAV));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("player", porEFPLAYER));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_Payoff_Mixed, porNUMBER, 2,
				       0, funcLISTABLE | funcGAMEMATCH));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("profile", porMIXED));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("player", porNFPLAYER));
  gsm->AddFunction(FuncObj);



  FuncObj = new FuncDescObj("SimpDivSolve", 2);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_Simpdiv_Nfg,
				       PortionSpec(porMIXED, 1), 9));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("support", porNFSUPPORT));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("nRestarts", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(0, 3, ParamInfoType("leashLength", porINTEGER,
					    new IntPortion(0)));
  FuncObj->SetParamInfo(0, 4, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(0, 5, ParamInfoType("nEvals", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(0, 6, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(0, 7, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull),
					    BYREF));
  FuncObj->SetParamInfo(0, 8, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_Simpdiv_Efg,
				       PortionSpec(porBEHAV, 1), 10));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("support", porEFSUPPORT));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("asNfg", porBOOL,
					    new BoolPortion(false)));
  FuncObj->SetParamInfo(1, 2, ParamInfoType("stopAfter", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(1, 3, ParamInfoType("nRestarts", porINTEGER,
					    new IntPortion(1)));
  FuncObj->SetParamInfo(1, 4, ParamInfoType("leashLength", porINTEGER,
					    new IntPortion(0)));
  FuncObj->SetParamInfo(1, 5, ParamInfoType("precision", porPRECISION,
              new PrecisionPortion(precDOUBLE)));
  FuncObj->SetParamInfo(1, 6, ParamInfoType("nEvals", porINTEGER,
					    new IntPortion(0), BYREF));
  FuncObj->SetParamInfo(1, 7, ParamInfoType("time", porNUMBER,
					    new NumberPortion(0.0), BYREF));
  FuncObj->SetParamInfo(1, 8, ParamInfoType("traceFile", porOUTPUT,
					    new OutputRefPortion(gnull),
					    BYREF));
  FuncObj->SetParamInfo(1, 9, ParamInfoType("traceLevel", porINTEGER,
					    new IntPortion(0)));
  gsm->AddFunction(FuncObj);




  FuncObj = new FuncDescObj("VertEnum", 2);
  FuncObj->SetFuncInfo(0, FuncInfoType(GSM_VertEnum_Float,
				       PortionSpec(porNUMBER, 2), 3));
  FuncObj->SetParamInfo(0, 0, ParamInfoType("A", PortionSpec(porNUMBER,2),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(0, 1, ParamInfoType("b", PortionSpec(porNUMBER,1),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(0, 2, ParamInfoType("start", PortionSpec(porNUMBER,1),
					    new ListValPortion(), BYVAL));

  FuncObj->SetFuncInfo(1, FuncInfoType(GSM_VertEnum_Rational,
				       PortionSpec(porNUMBER, 2), 3));
  FuncObj->SetParamInfo(1, 0, ParamInfoType("A", PortionSpec(porNUMBER,2),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(1, 1, ParamInfoType("b", PortionSpec(porNUMBER,1),
					    REQUIRED, BYVAL));
  FuncObj->SetParamInfo(1, 2, ParamInfoType("start", PortionSpec(porNUMBER,1),
					    new ListValPortion(), BYVAL));
  gsm->AddFunction(FuncObj);

}


