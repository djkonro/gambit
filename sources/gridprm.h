#ifndef GRIDPRM_H
#define GRIDPRM_H
#include "outprm.h"

template <class T>
class GridSolveParamsDialog : public PxiParamsDialog
{
private:
	float minLam, maxLam, delLam, delp, tol;
public:
	GridSolveParamsDialog(wxWindow *parent);
	~GridSolveParamsDialog(void);
	void GetParams(GridParams<T> &P);
};

//******************************** Constructor/main ************************
template <class T>
GridSolveParamsDialog<T>::GridSolveParamsDialog(wxWindow *parent)
				:PxiParamsDialog("grid","Grid Params",parent)
{
minLam=0.01;maxLam=3.0;delLam=.1;delp=.01;tol=.01;
Form()->Add(wxMakeFormFloat("L Start",&minLam,wxFORM_DEFAULT,0,0,wxVERTICAL,100));
Form()->Add(wxMakeFormFloat("L Stop",&maxLam,wxFORM_DEFAULT,0,0,wxVERTICAL,100));
Form()->Add(wxMakeFormFloat("L Step",&delLam,wxFORM_DEFAULT,0,0,wxVERTICAL,100));
Form()->Add(wxMakeFormNewLine());
Form()->Add(wxMakeFormFloat("P Step",&delp,wxFORM_DEFAULT,0,0,wxVERTICAL,100));
Form()->Add(wxMakeFormFloat("Tolerance",&tol,wxFORM_DEFAULT,0,0,wxVERTICAL,100));

// Now add the basic stuff
MakePxiFields();
MakeOutputFields();
Go();
}

template <class T>
GridSolveParamsDialog<T>::~GridSolveParamsDialog(void)
{}

template <class T>
void GridSolveParamsDialog<T>::GetParams(GridParams<T> &P)
{
P.minLam=minLam;P.maxLam=maxLam;P.delLam=delLam;P.tol=tol;P.delp=delp;
// Pxi stuff
P.type=PxiType();P.pxifile=PxiFile();
// Output stuff
P.plev=TraceLevel();P.outfile=OutFile();P.errfile=ErrFile();
}

#ifdef __GNUG__
#define TEMPLATE template
#elif defined __BORLANDC__
#pragma option -Jgd
#define TEMPLATE
#endif   // __GNUG__, __BORLANDC__
TEMPLATE class GridSolveParamsDialog<double> ;
TEMPLATE class GridSolveParamsDialog<gRational> ;
#ifdef __BORLANDC__
#pragma -Jgx
#endif

#endif
