#!/usr/bin/env python3
"""
Teste 53 — joint CLASS refit:
Planck 2018 Plik-lite TTTEEE high-l + DESI DR2 BAO + Gaussian tau prior.

This is the immediate degeneracy-breaking step after the high-l-only refit.

Likelihood pieces:
  * Planck 2018 Plik-lite TTTEEE high-l (613 bins)
  * DESI DR2 BAO 13-distance vector
  * tau = 0.0544 +/- 0.0073

BAO covariance:
  * BGS DV point independent
  * six anisotropic (DM/rd, DH/rd) 2x2 blocks
  * r_MH values from DESI DR2 Results II Table 4
  * baseline assumes no cross-redshift covariance, matching the published
    baseline treatment for BAO measurement systematics.

At each cosmological evaluation CLASS writes:
  cl_lensed.dat, background.dat, thermodynamics.dat.

Distances are derived directly from CLASS:
  DM = comoving distance (flat model)
  DH = 1/H(z) because CLASS H is in 1/Mpc with c=1
  rd = sound horizon evaluated at tau_drag = 1
  DV = [z DM^2 DH]^(1/3)

The same patched CLASS executable is used for LCDM and geodesic models.
Non-linear corrections are OFF.
"""

from __future__ import annotations
import numpy as np, subprocess, os, sys, json, time, csv, hashlib, math
from pathlib import Path
from scipy.optimize import minimize_scalar

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CLASS=os.environ.get("CLASS",str(ROOT/".runtime"/"class_iv"/"class"))
PLANCK_DIR=Path(os.environ.get("PLANCK_DIR",str(ROOT/".runtime"/"planck-lite-py")))
CAL=os.environ.get("CALIBRATION","fixed")
MAX_ITER=int(os.environ.get("TEST53_MAX_ITER","30"))
START_LIMIT=int(os.environ.get("TEST53_START_LIMIT","0"))
OUT=ROOT/f"results_joint_{CAL}"
OUT.mkdir(exist_ok=True)
WORK=ROOT/"work_joint"
WORK.mkdir(exist_ok=True)

TAU0,STAU=0.0544,0.0073
ACAL_SIG=0.0025
TCMB_UK2=(2.7255e6)**2

TEMPLATE=(ROOT/"config"/"base_template.ini").read_text()
# Force extra output needed for the BAO likelihood.
TEMPLATE += "\nwrite background = yes\nwrite thermodynamics = yes\n"

sys.path.insert(0,str(PLANCK_DIR))
cwd=os.getcwd()
os.chdir(PLANCK_DIR)
from planck_lite_py import PlanckLitePy
L=PlanckLitePy(data_directory="data",year=2018,spectra="TTTEEE",use_low_ell_bins=False)
os.chdir(cwd)
Lchol=np.linalg.cholesky(L.fisher)
NB=len(L.X_data)

# DESI DR2 BAO vector/covariance
bao=np.genfromtxt(ROOT/"data"/"desi_dr2_bao13.csv",delimiter=",",names=True,dtype=None,encoding=None)
B_LABEL=np.array(bao["label"])
B_Z=np.array(bao["z"],float)
B_DATA=np.array(bao["data"],float)
B_SIG=np.array(bao["sigma"],float)
CBAO=np.diag(B_SIG**2)

# six DM/DH pair correlations, rows 1-12 in pairs
for i in range(1,13,2):
    r=float(bao["r_MH"][i])
    CBAO[i,i+1]=CBAO[i+1,i]=r*B_SIG[i]*B_SIG[i+1]

LBAO=np.linalg.cholesky(CBAO)

# Start from the completed high-l-only refit, plus two alternative geodesic starts.
prev=json.loads((ROOT/"config"/"fit_summary_plancklite.json").read_text())
L0=prev["lcdm"]["params"]
G0=prev["geo"]["params"]

MODELS={
 "lcdm":dict(
   names=["omega_b","omega_cdm","h","ln10As","n_s","tau_reio"],
   starts=[
     [L0["omega_b"],L0["omega_cdm"],L0["h"],L0["ln10As"],L0["n_s"],L0["tau_reio"]],
     [0.02238,0.1201,0.6732,3.0448,0.9661,0.0544],
   ],
   step=[6e-5,6e-4,2.5e-3,6e-3,2.5e-3,4e-3],
   lo=[0.019,0.09,0.55,2.8,0.9,0.02],
   hi=[0.026,0.16,0.80,3.3,1.05,0.12]),
 "geo":dict(
   names=["omega_b","omega_idm_iv","h","ln10As","n_s","tau_reio","f_dyn_test53"],
   starts=[
     [G0["omega_b"],G0["omega_idm_iv"],G0["h"],G0["ln10As"],G0["n_s"],G0["tau_reio"],G0["f_dyn_test53"]],
     [0.02238,0.11833,0.6898,3.045,0.9660,0.0544,0.1917],
     [L0["omega_b"],L0["omega_cdm"],L0["h"],L0["ln10As"],L0["n_s"],L0["tau_reio"],0.0],
   ],
   step=[6e-5,6e-4,2.5e-3,6e-3,2.5e-3,4e-3,1.2e-2],
   lo=[0.019,0.09,0.55,2.8,0.9,0.02,0.0],
   hi=[0.026,0.16,0.80,3.3,1.05,0.12,1.0]),
}

eval_path=OUT/"evaluations_joint.csv"
newfile=not eval_path.exists()
csvf=eval_path.open("a",newline="")
W=csv.writer(csvf)
if newfile:
    W.writerow(["time","model","start_id","params_json","chi2_plik","chi2_bao",
                "A_planck","chi2_prior_tau","chi2_prior_cal","chi2_total",
                "rd_Mpc","status"])
cache={}

def ini_for(model,p):
    s=TEMPLATE
    if model=="lcdm":
        s+=f"omega_cdm = {p['omega_cdm']:.12g}\n"
    else:
        s+=(
          "omega_cdm = 0.\n"
          f"omega_idm_iv = {p['omega_idm_iv']:.12g}\n"
          "f_iv = 1.\nalpha_idm_iv = 0.\nbeta_idm_iv = 0.\n"
          f"f_dyn_test53 = {p['f_dyn_test53']:.12g}\n"
        )
    s+=(
      f"h = {p['h']:.12g}\n"
      f"omega_b = {p['omega_b']:.12g}\n"
      f"tau_reio = {p['tau_reio']:.12g}\n"
      f"A_s = {np.exp(p['ln10As'])*1e-10:.14e}\n"
      f"n_s = {p['n_s']:.12g}\n"
    )
    return s

def binned(TT,TE,EE):
    ls=np.arange(2,2+len(TT))
    fac=ls*(ls+1)/(2*np.pi)
    out=[]
    for D,nb in [(TT,215),(TE,199),(EE,199)]:
        C=D/fac
        out.append(np.array([
            np.sum(L.bin_w[L.blmin[i]:L.blmax[i]+1]*
                   C[L.blmin[i]+28:L.blmax[i]+29])
            for i in range(nb)
        ]))
    return np.concatenate(out)

def derive_bao(bg,th):
    # CLASS background:
    # col 0 z, col 3 H[1/Mpc], col 4 comoving distance[Mpc],
    # col 7 comoving sound horizon[Mpc].
    # thermo col 9 = tau_d.
    zt=th[:,0]; td=th[:,9]
    cross=np.where((td[:-1]-1.0)*(td[1:]-1.0)<=0)[0]
    if len(cross)==0:
        raise RuntimeError("tau_d=1 crossing not found")
    i=int(cross[0])
    zd=zt[i]+(1.0-td[i])*(zt[i+1]-zt[i])/(td[i+1]-td[i])

    order=np.argsort(bg[:,0])
    zb=bg[order,0]
    H=np.interp(B_Z,zb,bg[order,3])
    DM=np.interp(B_Z,zb,bg[order,4])
    DH=1.0/H
    rd=float(np.interp(zd,zb,bg[order,7]))

    pred=np.empty(13)
    pred[0]=(B_Z[0]*DM[0]**2*DH[0])**(1/3)/rd
    for j in range(1,13,2):
        z=B_Z[j]
        # B_Z[j] == B_Z[j+1]
        pred[j]=DM[j]/rd
        pred[j+1]=DH[j+1]/rd
    return pred,rd,zd

def residual(model,x,start_id=0,A_override=None):
    names=MODELS[model]["names"]
    p=dict(zip(names,map(float,x)))
    key=(model,tuple(np.round(x,12)),CAL,None if A_override is None else round(float(A_override),12))
    if key in cache:
        return cache[key]

    s=ini_for(model,p)
    hsh=hashlib.md5((model+CAL+s).encode()).hexdigest()[:14]
    root=WORK/f"{model}_{hsh}_"
    ini=WORK/f"{model}_{hsh}.ini"
    ini.write_text(s+f"root = {root}\n")

    r=subprocess.run([CLASS,str(ini)],cwd=ROOT,capture_output=True,text=True)
    fcl=Path(str(root)+"cl_lensed.dat")
    fbg=Path(str(root)+"background.dat")
    fth=Path(str(root)+"thermodynamics.dat")

    if r.returncode!=0 or not (fcl.exists() and fbg.exists() and fth.exists()):
        msg=(r.stdout+r.stderr).strip().splitlines()
        W.writerow([time.time(),model,start_id,json.dumps(p),"","","","","",1e10,"","CLASS_FAIL:"+((msg[-1][:120]) if msg else "")])
        csvf.flush()
        cache[key]=None
        return None

    try:
        d=np.loadtxt(fcl)
        thvec=binned(d[:,1]*TCMB_UK2,d[:,3]*TCMB_UK2,d[:,2]*TCMB_UK2)
        bg=np.loadtxt(fbg); th=np.loadtxt(fth)
        bp,rd,zd=derive_bao(bg,th)

        if A_override is not None:
            A=float(A_override)
        elif CAL=="profile":
            # A_planck affects only the Plik-lite block.
            def g(A):
                rv=L.X_data-thvec/A**2
                return rv@L.fisher@rv + ((A-1)/ACAL_SIG)**2
            A=float(minimize_scalar(g,bounds=(0.98,1.02),method="bounded",
                                    options={"xatol":1e-8}).x)
        else:
            A=1.0

        rv=L.X_data-thvec/A**2
        wplik=Lchol.T@rv
        wbao=np.linalg.solve(LBAO,bp-B_DATA)
        wtau=(p["tau_reio"]-TAU0)/STAU
        pcal=(A-1)/ACAL_SIG if CAL=="profile" else 0.0

        chiP=float(wplik@wplik)
        chiB=float(wbao@wbao)
        chiT=float(wtau**2)
        chiC=float(pcal**2)

        # Residual vector used by LM.
        full=np.concatenate([wplik,wbao,[wtau],([pcal] if CAL=="profile" else [])])
        res=(full,chiP,chiB,A,chiT,chiC,rd,zd,bp)
        W.writerow([time.time(),model,start_id,json.dumps(p),chiP,chiB,A,chiT,chiC,
                    chiP+chiB+chiT+chiC,rd,"ok"])
        csvf.flush()
    except Exception as e:
        W.writerow([time.time(),model,start_id,json.dumps(p),"","","","","",1e10,"","PARSE_FAIL:"+repr(e)[:120]])
        csvf.flush()
        res=None

    # Keep only INI in work; spectra/backgrounds are reproducible but huge in aggregate.
    for fp in [fcl,Path(str(root)+"cl.dat"),fbg,fth]:
        try: fp.unlink()
        except FileNotFoundError: pass

    cache[key]=res
    return res

def tot(res):
    return np.inf if res is None else float(res[0]@res[0])

def lm(model,x0,start_id,log):
    M=MODELS[model]
    x=np.array(x0,float)
    st=np.array(M["step"],float)
    lo=np.array(M["lo"],float)
    hi=np.array(M["hi"],float)
    lam=1e-2
    res=residual(model,x,start_id)
    c=tot(res)
    log(f"[{model} s{start_id}] start chi2={c:.5f}")

    lastJ=None
    for it in range(MAX_ITER):
        J=np.zeros((len(res[0]),len(x)))
        for k in range(len(x)):
            xk=x.copy()
            hh=st[k] if x[k]+st[k]<=hi[k] else -st[k]
            xk[k]+=hh
            rk=residual(model,xk,start_id)
            if rk is None:
                hh=-hh
                xk=x.copy(); xk[k]+=hh
                rk=residual(model,xk,start_id)
            J[:,k]=(rk[0]-res[0])/hh

        D=np.diag(1/st)
        Js=J@np.linalg.inv(D)
        A_=Js.T@Js
        g=Js.T@res[0]
        improved=False
        dc=0.0

        for _ in range(8):
            diag=np.diag(A_).copy()
            diag[diag<=0]=1.0
            try:
                dz=-np.linalg.solve(A_+lam*np.diag(diag),g)
            except np.linalg.LinAlgError:
                dz=-np.linalg.pinv(A_+lam*np.diag(diag))@g
            dx=dz/np.diag(D)
            xn=np.clip(x+dx,lo,hi)
            rn=residual(model,xn,start_id)
            cn=tot(rn)
            log(f"[{model} s{start_id}] it{it} lam={lam:.1e} chi2={cn:.5f} current={c:.5f}")
            if cn<c-1e-4:
                dc=c-cn
                x,res,c=xn,rn,cn
                lam=max(lam/5,1e-7)
                improved=True
                break
            lam*=8

        lastJ=J
        if not improved or dc<0.01:
            log(f"[{model} s{start_id}] converged it={it} last_dchi2={dc:.5f}")
            break

    if lastJ is not None:
        F=lastJ.T@lastJ
        if np.linalg.cond(F)<1e14:
            cov=np.linalg.inv(F)
            err=np.sqrt(np.clip(np.diag(cov),0,None))
        else:
            err=np.full(len(x),np.nan)
    else:
        err=np.full(len(x),np.nan)

    return x,res,c,err

def main():
    logf=(HERE/f"joint_{CAL}.log").open("a")
    def log(m):
        s=time.strftime("%H:%M:%S ")+m
        print(s,flush=True)
        logf.write(s+"\n"); logf.flush()

    log("==== Teste 53 JOINT Plik-lite TTTEEE + DESI DR2 BAO + tau prior ====")
    log(f"CALIBRATION={CAL}; nonlinear OFF; tau={TAU0}+-{STAU}")
    R={}

    for model in ["lcdm","geo"]:
        trials=[]
        starts=MODELS[model]["starts"]
        if START_LIMIT > 0:
            starts=starts[:START_LIMIT]
        for sid,x0 in enumerate(starts):
            x,res,c,err=lm(model,x0,sid,log)
            trials.append((c,x,res,err,sid))
        c,x,res,err,sid=min(trials,key=lambda t:t[0])
        names=MODELS[model]["names"]
        p=dict(zip(names,map(float,x)))
        e=dict(zip(names,map(float,err)))
        R[model]=dict(
            params=p,errors_fisher=e,chi2_total=float(c),
            chi2_plik=float(res[1]),chi2_bao=float(res[2]),
            A_planck=float(res[3]),chi2_prior_tau=float(res[4]),
            chi2_prior_cal=float(res[5]),rd_Mpc=float(res[6]),
            z_drag=float(res[7]),bao_prediction=list(map(float,res[8])),
            winning_start=int(sid),k=len(names)
        )
        (OUT/f"bestfit_{model}.ini").write_text(
            f"# joint best fit {model} | chi2={c:.8f}\n"
            +ini_for(model,p)
            +f"root = output/test53_joint_{model}_\n"
        )

    n=NB+13+1+(1 if CAL=="profile" else 0)
    dchi=R["geo"]["chi2_total"]-R["lcdm"]["chi2_total"]
    dk=R["geo"]["k"]-R["lcdm"]["k"]

    S=dict(
      likelihood="Planck2018 Plik-lite TTTEEE high-l + DESI DR2 BAO13 + Gaussian tau prior",
      nonlinear="off",tau_prior=[TAU0,STAU],calibration=CAL,
      n_data=n,
      bao_covariance="block diagonal; official r_MH within each anisotropic redshift bin",
      chi2_LCDM_min=R["lcdm"]["chi2_total"],
      chi2_GEO_min=R["geo"]["chi2_total"],
      delta_chi2=dchi,
      delta_AIC=dchi+2*dk,
      delta_BIC=dchi+dk*np.log(n),
      lcdm=R["lcdm"],geo=R["geo"]
    )
    json.dump(S,(OUT/"fit_summary_joint.json").open("w"),indent=2)

    # Human-readable BAO best-fit table
    with (OUT/"bao_bestfit_predictions.csv").open("w",newline="") as fh:
        ww=csv.writer(fh)
        ww.writerow(["label","z","data","sigma","lcdm","geo"])
        for i in range(13):
            ww.writerow([B_LABEL[i],B_Z[i],B_DATA[i],B_SIG[i],
                         R["lcdm"]["bao_prediction"][i],
                         R["geo"]["bao_prediction"][i]])

    log("")
    log("================ JOINT RESULT ================")
    log(f"chi2_LCDM = {R['lcdm']['chi2_total']:.6f} "
        f"(Plik {R['lcdm']['chi2_plik']:.4f} + BAO {R['lcdm']['chi2_bao']:.4f})")
    log(f"chi2_GEO  = {R['geo']['chi2_total']:.6f} "
        f"(Plik {R['geo']['chi2_plik']:.4f} + BAO {R['geo']['chi2_bao']:.4f})")
    log(f"Delta chi2 GEO-LCDM = {dchi:+.6f}")
    log(f"Delta AIC = {S['delta_AIC']:+.6f}")
    log(f"Delta BIC = {S['delta_BIC']:+.6f}")
    log(f"best GEO f_dyn = {R['geo']['params']['f_dyn_test53']:.8f}")
    log(f"best GEO H0 = {100*R['geo']['params']['h']:.5f} km/s/Mpc")
    log(f"rd LCDM={R['lcdm']['rd_Mpc']:.6f} Mpc GEO={R['geo']['rd_Mpc']:.6f} Mpc")

if __name__=="__main__":
    main()
