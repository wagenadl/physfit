#!/usr/bin/python3

import numpy as np
from scipy.optimize import curve_fit
import copy

maxfev = 10000


class PhysFit:
    def __init__(self):
        self.form = None # name of function
        self.foo = None # actual function
        self.p = None # fit parameters for fitting to (X ± SX, Y ± SY)
        self.s = None # standard errors on those parameters
        self.cov = None # full covariance matrix for the fit parameters
        self.chi2 = None # chi^2 value for the fit (iff SY given)
        self.sumsquares = None
        self.R2 = None # R-squared "coefficient of determination",

    def apply(self, xx=None):
        if xx is None:
            xx = self.xx
        xx = np.array(xx)
        return self.foo(xx, *self.p)

    def dfdp(self, xx):
        # the gradient vector at points xx
        K = len(self.p)
        res = []
        for k in range(K):
            p1 = self.p.copy()
            p2 = self.p.copy()
            dpk = 0.3 * self.s[k]
            p1[k] -= dpk / 2
            p2[k] += dpk / 2
            res.append((self.foo(xx, *p2) - self.foo(xx, *p1)) / dpk)
        return np.array(res)

    def errorat(self, xx):
        dfdp = self.dfdp(xx)
        K = len(self.p)
        xx = np.array(xx)
        res = np.zeros(xx.shape)
        for k in range(K):
            for k2 in range(K):
                res += dfdp[k] * dfdp[k2] * self.cov[k, k2]
        return res ** 0.5

    def __repr__(self):
        form = repr(self.foo) if self.form is None else self.form
        s = f'''
form: {form}
p:    {self.p}
s:    {self.s}
'''
        for k, line in enumerate(f"{self.cov}".split("\n")):
            if k:
                s += "      " + line + "\n"
            else:
                s += "cov:  " + line + "\n"
        if self.chi2 is not None:
            s += f'chi2: {self.chi2:.4g}\n'
        if self.R2 is not None:
            s += f'R2:   {self.R2:.4f}\n'
        return s + '\n'



def p0_power(x, y):
    p = np.polyfit(np.log(x), np.log(y), 1)
    return np.array([np.exp(p[1]), p[0]])


def p0_exp(x, y):
    p = np.polyfit(x, np.log(y), 1)
    return np.array([np.exp(p[1]), p[0]])


def p0_expc(x, y):
    lp1 = np.polyfit(x, y, 1)
    lp2 = np.polyfit(x, y, 2)
    sgnB = np.sign(lp2[0]) * np.sign(lp1[0])      
    sgnA = np.sign(lp2[0])
    y_ = np.unique(np.sort(y))
    if sgnA<0:
        y_ = np.flip(y_)
    if len(y_)==1:
        c0 = y_[0]
    else:
        c0 = y_[0] - 1*(y_[1]-y_[0])
    lp = np.polyfit(x, np.log((y-c0)*sgnA), 1)
    return np.array([sgnA*np.exp(lp[1]), lp[0], c0])


def p0_cos(x, y):
    def foo(x, a, b, c): return a*np.cos(b*x+c)
    p,s = curve_fit(foo, x, y)
    if p[0]<0:
        p[0] = -p[0]
        p[2] += np.pi
        if p[2] >= np.pi:
            p[2] -= 2*np.pi
    return p


forms = {
    'slope': ( 'A*x',
               lambda x, a: a*x,
               lambda x, a: a,
               lambda x, y: np.array(np.sum(x*y)/sum(x**2)) ),
    'linear': ( 'A*x + B',
                lambda x, a,b: a*x + b,
                lambda x, a,b: a,
                lambda x, y: np.polyfit(x, y, 1) ),
    'quadratic': ( 'A*x**2 + B*x + C',
                   lambda x, a,b,c: a*x**2 + b*x + c,
                   lambda x, a,b,c: 2*a*x + b,
                   lambda x, y: np.polyfit(x, y, 2) ),
    'power': ( 'A*x**B',
               lambda x, a,b: a*x**b,
               lambda x, a,b: a*b*x**(b-1),
               p0_power ),
    'log': ( 'A*log(x) + B',
             lambda x, a,b: a*np.log(x) + b,
             lambda x, a,b: a/x,
             lambda x, y: np.polyfit(np.log(x), y, 1) ),
    'exp': ( 'A*exp(B*x)',
             lambda x, a,b: a*np.exp(b*x),
             lambda x, a,b: a*b*np.exp(b*x),
             p0_exp ),
    'expc': ( 'A*exp(B*x) + C',
              lambda x, a,b,c: a * np.exp(b*x) + c,
              lambda x, a,b,c: a*b * np.exp(b*x),
              p0_expc ),
    'cos': ( 'A*cos(B*x + C)',
             lambda x, a,b,c: a * np.cos(b*x + c),
             lambda x, a,b,c: -a*b * np.sin(b*x + c),
             p0_cos )
}
    

def physfit(f, x, y, sy=None, sx=None, p0=None, sxy=None):
    '''PHYSFIT Function fitting using errors on both X and Y
    
    fit = PHYSFIT(fform, x, y, sy, sx) fits the data (X ± SX, Y ± SY)
    to the given functional form FFORM.

    The result is a class containing:
    
      p:    fit parameters for fitting to (X ± SX, Y ± SY).
      s:    standard errors on those parameters.
      cov:  full covariance matrix for the fit parameters.
      chi2: χ² value for the fit (only defined if SY is given).
      R2:   R-squared "coefficient of determination"

    as well as a method “apply” that can be used to apply the fit
    to new X values.
    
    FFORM may be one of several standard forms:
    
      slope:      y = A x
      linear:     y = A x + B
      quadratic:  y = A x^2 + B x + C
      poly-N:     y = A x^N + B x^(N-1) + ... + Z
      power:      y = A x^B
      exp:        y = A exp(B x)
      expc:       y = A exp(B x) + C
      log:        y = A log(x) + B
      cos:        y = A cos(B x + C)
    
    Alternatively, a callable may be given that takes a first vector
    argument of x-values followed by the fit parameters as separate
    arguments.
    
    PHYSFIT(fform, x, y, sy, sx, p0) specifies initial parameters
    values.  This is optional for the standard forms, but required for
    the functional form.
    
    Leave out SX to not specify errors on X.  Leave SY out to not
    specify errors at all. If you only have errors on X and not on Y,
    perform the fits backwards (i.e., fit X against Y instead of Y
    against X.)
    
    To specify known correlations between the errors in X and Y
    observations, use optional argument SXY to specify the covariance
    (not its square root!).

    If optimization fails, a RunTimeError is raised. If covariance
    cannot be estimated, an OptimizeWarning is generated. See
    numpy.seterr and the python warnings module for more information.

    Note: The S and COV outputs are corrected for CHI2. That is, even
    if you get a large CHI2, the returned S and COV are
    reasonable. You are just being informed that the SX or SY you
    passed in may have been overly optimistic.

    '''

    x = np.array(x)
    yy = np.array(y)

    if sxy is None:
        sxy = 0*x
    else:
        sxy = 0*x + np.array(sxy)
    if sy is None:
        sy = 0*x
    else:
        sy = 0*x + np.array(sy)
    if sx is None:
        sx = 0*x
    else:
        sx = 0*x + np.array(sx)

    if type(f)==str:
        if f in forms:
            form, foo, dfdx, fp0 = forms[f]
            p0 = fp0(x, y)
        elif f.startswith('poly-'):
            # Following is rather ugly way to synthesize functions with
            # call signatures like "def poly(x,a,b,c): return a*x**2+b*x+c"
            # This is necessary because curve_fit insists on passing each
            # parameter as a separate argument.
            N = int(f[5:])
            if N<=0 or N>20:
                raise ValueError(f'Bad polynomic functional form {f}')
            form = []
            for n in range(N+1):
                form.append('%c*x**%i' % (chr(ord('A')+n), N-n))
            form = ' + '.join(form)
            pars = []
            for n in range(N+1):
               pars.append(chr(ord('a')+n))
            pars = ','.join(pars)
            poly = []
            for n in range(N+1):
                poly.append('%c*x**%i' % (chr(ord('a')+n), N-n))
            poly = '+'.join(poly)
            ddxpoly = []
            for n in range(N):
                ddxpoly.append('%i*%c*x**%i' % (N-n,chr(ord('a')+n), N-1-n))
            ddxpoly = '+'.join(ddxpoly)
            foo = eval('lambda x, ' + pars + ': ' + poly)
            dfdx = eval('lambda x, ' + pars + ': ' + ddxpoly)
            p0 = np.polyfit(x, y, N)
        else:
            raise ValueError(f'Unknown functional form: {f}')
    else:
        dfdx = None
        form = repr(f)
        foo = f
        if p0 is None:
            raise ValueError('Must provide parameters for functional form')
        p0 = np.array(p0)

    # Now foo is the function to be fitted, p0 are initial values

    if len(p0.shape):
        df = len(p0)
    else:
        df = 1
    N = len(x)
    fits = []
    
    ## --------- Fit without SX or SY ----------
    p,cov = curve_fit(foo, x, y, p0, maxfev=maxfev)
    fit = PhysFit()
    fit.p = p
    fit.s = np.sqrt(np.diag(cov))
    fit.cov = cov
    fit.sumsquares = np.sum((foo(x, *p) - y)**2)
    EPS = 1e-99
    fit.R2 = 1 - fit.sumsquares / (np.sum((y - np.mean(y))**2) + EPS)
    fits.append(fit)

    ## ---------- Fit with only SY -------------
    if np.max(sy)>0:
        p, cov = curve_fit(foo, x, y, p0, sigma=sy, maxfev=maxfev)
        fit = PhysFit()
        fit.p = p
        fit.sumsquares = np.sum((foo(x, *p) - y)**2 / sy**2)
        fit.chi2 = fit.sumsquares / (N - df)
        fit.s = np.sqrt(np.diag(cov))
        ss0 = np.sum((y - np.mean(y))**2 / sy**2)
        fit.R2 = 1 - fit.sumsquares / (ss0 + EPS)
        fit.cov = cov
        fits.append(fit)

    ## ---------- Fit with SX and SY -------------
    if np.max(sy)>0 and np.max(sx)>0:
        # Set effective uncertainty to
        #
        #   sy_eff^2 = sy^2 + (df/dx)^2 * sx^2
        #
        # We iterate several times to get closer to optimal estimates of df/dx
  
        fit = copy.copy(fit)
        ok = False
        for iter in range(5):
            if fit.p is None or any(np.isnan(fit.p)) \
               or any(np.isnan(fit.s)):
                p0 = fits[0].p
            else:
                p0 = fit.p
            if dfdx is None:
                # Primitive attempt to calculate derivative
                yR = foo(x+1e-10, *p0)
                yL = foo(x-1e-10, *p0)
                dfdx_ = (yR-yL) / 2e-10
                if any(np.isnan(dfdx_)):
                    warnings.warn('Some uncertainties on X were dropped near edge of function domain.')
                    dfdx_[np.isnan(dfdx_)] = 0
            else:
                dfdx_ = dfdx(x, *p0)
    
            sy_eff = np.sqrt(sy**2 + dfdx_**2*sx**2 + dfdx_*sxy)
            try:
                p, cov = curve_fit(foo, x, y, sigma=sy_eff, maxfev=maxfev)
                fit.p = p
                fit.sumsquares = np.sum((foo(x, *p) - y)**2 / sy_eff**2)
                fit.chi2 = fit.sumsquares / (N - df)
                fit.s = np.sqrt(np.diag(cov))
                ss0 = np.sum((y - np.mean(y))**2 / sy_eff**2)
                fit.R2 = 1 - fit.sumsquares / (ss0 + EPS)
                fit.cov = cov
                ok = True
            except Exception as e:
                err = e
                pass
        if not ok:
            raise err
        fits.append(fit)

    for k in range(len(fits)):
        fits[k].form = form
        fits[k].xx = x # for “apply”
        fits[k].foo = foo
        
    return fits[-1]
