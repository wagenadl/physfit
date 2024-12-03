#!/usr/bin/python3


## physfit - Function fitting with errors on both x and y
## Copyright (C) 2024  Daniel A. Wagenaar
## 
## This program is free software: you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation, either version 3 of the
## License, or (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from scipy.optimize import curve_fit

__all__ = ["FitResults", "fit"]

maxfev = 10000


class FitResults:
    """The results from a function fit

    This object has the following attributes:

    - p               - the fitted parameters as a vector p = [A, B, C, ...]
    - s               - the uncertainties as a vector
    - cov             - the covariance matrix of the parameters
    - f               - the fitted function as a callable, i.e., f: X ↦ Y
    - df              - a callable to calculate fit uncertainties,
                        i.e., df: X ↦ dY.
    - R2              - R² “coefficient of determination”
    - chi2            - χ² value for the fit (if sy was given)
    - form            - a string representation of the form of the function

    In addition, the following are defined for convenience:

    - A, B, C, ...    - the fitted parameter values individually
    - sA, sB, sC, ... - the uncertainties on those parameters
    - sAA, sAB, ...   - the coefficients of the covariance matrix of the
                        parameters
    
    For compatibility with older versions, “apply” is provided as a
    synonym for “f”. Additionally, the object may be called directly
    (using “(...)” syntax) with the same effect as calling its “f”
    attribute.

    If “fit” is a FitResults object obtained as the result of fitting
    with specified uncertainties on x and/or y, the fit results
    without consideration of those uncertainties may be accessed as
    “fit[0]”. The fit results with consideration of y but not x
    uncertainties as “fit[1]”. (For compatibility with older versions,
    the final fit results may be accessed as “fit[-1]” as well.)

    Note: The “s” and “cov” attributes are corrected for χ². That is,
    even if you get a large χ², the “s” and “cov” are reasonable. You
    are just being informed that the uncertainties you passed in may
    have been overly optimistic.

    """

    def __init__(self):
        self.form = None # name of function
        self.foo = None # actual function
        self.p = None # fit parameters for fitting to (X ± SX, Y ± SY)
        self.s = None # standard errors on those parameters
        self.cov = None # full covariance matrix for the fit parameters
        self.chi2 = None # chi^2 value for the fit (iff SY given)
        self.sumsquares = None
        self.R2 = None # R-squared "coefficient of determination",
        self.hassy = False
        self.hassx = False

    def apply(self, xx=None):
        """Equivalent to f(), for compatibility with old code"""
        if xx is None:
            xx = self.xx
        xx = np.array(xx)
        return self.f(xx)

    def f(self, xx):
        """This is the fitted function
        """
        return self.foo(xx, *self.p)

    def _dfdp(self, xx):
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

    def df(self, xx):
        """Uncertainties of fit result at location in input space
        """
        dfdp = self._dfdp(xx)
        K = len(self.p)
        xx = np.array(xx)
        res = np.zeros(xx.shape)
        for k in range(K):
            for k2 in range(K):
                res += dfdp[k] * dfdp[k2] * self.cov[k, k2]
        return res ** 0.5

    def __repr__(self):
        K = len(self.p)
        hdr = f"Fit of {len(self.xx)} points "
        if self.hassx:
            hdr += "with specified errors on X and Y"
        elif self.hassy:
            hdr += "with specified errors on Y"
        else:
            hdr += "without specified errors"
        bits = self.form.split(" ")
        if bits[0] == "<function":
            if len(bits) >= 2 and bits[1] != "<lambda>":
                res = [f"{hdr} to:", "", f"  y = {bits[1]}(x)", ""]
            else:
                res = [f"{hdr} to lambda:", ""]
        else:
            res = [f"{hdr} to:", "", f"  y = {self.form}", ""]

        for k in range(K):
            p = self.p[k]
            s = self.s[k]
            prec = int(-min(np.floor(np.log10(s/3.5)),
                            np.floor(np.log10(np.abs(p)/10 + 1e-99))))
            if prec < 0:
                prec = 0
            # For python < 3.6:
            #   fmt = f"{{:.{prec}f}}"
            #   line = f"  {chr(65 + k)} = {fmt} ± {fmt}"
            #   res.append(line.format(p, s)
            res.append(f"  {chr(65 + k)} = {p:.{prec}f} ± {s:.{prec}f}")
        res.append("")
        # For cov, use smallest diag term to figure digits of precision
        # and abs. largest to figure space needs
        mag = 0
        prec = 0
        for k in range(K):
            prec = max(prec, int(-np.floor(np.log10(self.cov[k,k]/10))))
        if prec < 0:
            prec = 0
        mag = int(np.max(np.log10(np.round(np.abs(self.cov + 1e-99), prec))))
        if mag < 0:
            mag = 0
        if prec < 0:
            spc = mag + 2
        else:
            spc = mag + 3 + prec            
        for k in range(K):
            if k == (K-1)//2:
                pfx = "cov = ["
            else:
                pfx = "      ["
            bits = [f"{c:{spc}.{prec}f}" for c in self.cov[:,k]]
            res.append(pfx + " ".join(bits) + "]")

        res.append("") 
        res.append(f"R² = {self.R2:.3f}")
        if self.chi2 is not None:
            res.append(f"χ² = {self.chi2:4g}")
            
        return "\n".join(res)

    def __getitem__(self, k):
        K = len(self.fits) + 1
        # we ourselves are implicitly part of the vector
        if k < 0:
            k = K + k
        if k == K - 1:
            return self
        if k < 0 or k >= K:
            raise IndexError("Prefit index out of range")
        return self.fits[k]

    def __call__(self, xx):
        return self.f(xx)

    def __getattr__(self, a):
        K = len(self.p)
        if len(a) == 1:
            k = ord(a[0]) - ord('A')
            if k >= 0 and k < K:
                return self.p[k]
        elif len(a) == 2 and a[0] == 's':
            k = ord(a[1]) - ord('A')
            if k >= 0 and k < K:
                return self.s[k]
        elif len(a) == 3 and a[0] == 's':
            k = ord(a[1]) - ord('A')
            k2 = ord(a[2]) - ord('A')
            if k >= 0 and k < K:
                if k2 >= 0 and k2 < K:
                    return self.cov[k,k2]
        raise AttributeError(f"'FitResults' object has no attribute '{a}'")

    
def _p0_power(x, y):
    p = np.polyfit(np.log(x), np.log(y), 1)
    return np.array([np.exp(p[1]), p[0]])


def _p0_exp(x, y):
    p = np.polyfit(x, np.log(y), 1)
    return np.array([np.exp(p[1]), p[0]])


def _p0_expc(x, y):
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


def _p0_cos(x, y):
    def foo(x, a, b, c): return a*np.cos(b*x+c)
    p,s = curve_fit(foo, x, y)
    if p[0]<0:
        p[0] = -p[0]
        p[2] += np.pi
        if p[2] >= np.pi:
            p[2] -= 2*np.pi
    return p


_forms = {
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
               _p0_power ),
    'log': ( 'A*log(x) + B',
             lambda x, a,b: a*np.log(x) + b,
             lambda x, a,b: a/x,
             lambda x, y: np.polyfit(np.log(x), y, 1) ),
    'exp': ( 'A*exp(B*x)',
             lambda x, a,b: a*np.exp(b*x),
             lambda x, a,b: a*b*np.exp(b*x),
             _p0_exp ),
    'expc': ( 'A*exp(B*x) + C',
              lambda x, a,b,c: a * np.exp(b*x) + c,
              lambda x, a,b,c: a*b * np.exp(b*x),
              _p0_expc ),
    'cos': ( 'A*cos(B*x + C)',
             lambda x, a,b,c: a * np.cos(b*x + c),
             lambda x, a,b,c: -a*b * np.sin(b*x + c),
             _p0_cos )
}
    

def fit(form, x, y, sy=None, sx=None, sxy=None, p0=None):
    '''Function fitting using errors on both X and Y

    Arguments
    ---------

       form: functional form (see below)
       x: x data (a vector)
       y: y data (a vector)
       sy: optional uncertainties on y data (scalar or vector)
       sx: optional uncertainties on x data (ditto)
       sxy: optional covariances between x and y errors (ditto)
       p0: optional initial parameters

    Returns
    -------

    A FitResults object containing the results of the fit.

    Supported functional forms
    --------------------------

    The following standard forms are accepted:

        slope:      y = A x
        linear:     y = A x + B
        quadratic:  y = A x^2 + B x + C
        power:      y = A x^B
        exp:        y = A exp(B x)
        expc:       y = A exp(B x) + C
        log:        y = A log(x) + B
        cos:        y = A cos(B x + C)

    In addition, the form “poly-N” is accepted for any 0 ≤ N ≤ 20:

        poly-N:     y = A x^N + B x^(N-1) + ... + Z
    
    Alternatively, a callable may be given that takes a first vector
    argument of x-values followed by the fit parameters as separate
    arguments.

    When one of the named forms is used, initial parameters are
    optional. However, for the function form, they are required.

    Fitting with or without specified uncertainties is supported. But
    if you specify uncertainties on X, you must also specify
    uncertainties on Y. (If you cannot, consider fitting backwards,
    i.e., fit X against Y instead of Y against X.) Specified
    uncertainties are interpreted as 1σ values.
    
    To specify known correlations between the errors in X and Y
    observations, use optional argument SXY to specify the covariance
    (not its square root!).

    If optimization fails, a RunTimeError is raised. If covariance
    cannot be estimated, an OptimizeWarning is generated. See
    numpy.seterr and the python warnings module for more information.

    This uses the scipy.optimize.curve_fit function with default "lm"
    method.

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

    if type(form)==str:
        if form in _forms:
            form, foo, dfdx, fp0 = _forms[form]
            p0 = fp0(x, y)
        elif form.startswith('poly-'):
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
        foo = form
        form = repr(form)
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
    fit = FitResults()
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
        fit = FitResults()
        fit.p = p
        fit.sumsquares = np.sum((foo(x, *p) - y)**2 / sy**2)
        fit.chi2 = fit.sumsquares / (N - df)
        fit.s = np.sqrt(np.diag(cov))
        ss0 = np.sum((y - np.mean(y))**2 / sy**2)
        fit.R2 = 1 - fit.sumsquares / (ss0 + EPS)
        fit.cov = cov
        fit.hassy = True
        fits.append(fit)

    ## ---------- Fit with SX and SY -------------
    if np.max(sy)>0 and np.max(sx)>0:
        # Set effective uncertainty to
        #
        #   sy_eff^2 = sy^2 + (df/dx)^2 * sx^2
        #
        # We iterate several times to get closer to optimal estimates of df/dx
        fit = FitResults()
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
        fit.hassy = True
        fit.hassx = True
        fits.append(fit)

    for k in range(len(fits)):
        fits[k].form = form
        fits[k].xx = x # for “apply”
        fits[k].foo = foo

    fit = fits.pop()
    fit.fits = fits
    return fit
