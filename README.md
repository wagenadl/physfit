# physfit — Function fitting with errors on both x and y

## Introduction

This package provides a simple framework for fitting functions to
2-dimensional data. What makes physfit stand out is that it allows you
to specify uncertainties not only on y-coordinates, but also on
x-coordinates.


## Installation

Will soon be as easy as

    pip install physfit
    
For now, please copy the file
[“physfit.py”](./src/ppersist/physfit.py) into your local Python
environment.


## Examples



    import numpy as np
    np.set_printoptions(precision=4) 
    from physfit import physfit
    rng = np.random.default_rng(12345)
    import matplotlib.pyplot as plt
    plt.ion()
    
    def plotwitherrors(xx, yy, sx, sy):
        for x, y, dx, dy in zip(xx, yy, sx, sy):
            plt.plot([x, x], [y-dy, y+dy], '-', color=(.7, .7, .7))
            plt.plot([x-dx, x+dx], [y, y], '-', color=(.7, .7, .7))
        plt.plot(xx, yy, 'ko')
    

    xx = [0.1, 1.05, 1.95, 3.1, 5.3]
    yy = [5.1, 5.9, 7.0, 7.9, 9.1]
    sy = [0.1, 0.1, 0.1, 0.1, 0.3]
    sx = [0.1, 0.1, 0.1, 0.1, .8]
    
    p1 = physfit('linear', xx, yy)
    p2 = physfit('linear', xx, yy, sy=sy)
    p3 = physfit('linear', xx, yy, sx=sx, sy=sy)
    
    plt.figure(1)
    plt.clf()
    plotwitherrors(xx, yy, sx, sy)
    x0a = np.arange(-0.5, 5.6, .1)
    for p in [p1, p2, p3]:
        plt.plot(x0a, p.apply(x0a), '-')
    
    
    
    xx = np.arange(0, 8*np.pi, np.pi/30)
    yy0 = np.sin(xx)
    N = len(xx)
    sy = 0.5
    yy = yy0 + sy * rng.normal(size=N)
    
    p1 = physfit('cos', xx, yy)

    plt.figure(2)
    plt.clf()
    plotwitherrors(xx, yy, 0*xx, sy+0*yy)
    plt.plot(xx, p1.apply(xx))
    
    
    xx0 = np.arange(-3, 3, 0.2)
    N = len(xx0)
    yy0 = np.exp(-0.5 * xx0**2)
    sy0 = 0.1
    yy = yy0 + sy0 * rng.normal(size=N) * np.sqrt(yy0)
    sy = sy0 * np.sqrt(yy0)
    sx = 0.1 * (1 + xx0**2)
    xx = xx0 + sx * rng.normal(size=N)
    
    form = lambda xx, a, b: a * np.exp(-b*xx**2)
    
    p1 = physfit(form, xx, yy, p0=[1, 1])
    p2 = physfit(form, xx, yy, sy=sy, p0=[1, 1])
    p3 = physfit(form, xx, yy, sy=sy, sx=sx, p0=[1, 1])
    
    plt.figure(3)
    plt.clf()
    plotwitherrors(xx, yy, sx, sy)
    x0a = np.arange(-4, 4, 0.1)
    for p in [p1, p2, p3]:
        plt.plot(x0a, p.apply(x0a))
        
    
    xx = np.arange(3, 8, .5)
    N = len(xx)
    yy0 = 2 * xx**2 - 3
    sy = 10
    yy = yy0 + sy * rng.normal(size=N)
    
    p1 = physfit("quadratic", xx, yy)
    p2 = physfit(lambda xx, a, b: a*xx**2 + b, xx, yy, p0=[1, 0])
    
    f, ax = plt.subplots(1, 2)
    for a, p in zip(ax, [p1, p2]):
        plt.axes(a)
        x0a = np.arange(0., 8.5, 0.1)
        plt.fill_between(x0a, p.apply(x0a) - p.errorat(x0a), 
                   p.apply(x0a) + p.errorat(x0a),  alpha=.4)
        plt.plot(x0a, p.apply(x0a))
        plotwitherrors(xx, yy, 0*xx, sy+0*yy)
 
