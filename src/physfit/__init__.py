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


"""
physfit - Function fitting with errors on both x and y
This package provides a simple framework for fitting functions to
2-dimensional data. What makes physfit stand out is that it allows you
to specify uncertainties not only on y-coordinates, but also on
x-coordinates."""

from .physfit import physfit

