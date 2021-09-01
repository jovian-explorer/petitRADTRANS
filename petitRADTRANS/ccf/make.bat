@echo off

rem Build the Fortran modules for pRT using f2py (Windows).

rem Execute in current directory
pushd "%~dp0"
rem Build
f2py -c --opt="-O3 -funroll-loops -ftree-vectorize -ftree-loop-optimize -msse -msse2 -m3dnow" -m rebin_give_width rebin_give_width.f90

rem Move dlls into parent directory, necessary in Windows for Python to recognise the modules
move .\rebin_give_width\.libs\*.dll .\

rem Remove auto-generated directories
rmdir /S /Q .\rebin_give_width
popd