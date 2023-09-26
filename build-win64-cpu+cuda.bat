@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Add the path to the OpenCV DLLs
set PATH=%PATH%;"C:\Users\tjerf\Downloads\opencv\build\x64\vc16\bin"

call "vcvarsall.bat" amd64

:: Include the path to Python headers
set PYTHON_INCLUDE="C:/Python311/include"
:: Include the path to Python libs
set PYTHON_LIB="C:/Python311/libs"

nvcc -gencode arch=compute_86,code=sm_86 src/ebsynth.cpp src/ebsynth_cpu.cpp src/ebsynth_cuda.cu src/wrapper.cpp -DNDEBUG -O3 -I "include" -I %PYTHON_INCLUDE% -I "C:/Users/tjerf/vcpkg/installed/x64-windows-static/include" -I "C:/Users/tjerf/vcpkg/installed/x64-windows/include" -o "bin/ebsynth.pyd" -Xcompiler "/openmp /fp:fast" -Xlinker "/IMPLIB:lib/ebsynth.lib" -shared "C:/Python311/libs/python311.lib" "C:/Users/tjerf/Downloads/opencv/build/x64/vc16/lib/opencv_world480.lib" -DEBSYNTH_API=__declspec(dllexport) -w || goto error

del dummy.lib;dummy.exp 2> NUL
goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
