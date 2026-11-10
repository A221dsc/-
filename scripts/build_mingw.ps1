param(
    [string]$LsiRoot = 'D:/anaconda/Library',
    [string]$Build = 'build'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $Build | Out-Null
$env:PATH = "$LsiRoot/bin;" + $env:PATH
$flags = @('-std=c++20', '-O3', '-DNDEBUG', '-Isrc', "-I$LsiRoot/include")
$common = @(
    'src/indexes/grid.cpp', 'src/indexes/quad.cpp', 'src/indexes/boost.cpp',
    'src/indexes/lsi.cpp', 'src/indexes/regional.cpp', 'src/indexes/selector.cpp',
    'src/validation/oracle.cpp'
)
$objects = foreach ($src in $common) {
    $obj = Join-Path $Build ([IO.Path]::GetFileNameWithoutExtension($src) + '.o')
    & g++ @flags -c $src -o $obj
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $obj
}
$dll = "$LsiRoot/bin/spatialindex_c-64.dll"
& g++ @flags @objects tests/index_tests.cpp $dll -o "$Build/index_tests.exe"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& g++ @flags @objects src/workload/m3.cpp tests/m3_tests.cpp $dll -lpsapi -o "$Build/m3_tests.exe"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& g++ @flags @objects src/workload/replay.cpp src/benchmark/main.cpp $dll -lpsapi -o "$Build/spatial_bench.exe"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& g++ @flags @objects src/workload/m3.cpp src/benchmark/m3.cpp $dll -lpsapi -o "$Build/spatial_m3.exe"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
'#include <boost/version.hpp>' | & g++ @flags -E -dM -x c++ - | Select-String 'define BOOST_LIB_VERSION '
& "$Build/index_tests.exe"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& "$Build/m3_tests.exe"
exit $LASTEXITCODE
