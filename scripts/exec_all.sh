in=$1 # input files dir
out=$2 # time files dir

./scripts/exec.sh src/fixmergemgpu.exe $in > $out/fixmergemgpu.time
#/scripts/exec.sh src/fixpassdiff.exe $in > $out/fixpassdiff.time
./scripts/exec.sh src/fixpass.exe $in > $out/fixpass.time
#./scripts/exec.sh src/bitonicseg/bitonicseg.exe $in > $out/bitonicseg.time
./scripts/exec.sh src/mergeseg.exe $in > $out/mergeseg.time
./scripts/exec.sh src/radixseg.exe $in > $out/radixseg.time
./scripts/exec.sh src/fixcub.exe $in > $out/fixcub.time
./scripts/exec.sh src/fixthrust.exe $in > $out/fixthrust.time
#./scripts/exec.sh src/nthrust.exe $in > $out/nthrust.time


