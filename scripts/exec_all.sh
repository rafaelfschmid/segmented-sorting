in=$1 # input files dir
out=$2 # time files dir

#./../scripts/exec.sh mergeseg.exe $in > $out/mergeseg.time
#./../scripts/exec.sh radixseg.exe $in > $out/radixseg.time
#./../scripts/exec.sh fixcub.exe $in > $out/fixcub.time
#./../scripts/exec.sh fixthrust.exe $in > $out/fixthrust.time
./../scripts/execnthrust.sh nthrust.exe $in > $out/nthrust.time

#./scripts/exec.sh src/fixmergemgpu.exe $in > $out/fixmergemgpu.time
#/scripts/exec.sh src/fixpassdiff.exe $in > $out/fixpassdiff.time
#./scripts/exec.sh src/fixpass.exe $in > $out/fixpass.time
#./scripts/exec.sh src/bitonicseg/bitonicseg.exe $in > $out/bitonicseg.time

