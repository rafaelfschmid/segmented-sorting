in=$1 # input files dir
out=$2 # time files dir

#./../scripts/exec.sh mergeseg.exe $in > $out/mergeseg.time
#./../scripts/exec.sh radixseg.exe $in > $out/radixseg.time
#./../scripts/exec.sh fixcub.exe $in > $out/fixcub.time
#./../scripts/exec.sh fixthrust.exe $in > $out/fixthrust.time
./scripts/exec.sh src/fixseq.exe $in > $out/fixseq.time
./scripts/exec.sh src/mseq.exe $in > $out/mseq.time
#./../scripts/execnthrust.sh nthrust.exe $in > $out/nthrust.time

