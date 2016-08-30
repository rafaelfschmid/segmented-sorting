in=$1 # dir input files
result=$2 # correct result files
error=$3 # dir errors

#echo "FIXCUB"
#./scripts/test.sh fixcub.exe $in $result $error/fixcub
#echo "FIXMERGEMGPU"
#./scripts/test.sh fixmergemgpu.exe $in $result $error/fixmergemgpu
#echo "FIXTHRUST"
#./scripts/test.sh fixthrust.exe $in $result $error/fixthrust
#echo "MERGESEG"
#./scripts/test.sh mergeseg.exe $in $result $error/mergeseg
#echo "RADIXSEG"
#./scripts/test.sh radixseg.exe $in $result $error/radixseg
echo "BITONICSEG"
./scripts/test.sh bitonicseg.exe $in $result $error/bitonicseg
#echo "FIXMERGE"
#./scripts/test.sh fixsort/mergesort/mergesort.exe $in $result $error/fixmerge
#echo "FIXODDEVEN"
#./scripts/test.sh fixsort/oddevensort/oddevensort.exe $in $result $error/fixoddeven
#echo "FIXQUICK"
#./scripts/test.sh fixsort/quicksort/quicksort.exe $in $result $error/fixquick

