dir=$1

./parser.exe $dir/fixcub.time $dir/00fixcub.time
./parser.exe $dir/fixthrust.time $dir/00fixthrust.time
./parser.exe $dir/mergeseg.time $dir/00mergeseg.time
./parser.exe $dir/radixseg.time $dir/00radixseg.time
./parser1024.exe $dir/nthrust.time $dir/00nthrust.time

#./utils/parser.exe $dir/fixpass.time $dir/00fixpass.time
#./parser.exe $dir/fixpassdiff.time $dir/00fixpassdiff.time

