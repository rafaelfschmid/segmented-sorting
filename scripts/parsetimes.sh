dir=$1

#./parser.exe $dir/fixcub.time $dir/00fixcub.time
#./parser.exe $dir/fixthrust.time $dir/00fixthrust.time
#./parser.exe $dir/mergeseg.time $dir/00mergeseg.time
#./parser.exe $dir/radixseg.time $dir/00radixseg.time
#./parser1024.exe $dir/nthrust.time $dir/00nthrust.time
#./parser.exe $dir/fixseq.time $dir/00fixseq.time
#./parser.exe $dir/mseq.time $dir/00mseq.time
#./utils/parser.exe $dir/fixpass.time $dir/00fixpass.time
#./parser.exe $dir/fixpassdiff.time $dir/00fixpassdiff.time

./parser.exe $dir/kernel1.time $dir/00kernel1.time
./parser.exe $dir/kernel2.time $dir/00kernel2.time
./parser.exe $dir/kernel4.time $dir/00kernel4.time
./parser.exe $dir/kernel8.time $dir/00kernel8.time
./parser.exe $dir/kernel16.time $dir/00kernel16.time
./parser.exe $dir/kernel32.time $dir/00kernel32.time


