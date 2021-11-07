.PHONY: pick clean download-gtest kepler maxwell pascal volta turing ampere check

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo "		make lib"
	@echo


cgbn_math.o: cgbn_math.cu
	nvcc -arch=sm_75 cgbn_math.cu -Xcompiler="-fPIC" -rdc=true -c -o cgbn_math.o -I./include -I./samples
cgbn_fp.o: cgbn_fp.cu
	nvcc -arch=sm_75 cgbn_fp.cu -Xcompiler="-fPIC" -c -o cgbn_fp.o -I./include -I./samples
cgbn_fp2.o: cgbn_fp2.cu
	nvcc -arch=sm_75 cgbn_fp2.cu -Xcompiler="-fPIC" -c -o cgbn_fp2.o -I./include -I./samples
cgbn_alt_bn128_g1.o: cgbn_alt_bn128_g1.cu
	nvcc -arch=sm_75 cgbn_alt_bn128_g1.cu -Xcompiler="-fPIC" -c -o cgbn_alt_bn128_g1.o -I./include -I./samples
cgbn_alt_bn128_g2.o: cgbn_alt_bn128_g2.cu
	nvcc -arch=sm_75 cgbn_alt_bn128_g2.cu -Xcompiler="-fPIC" -c -o cgbn_alt_bn128_g2.o -I./include -I./samples
cgbn_multi_exp.o: cgbn_multi_exp.cu
	nvcc -arch=sm_75 cgbn_multi_exp.cu -Xcompiler="-fPIC" -c -o cgbn_multi_exp.o -I./include -I./samples
cgbn_multi_exp_g2.o: cgbn_multi_exp_g2.cu
	nvcc -arch=sm_75 cgbn_multi_exp_g2.cu -Xcompiler="-fPIC" -c -o cgbn_multi_exp_g2.o -I./include -I./samples

lib: cgbn_math.o cgbn_fp.o cgbn_fp2.o cgbn_alt_bn128_g1.o cgbn_alt_bn128_g2.o cgbn_multi_exp.o cgbn_multi_exp_g2.o
#nvcc -arch=sm_75 cgbn_math.cu cgbn_fp.cu cgbn_fp2.cu cgbn_alt_bn128_g1.cu cgbn_alt_bn128_g2.cu cgbn_multi_exp.cu cgbn_multi_exp_g2.cu -Xcompiler -fPIC -shared -o libcgbn_math.so -I./include -I./samples
	nvcc -arch=sm_75 $^ -Xcompiler -fPIC -shared -o libcgbn_math.so -I./include -I./samples
	nvcc -arch=sm_75 $^ -lib -o libcgbn_math.a -I./include -I./samples
#nvcc -arch=sm_75 cgbn_math.cu cgbn_fp.cu cgbn_alt_bn128_g1.cu cgbn_multi_exp.cu -lib -o libcgbn_math.a -I./include -I./samples

test: lib test.cpp
	g++ -O3 test.cpp -o test -lcgbn_math -lcudart -lgmp -L./ -L/usr/local/cuda/lib64/ -I/usr/local/cuda/include -Iinclude/ -Isamples -I./
	g++ -O3 test_alt_bn128_g1.cpp -o test_alt_bn128_g1 -lcgbn_math -lcudart -lgmp -L./ -L/usr/local/cuda/lib64/ -I/usr/local/cuda/include -Iinclude/ -Isamples -I./

clean:
	make -C samples clean
	make -C unit_tests clean
	make -C perf_tests clean

download-gtest:
	wget 'https://github.com/google/googletest/archive/master.zip' -O googletest-master.zip
	unzip googletest-master.zip 'googletest-master/googletest/*'
	mv googletest-master/googletest gtest
	rmdir googletest-master
	rm -f googletest-master.zip

kepler: check
	make -C samples kepler
	make -C unit_tests kepler
	make -C perf_tests kepler

maxwell: check
	make -C samples maxwell
	make -C unit_tests maxwell
	make -C perf_tests maxwell

pascal: check
	make -C samples pascal
	make -C unit_tests pascal
	make -C perf_tests pascal

volta: check
	make -C samples volta
	make -C unit_tests volta
	make -C perf_tests volta

turing: check
	make -C samples turing
	make -C unit_tests turing
	make -C perf_tests turing

ampere: check
	make -C samples ampere
	make -C unit_tests ampere
	make -C perf_tests ampere

check:
	@if [ -z "$(GTEST_HOME)" -a ! -d "gtest" ]; then echo "Google Test framework required, see documentation"; exit 1; fi

