CXX=nvcc
LD=${CXX}
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std=c++11 -O3 -I${OPENBLAS_ROOT}/include
LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lopenblas

OBJS=cg_main.o cg.o matrix.o matrix_coo.o mmio.o cg_gpu.o

all: cgsolver

cgsolver: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

%.o:%.c        
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o:%.cu
	$(CXX) $(CXXFLAGS) -dc -c $< -o $@

clean:
	rm -f cgsolver *.o *~
