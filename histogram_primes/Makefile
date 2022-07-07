ifndef KERNEL
KERNEL := dynamic
endif

ifndef Q_SIZE
Q_SIZE := 1
endif

ifndef A_SIZE
A_SIZE := 64
endif

# For Pipe Array headers, etc
# INC := /home/rob/git/oneAPI-samples/DirectProgramming/DPC++FPGA/include

SRC := src/main.cpp
HDR := src/kernel_$(KERNEL).hpp
BIN := bin/histogram_$(KERNEL)_$(A_SIZE)asize

ifeq ($(KERNEL), dynamic)
	BIN := bin/histogram_$(KERNEL)_$(Q_SIZE)qsize_$(A_SIZE)asize
endif


CXX := dpcpp
CXXFLAGS += -std=c++17 -O2 -D$(KERNEL)_sched -DQ_SIZE=$(Q_SIZE) -DA_SIZE=$(A_SIZE) -qactypes
# CXXFLAGS += -Xsprofile
# CXXFLAGS += -g
# CXXFLAGS += -Xsghdl

HARDWARE_FLAGS := -Xshardware -DFPGA=1 
# HARDWARE_FLAGS += -reuse-exe=$(BIN).fpga
# -Xssfc-exit-fifo-type=zero-latency

SIMULATION_FLAGS := -Xssimulation -DFPGA=1 
# -reuse-exe=$(BIN).fpga
# -Xssfc-exit-fifo-type=zero-latency


.PHONY: host fpga_emu fpga 

all: host
host: $(BIN)
fpga_emu: $(BIN).fpga_emu
fpga_sim: $(BIN).fpga_sim
fpga: $(BIN).fpga
report: $(BIN).a 


$(BIN): $(SRC) $(HDR) | bin
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

$(BIN).fpga_emu: $(SRC) $(HDR) | bin
	$(CXX) $(CXXFLAGS) -fintelfpga $< -o $@ -DFPGA_EMULATOR=1

$(BIN).fpga_sim: $(BIN).dev.o | bin
	$(CXX) $(CXXFLAGS) -fintelfpga $< -o $@ $(SIMULATION_FLAGS)

$(BIN).fpga: $(BIN).dev.o | bin
	$(CXX) $(CXXFLAGS) -fintelfpga $< -o $@ $(HARDWARE_FLAGS)

$(BIN).dev.o: $(SRC) $(HDR) | bin
	$(CXX) $(CXXFLAGS) -fintelfpga -c $< -o $@ -DFPGA=1 

# This is just for generating fpga resource report.
$(BIN).a:  $(BIN).dev.o $(HDR) | bin
	$(CXX) $(CXXFLAGS) -fintelfpga -fsycl-link $< -o $@ -Xshardware

# Make bin/ dir if doesn't exist
bin:
	mkdir $@

clean:
	rm -rf *.o *.d *.out *.mon *.emu *.aocr *.aoco *.prj *.fpga_emu *.fpga *.log *.a bin/*
