ifndef KERNEL
KERNEL := dynamic
endif

SRC := src/main.cpp
HDR := src/kernel_$(KERNEL).hpp
BIN := bin/if_else_mul_$(KERNEL)
# BIN := bin/if_else_mul_$(KERNEL)_separate_read_write

CXX := dpcpp
CXXFLAGS += -std=c++17 -O2 -D$(KERNEL)_sched 
# CXXFLAGS += -Xsprofile
# CXXFLAGS += -g
# CXXFLAGS += -Xsghdl

HARDWARE_FLAGS := -Xshardware -DFPGA=1 
HARDWARE_FLAGS += -reuse-exe=$(BIN).fpga
# -Xssfc-exit-fifo-type=zero-latency

SIMULATION_FLAGS := -Xssimulation -DFPGA=1 
# -reuse-exe=$(BIN).fpga
# -Xssfc-exit-fifo-type=zero-latency


.PHONY: host fpga_emu fpga_hw

all: host
host: $(BIN)
fpga_emu: $(BIN).fpga_emu
fpga_sim: $(BIN).fpga_sim
fpga_hw: $(BIN).fpga_hw
report: $(BIN).a 


$(BIN): $(SRC) $(HDR) | bin
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

$(BIN).fpga_emu: $(SRC) $(HDR) | bin
	$(CXX) $(CXXFLAGS) -fintelfpga $< -o $@ -DFPGA_EMULATOR=1

# $(BIN).fpga_sim: $(SRC) $(HDR) | bin
$(BIN).fpga_sim: $(BIN).dev.o | bin
	$(CXX) $(CXXFLAGS) -fintelfpga $< -o $@ $(SIMULATION_FLAGS)

# To reuse fpga image add: -reuse-exe=$(BIN).fpga
$(BIN).fpga_hw: $(BIN).dev.o | bin
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
