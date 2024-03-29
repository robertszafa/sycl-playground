BENCHMARK := histogram_if

ifndef KERNEL
KERNEL := dynamic
endif
KERNEL_SRC := src/kernel_$(KERNEL).hpp
ifeq ($(KERNEL), dynamic_no_forward)
	KERNEL_SRC := src/kernel_dynamic.hpp
endif

ifndef Q_SIZE
Q_SIZE := 2
endif

# Store Queue
INC := ../include

SRC := src/main.cpp
HDR := $(KERNEL_SRC) $(INC)/store_queue.hpp
BIN := bin/$(BENCHMARK)_$(KERNEL)

ifeq ($(KERNEL), dynamic)
	BIN := bin/$(BENCHMARK)_$(KERNEL)_$(Q_SIZE)qsize
endif
ifeq ($(KERNEL), dynamic_no_forward)
	BIN := bin/$(BENCHMARK)_$(KERNEL)_$(Q_SIZE)qsize
endif


CXX := dpcpp
CXXFLAGS += -std=c++17 -O2 -D$(KERNEL)_sched -DQ_SIZE=$(Q_SIZE) -I$(INC)
CXXFLAGS += -qactypes
# CXXFLAGS += -Xsprofile
# CXXFLAGS += -g
# CXXFLAGS += -Xsghdl

HARDWARE_FLAGS := -Xshardware -DFPGA=1 
HARDWARE_FLAGS += -Xsboard=/opt/intel/oneapi/intel_a10gx_pac:pac_a10
# HARDWARE_FLAGS += -reuse-exe=$(BIN).fpga

SIMULATION_FLAGS := -Xssimulation -DFPGA=1 
# SIMULATION_FLAGS += -reuse-exe=$(BIN).fpga_sim


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

$(BIN).fpga_sim: $(BIN).dev.o | bin
	$(CXX) $(CXXFLAGS) -fintelfpga $< -o $@ $(SIMULATION_FLAGS)

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
