ifndef KERNEL
KERNEL := dynamic
endif


CXX := dpcpp
CXXFLAGS += -std=c++17 -O2 -D$(KERNEL)_sched
# CXXFLAGS += -Xsprofile
# CXXFLAGS += -g
# CXXFLAGS += -Xsghdl

SRC := src/main.cpp
HDR := src/kernel_$(KERNEL).hpp
BIN := bin/gram_schmidt_$(KERNEL)

.PHONY: host fpga_emu fpga 

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

$(BIN).fpga_sim: $(SRC) $(HDR) | bin
	$(CXX) $(CXXFLAGS) -fintelfpga -Xssimulation $< -o $@ -DFPGA=1

# To reuse fpga image add: -reuse-exe=$(BIN).fpga
$(BIN).fpga_hw: $(BIN).dev.o | bin
	$(CXX) $(CXXFLAGS) -fintelfpga $< -o $@ -Xshardware

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
