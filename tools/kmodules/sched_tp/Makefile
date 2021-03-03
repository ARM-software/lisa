obj-m += sched_tp.o

EXTRA_CFLAGS = -I$(src)

VMLINUX_DEPS_H = vmlinux_deps.h
VMLINUX_H = vmlinux.h

VMLINUX_DEPS_TXT = $(MODULE_SRC)/vmlinux_deps.txt
VMLINUX_TXT = $(MODULE_SRC)/vmlinux.txt

VMLINUX = $(KERNEL_SRC)/vmlinux

.PHONY: all build install clean

all: install

build: $(VMLINUX_H)
	$(MAKE) -C $(KERNEL_SRC) M=$(MODULE_SRC) modules

install: build
	$(MAKE) -C $(KERNEL_SRC) M=$(MODULE_SRC) modules_install

clean:
	rm -f $(VMLINUX_H) $(VMLINUX_DEPS_H)

$(VMLINUX_DEPS_H): $(VMLINUX_DEPS_TXT) $(VMLINUX)
	pahole -C file://$(VMLINUX_DEPS_TXT) $(VMLINUX) > $@

$(VMLINUX_H): $(VMLINUX_DEPS_H) $(VMLINUX_TXT) $(VMLINUX)
	pahole -C file://$(VMLINUX_TXT) $(VMLINUX) > $@
