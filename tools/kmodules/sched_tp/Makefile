obj-m += sched_tp.o

EXTRA_CFLAGS = -I$(src)

VMLINUX_DEPS_H = vmlinux_deps.h
VMLINUX_H = vmlinux.h

VMLINUX_DEPS_TXT = vmlinux_deps.txt
VMLINUX_TXT = vmlinux.txt

VMLINUX = $(KERNEL_SRC)/vmlinux

all: $(VMLINUX_H)
	make -C $(KERNEL_SRC) M=$(PWD) modules

install:
	make -C $(KERNEL_SRC) M=$(PWD) modules_install

clean:
	make -C $(KERNEL_SRC) M=$(PWD) clean
	rm -f $(VMLINUX_H) $(VMLINUX_DEPS_H)

$(VMLINUX_DEPS_H): $(VMLINUX_DEPS_TXT) $(VMLINUX)
	pahole -C file://$(VMLINUX_DEPS_TXT) $(VMLINUX) > $@

$(VMLINUX_H): $(VMLINUX_DEPS_H) $(VMLINUX_TXT) $(VMLINUX)
	pahole -C file://$(VMLINUX_TXT) $(VMLINUX) > $@
