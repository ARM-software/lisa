#! /bin/sh

set -e

OUT=$1
MODULE_SRC=$2
MODULE_OBJ=$3
VENV=$4
CC=$5
PRIVATE_TYPES_TXT=$6
VMLINUX=$7
UNUSED_KERNEL_PRIVATE_TYPES_PREFIX=$8
KALLSYMS=$9

# Some options are not upstream (yet) but they have all be published on the
# dwarves mailing list
#
# Options:
# -F btf,dwarf: Use BTF first
# -E: Expand nested type definitions
# --suppress_force_paddings: Remove the "artificial" padding members pahole adds
#   to make padding more visible. They are not always valid C syntax and can
#   break build
# --skip_missing: Keep going if one of the types is not found
# --expand_types_once (non upstream): Only expand a given type once, to avoid type redefinition
#   (C does not care about nesting types, there is a single namespace).
#
# We then post-process the header to add a prefix to each type expanded by -E
# that was not explicitly asked for. This avoids conflicting with type
# definitions that would come from public kernel headers, while still allowing
# easy attribute access.
pahole -F btf,dwarf -E --compile --suppress_force_paddings --show_only_data_members --skip_missing --expand_types_once -C "file://$PRIVATE_TYPES_TXT" "$VMLINUX" > "$MODULE_OBJ/_header"

# Create a header with all the types we can for introspection purposes. This
# will not be included in any of the module sources, so all we care about is
# that pycparser can parse it. Specifically, some types may be re-defined with
# incompatible definitions. This is expected as various drivers use the same
# struct names.
pahole -F btf,dwarf -E --expand_types_once --suppress_force_paddings --suppress_aligned_attribute --suppress_packed --show_only_data_members --compile --fixup_silly_bitfields "$VMLINUX" > "$MODULE_OBJ/_full_header"

# Strip comments to avoid matching them with the sed regex.
"$CC" -P -E - < "$MODULE_OBJ/_header" > "$MODULE_OBJ/_header_no_comment"
# Create forward declaration of every type to ensure the header can be parsed correctly.
sed -r -n 's/.*(struct|union|enum) ([0-9a-zA-Z_]*) .*/\1 \2;/p' "$MODULE_OBJ/_header_no_comment" | sort -u > "$MODULE_OBJ/_fwd_decl"

# Rename all the kernel private types we are not directly interested in to avoid any clash
cat "$MODULE_OBJ/_fwd_decl" "$MODULE_OBJ/_header_no_comment" > "$MODULE_OBJ/_header"
. "$VENV/bin/activate" && python3 "$MODULE_SRC/introspect_header.py" --header "$MODULE_OBJ/_header" --type-prefix "$UNUSED_KERNEL_PRIVATE_TYPES_PREFIX" --non-renamed-types "$PRIVATE_TYPES_TXT" > "$MODULE_OBJ/_renamed_header"

# Create type introspection macros
. "$VENV/bin/activate" && python3 "$MODULE_SRC/introspect_header.py" --header "$MODULE_OBJ/_full_header" --introspect --kallsyms "$KALLSYMS" --kernel-features "$MODULE_SRC/kernel_features.json" >> "$MODULE_OBJ/_introspection_header"

# Build the final header
printf '#pragma once\n' > "$OUT"
cat "$MODULE_OBJ/_introspection_header" "$MODULE_OBJ/_renamed_header" >> "$OUT"
