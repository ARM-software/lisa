/* SPDX-License-Identifier: GPL-2.0 */

/* Include here headers containing the types listed in intropsection.json.
 * We process the debug info of the compiled object file like we would do with
 * the BTF blob and generate the introspection_data.h header, which is then
 * consumed the exact same way as an out-of-tree build. This allows controlling
 * the format ofthat header for introspect_header.py.
 */

#ifdef _IN_TREE_BUILD
#include <linux/sched/cputime.h>
#include <kernel/sched/sched.h>
#include <kernel/sched/autogroup.h>
#endif
