from wa.utils.misc import write_table


def diff_interrupt_files(before, after, result):  # pylint: disable=R0914
    output_lines = []
    with open(before) as bfh:
        with open(after) as ofh:
            for bline, aline in izip(bfh, ofh):
                bchunks = bline.strip().split()
                while True:
                    achunks = aline.strip().split()
                    if achunks[0] == bchunks[0]:
                        diffchunks = ['']
                        diffchunks.append(achunks[0])
                        diffchunks.extend([diff_tokens(b, a) for b, a
                                           in zip(bchunks[1:], achunks[1:])])
                        output_lines.append(diffchunks)
                        break
                    else:  # new category appeared in the after file
                        diffchunks = ['>'] + achunks
                        output_lines.append(diffchunks)
                        try:
                            aline = ofh.next()
                        except StopIteration:
                            break

    # Offset heading columns by one to allow for row labels on subsequent
    # lines.
    output_lines[0].insert(0, '')

    # Any "columns" that do not have headings in the first row are not actually
    # columns -- they are a single column where space-spearated words got
    # split. Merge them back together to prevent them from being
    # column-aligned by write_table.
    table_rows = [output_lines[0]]
    num_cols = len(output_lines[0])
    for row in output_lines[1:]:
        table_row = row[:num_cols]
        table_row.append(' '.join(row[num_cols:]))
        table_rows.append(table_row)

    with open(result, 'w') as wfh:
        write_table(table_rows, wfh)


def diff_sysfs_dirs(before, after, result):  # pylint: disable=R0914
    before_files = []
    os.path.walk(before,
                 lambda arg, dirname, names: arg.extend([os.path.join(dirname, f) for f in names]),
                 before_files
                 )
    before_files = filter(os.path.isfile, before_files)
    files = [os.path.relpath(f, before) for f in before_files]
    after_files = [os.path.join(after, f) for f in files]
    diff_files = [os.path.join(result, f) for f in files]

    for bfile, afile, dfile in zip(before_files, after_files, diff_files):
        if not os.path.isfile(afile):
            logger.debug('sysfs_diff: {} does not exist or is not a file'.format(afile))
            continue

        with open(bfile) as bfh, open(afile) as afh:  # pylint: disable=C0321
            with open(_f(dfile), 'w') as dfh:
                for i, (bline, aline) in enumerate(izip_longest(bfh, afh), 1):
                    if aline is None:
                        logger.debug('Lines missing from {}'.format(afile))
                        break
                    bchunks = re.split(r'(\W+)', bline)
                    achunks = re.split(r'(\W+)', aline)
                    if len(bchunks) != len(achunks):
                        logger.debug('Token length mismatch in {} on line {}'.format(bfile, i))
                        dfh.write('xxx ' + bline)
                        continue
                    if ((len([c for c in bchunks if c.strip()]) == len([c for c in achunks if c.strip()]) == 2) and
                            (bchunks[0] == achunks[0])):
                        # if there are only two columns and the first column is the
                        # same, assume it's a "header" column and do not diff it.
                        dchunks = [bchunks[0]] + [diff_tokens(b, a) for b, a in zip(bchunks[1:], achunks[1:])]
                    else:
                        dchunks = [diff_tokens(b, a) for b, a in zip(bchunks, achunks)]
                    dfh.write(''.join(dchunks))
