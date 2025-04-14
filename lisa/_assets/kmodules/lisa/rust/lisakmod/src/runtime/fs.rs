/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{ffi::CString, vec::Vec};
use core::{
    cmp::max,
    ffi::{CStr, c_int},
    ptr::NonNull,
};

use embedded_io::Read;
use lisakmod_macros::inlinec::{NegativeError, PtrError, cconstant, cfunc, opaque_type};

use crate::{
    error::{Error, error},
    runtime::printk::pr_err,
};

pub enum OpenFlags {
    ReadOnly,
    WriteOnly,
}

impl From<OpenFlags> for c_int {
    fn from(flags: OpenFlags) -> c_int {
        match flags {
            OpenFlags::ReadOnly => cconstant!("#include <linux/fs.h>", "O_RDONLY").unwrap(),
            OpenFlags::WriteOnly => cconstant!("#include <linux/fs.h>", "O_WRONLY").unwrap(),
        }
    }
}

opaque_type!(
    struct CFile,
    "struct file",
    "linux/fs.h",
);

pub struct File {
    c_file: NonNull<CFile>,
    pos: usize,
    path: CString,
}

impl File {
    pub fn open(path: &str, flags: OpenFlags, _mode: u32) -> Result<File, Error> {
        // kernel_file_open() would be more appropriate for in-kernel use, as filp_open() opens in
        // the context of the current userspace thread. It's somewhat ok since we only open files
        // during module init, and this runs as root anyway.
        #[cfunc]
        fn filp_open(path: &CStr, flags: c_int, mode: u32) -> Result<NonNull<CFile>, PtrError> {
            r#"
            #include <linux/fs.h>
            #include "introspection.h"
            "#;

            r#"
            #if HAS_KERNEL_FEATURE(FILE_IO)
                return filp_open(path, flags, mode);
            #else
                return ERR_PTR(-ENOSYS);
            #endif
            "#
        }
        let path: CString = CString::new(path)
            .map_err(|err| error!("Could not convert path {path} to CString: {err}"))?;
        let c_file = filp_open(path.as_c_str(), flags.into(), 0)
            .map_err(|err| error!("Could not open file at {path:?}: {err}"))?;
        Ok(File {
            c_file,
            pos: 0,
            path,
        })
    }

    pub fn read_to_end(&mut self) -> Result<Vec<u8>, Error> {
        let increment = crate::runtime::alloc::PAGE_SIZE;
        let mut buf = Vec::new();
        let mut idx = 0;
        loop {
            let _buf = {
                let len = buf.len();
                if idx >= len {
                    let _increment = max(increment, idx - len);
                    buf.resize(len + _increment, 0);
                }
                &mut buf[idx..]
            };
            match self.read(_buf) {
                // Reached EOF
                Ok(0) => {
                    buf.resize(idx, 0);
                    return Ok(buf);
                }
                Ok(read) => {
                    idx += read;
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }
    }
}

unsafe impl Send for File {}

impl Drop for File {
    fn drop(&mut self) {
        #[cfunc]
        unsafe fn filp_close(file: NonNull<CFile>) -> Result<(), c_int> {
            r#"
            #include <linux/fs.h>
            #include "introspection.h"
            "#;

            r#"
            #if HAS_KERNEL_FEATURE(FILE_IO)
                return filp_close(file, 0);
            #else
                return -ENOSYS;
            #endif
            "#
        }
        unsafe {
            filp_close(self.c_file)
                .map_err(|err| pr_err!("Could not close file: {err}"))
                .expect("Failed to close file");
        }
    }
}

impl embedded_io::ErrorType for File {
    type Error = Error;
}

impl embedded_io::Read for File {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        #[cfunc]
        unsafe fn kernel_read(
            file: NonNull<CFile>,
            buf: *mut u8,
            count: usize,
            pos: usize,
        ) -> Result<usize, NegativeError<usize>> {
            r#"
            #include <linux/fs.h>
            #include "introspection.h"
            "#;

            r#"
            #if HAS_KERNEL_FEATURE(FILE_IO)
                loff_t off = pos;
                return kernel_read(file, buf, count, &off);
            #else
                return -ENOSYS;
            #endif
            "#
        }

        let count = buf.len();
        if count == 0 {
            Err(error!("Tried to read into an empty buffer"))
        } else {
            let ptr = buf.as_mut_ptr();
            let read = unsafe { kernel_read(self.c_file, ptr, count, self.pos) }
                .map_err(|err| error!("Could not read file {:?}: {err}", self.path))?;
            self.pos += read;
            Ok(read)
        }
    }
}

impl embedded_io::Write for File {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        #[cfunc]
        unsafe fn kernel_write(
            file: NonNull<CFile>,
            buf: *const u8,
            size: usize,
            pos: usize,
        ) -> Result<usize, NegativeError<usize>> {
            r#"
            #include <linux/fs.h>
            #include "introspection.h"
            "#;

            r#"
            #if HAS_KERNEL_FEATURE(FILE_IO)
                loff_t off = pos;
                return kernel_write(file, buf, size, &off);
            #else
                return -ENOSYS;
            #endif
            "#
        }

        let count = buf.len();
        let ptr = buf.as_ptr();
        let read = unsafe { kernel_write(self.c_file, ptr, count, self.pos) }
            .map_err(|err| error!("Could not write file {:?}: {err}", self.path))?;
        self.pos += read;
        Ok(read)
    }

    #[inline]
    fn flush(&mut self) -> Result<(), Self::Error> {
        #[cfunc]
        unsafe fn flush(file: NonNull<CFile>) -> Result<(), c_int> {
            r#"
            #include <linux/fs.h>
            #include "introspection.h"
            "#;

            r#"
            #if HAS_KERNEL_FEATURE(FILE_IO)
                if (file->f_op->flush) {
                    return file->f_op->flush(file, NULL);
                } else {
                    return 0;
                }
            #else
                return -ENOSYS;
            #endif
            "#
        }
        unsafe { flush(self.c_file) }
            .map_err(|err| error!("Could not flush file {:?}: {err}", self.path))
    }
}

impl embedded_io::Seek for File {
    #[inline]
    fn seek(&mut self, pos: embedded_io::SeekFrom) -> Result<u64, Self::Error> {
        match pos {
            embedded_io::SeekFrom::Start(start) => {
                self.pos = start.try_into().unwrap();
                Ok(start)
            }
            embedded_io::SeekFrom::Current(delta) => {
                let cur: i64 = self.pos.try_into().unwrap();
                let new = cur + delta;
                self.pos = new.try_into().unwrap();
                Ok(new.try_into().unwrap())
            }
            embedded_io::SeekFrom::End(_end) => Err(error!("Seek from end is not supported")),
        }
    }
}
